import unittest
from unittest import mock

from .test_data import YRI_CEU_DATA
from .test_cli import get_settings_test, check_output_files
from gadma import *
import gadma
from gadma.core import SharedDictForCoreRun
from gadma.utils import StdAndFileLogger
import dadi
import copy
import pickle
from pathlib import Path
import shutil
import os
import sys

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

def rmdir(dirname):
    if not os.path.exists(dirname):
        return
    for filename in os.listdir(dirname):
        path = os.path.join(dirname, filename)
        if os.path.isdir(path):
            rmdir(path)
        else:
            os.remove(path)
    os.rmdir(dirname)


def rosenbrock(X):
    """
    This R^2 -> R^1 function should be compatible with algopy.
    http://en.wikipedia.org/wiki/Rosenbrock_function
    A generalized implementation is available
    as the scipy.optimize.rosen function
    """
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.


class TestRestore(unittest.TestCase):
    def tearDown(self):
        for filename in ["./save_file", "./history_save_file",
                         "./report_file"]:
            if Path(filename).exists():
                os.remove(filename)

    def test_ga_restore(self):
        ga = get_global_optimizer("Genetic_algorithm")
        f = rosenbrock
        variables = [ContinuousVariable('var1', [-1, 2]),
                     ContinuousVariable('var2', [-2, 3])]
        save_file = "save_file"
        report_file = "report_file"
        res1 = ga.optimize(f, variables, maxiter=5, verbose=1,
                           report_file=report_file,
                           save_file=save_file)

        res2 = ga.optimize(f, variables, maxiter=10, verbose=1,
                           report_file=report_file,
                           restore_file=save_file)

        res3 = ga.optimize(f, variables, maxiter=5, verbose=1,
                           report_file=report_file,
                           restore_file=save_file)

        self.assertEqual(res1.y, res3.y, msg=f"{res1}\n{res3}")
        self.assertTrue(res1.y >= res2.y, msg=f"{res1}\n{res2}")

    def test_evaluations_of_resumed_run(self):
        """
        Resume of the terminated run continues it: evaluations that were
        made before are counted and the run has what is left of `maxeval`.
        If only points of the previous run are restored then evaluations
        are counted from zero.
        """
        class Terminated(Exception):
            pass

        def get_f_and_calls():
            calls = []

            def f(x):
                calls.append(x)
                return rosenbrock(x)
            return f, calls

        # positive domain because we check log scaling optimizations too
        variables = [ContinuousVariable('var1', [0.1, 20]),
                     ContinuousVariable('var2', [0.1, 20])]
        save_file = "save_file"
        maxeval = 60
        optimizers = [(get_global_optimizer("Genetic_algorithm"), {}),
                      (get_local_optimizer("BFGS_log"), {"x0": [15, 0.5]}),
                      (get_local_optimizer("L-BFGS-B"), {"x0": [15, 0.5]})]
        for opt, kwargs in optimizers:
            # run is terminated after 20 evaluations
            f, calls = get_f_and_calls()

            def callback(x, y):
                if len(calls) >= 20:
                    raise Terminated()

            self.assertRaises(Terminated, opt.optimize, f, variables,
                              maxeval=maxeval, save_file=save_file,
                              callback=callback, **kwargs)
            n_eval = opt.load(save_file).result.n_eval
            self.assertTrue(0 < n_eval <= len(calls), msg=f"{opt.id}")

            # the run is continued
            f, calls = get_f_and_calls()
            res = opt.optimize(f, variables, maxeval=maxeval,
                               restore_file=save_file, **kwargs)
            msg = f"{opt.id}: {n_eval} evaluations were saved, "\
                  f"{len(calls)} new ones, result has {res.n_eval}."
            self.assertEqual(res.n_eval, n_eval + len(calls), msg=msg)
            self.assertTrue(res.n_eval <= maxeval, msg=msg)
            if opt.id == "Genetic_algorithm":
                self.assertEqual(res.n_eval, maxeval, msg=msg)

            # only points are taken from the previous run
            f, calls = get_f_and_calls()
            res = opt.optimize(f, variables, maxeval=maxeval,
                               restore_file=save_file,
                               restore_points_only=True, **kwargs)
            msg = f"{opt.id}: {len(calls)} new evaluations, result has "\
                  f"{res.n_eval}."
            self.assertEqual(res.n_eval, len(calls), msg=msg)
            self.assertTrue(res.n_eval <= maxeval, msg=msg)
            if opt.id == "Genetic_algorithm":
                self.assertEqual(res.n_eval, maxeval, msg=msg)

    def test_history_of_saved_run(self):
        """
        The run is saved after every iteration. Its history (all evaluated
        points and their values) is kept in another file than the rest of
        information: only new points are written there, the whole history
        is not rewritten at each save.
        """
        import numpy as np
        from gadma.utils import serialize_meta_array, ident_transform

        def assert_same_run(opt, run_info, other_run_info, msg):
            res1, res2 = run_info.result, other_run_info.result
            self.assertEqual(len(res1.X), len(res2.X), msg=msg)
            self.assertEqual(list(res1.Y), list(res2.Y), msg=msg)
            for list1, list2 in [(res1.X, res2.X), (res1.X_out, res2.X_out)]:
                for x1, x2 in zip(list1, list2):
                    self.assertTrue(np.array_equal(x1, x2), msg=msg)
            self.assertEqual(list(res1.Y_out), list(res2.Y_out), msg=msg)
            self.assertTrue(np.array_equal(res1.x, res2.x), msg=msg)
            for name in ["y", "n_eval", "n_iter", "success"]:
                self.assertEqual(getattr(res1, name), getattr(res2, name),
                                 msg=f"{name}. {msg}")

        variables = [ContinuousVariable('var1', [0.1, 20]),
                     ContinuousVariable('var2', [0.1, 20])]
        save_file = "save_file"
        optimizers = [(get_global_optimizer("Genetic_algorithm"), {}),
                      (get_local_optimizer("BFGS_log"), {"x0": [15, 0.5]}),
                      (get_local_optimizer("L-BFGS-B"), {"x0": [15, 0.5]})]
        for opt, kwargs in optimizers:
            msg = f"optimizer {opt.id}"
            history_file = opt.get_history_file(save_file)
            for filename in [save_file, history_file]:
                if os.path.exists(filename):
                    os.remove(filename)
            sizes = []

            def callback(x, y):
                # saved run is the run of this moment
                assert_same_run(opt, opt.load(save_file), opt.run_info, msg)
                sizes.append((os.path.getsize(save_file),
                              os.path.getsize(history_file)))

            res = opt.optimize(rosenbrock, variables, maxiter=30,
                               save_file=save_file, callback=callback,
                               **kwargs)
            self.assertTrue(len(res.X) > 30, msg=msg)
            assert_same_run(opt, opt.load(save_file), opt.run_info, msg)
            # points are added to the history, the other file is small
            self.assertTrue(sizes[-1][1] > 3 * sizes[2][1], msg=msg)
            self.assertTrue(sizes[-1][0] < 2 * sizes[2][0], msg=msg)

            # writing of the history was stopped in the middle
            with open(history_file, "ab") as fl:
                fl.write(pickle.dumps(("some", "record"))[:10])
            assert_same_run(opt, opt.load(save_file), opt.run_info, msg)
            # Next run with this file. Nothing is removed from the file of
            # history, new records are written after the old ones.
            run_info = opt.run_info
            with open(history_file, "rb") as fl:
                content = fl.read()
            res = opt.optimize(rosenbrock, variables, maxiter=5,
                               save_file=save_file, **kwargs)
            assert_same_run(opt, opt.load(save_file), opt.run_info, msg)
            with open(history_file, "rb") as fl:
                new_content = fl.read()
            self.assertTrue(len(new_content) > len(content), msg=msg)
            self.assertEqual(new_content[:len(content)], content, msg=msg)

            # save file of the previous versions has the history inside
            info = opt._apply_transform_to_run_info(
                run_info, x_transform=serialize_meta_array,
                y_transform=ident_transform)
            os.remove(history_file)
            with open(save_file, "wb") as fl:
                pickle.dump({opt.id: info}, fl)
            self.assertTrue(opt.valid_restore_file(save_file), msg=msg)
            assert_same_run(opt, opt.load(save_file), run_info, msg)

    def test_save_file_after_stop_during_saving(self):
        """
        The save file of the previous iteration is kept when the run is
        stopped during the writing of the new one.
        """
        ga = get_global_optimizer("Genetic_algorithm")
        variables = [ContinuousVariable('var1', [-1, 2]),
                     ContinuousVariable('var2', [-2, 3])]
        save_file = "save_file"
        ga.optimize(rosenbrock, variables, maxiter=3, save_file=save_file)
        with open(save_file, "rb") as fl:
            content = fl.read()
        files_before = set(os.listdir("."))

        def stopped_dump(obj, fl, *args, **kwargs):
            fl.write(pickle.dumps(obj)[:10])
            raise KeyboardInterrupt()

        try:
            with mock.patch("pickle.dump", side_effect=stopped_dump):
                self.assertRaises(KeyboardInterrupt, ga.save, ga.run_info,
                                  save_file)
            with open(save_file, "rb") as fl:
                self.assertEqual(fl.read(), content)
            self.assertTrue(ga.valid_restore_file(save_file))
        finally:
            for filename in set(os.listdir(".")) - files_before:
                os.remove(filename)

    def test_ga_restore_after_other_optimizer(self):
        """
        Genetic algorithm continues the terminated run of Bayesian
        optimization: it takes its points and count of evaluations.
        """
        import numpy as np
        from gadma.optimizers import GlobalOptimizer, GeneticAlgorithm
        from gadma.optimizers import global_optimizer
        from gadma.optimizers.global_optimizer import \
            register_global_optimizer
        from gadma.optimizers.bayesian_optimization import \
            SMACBOKernelCombination
        bo_id = "SMAC_BO_combination"
        self.assertTrue(
            bo_id in GeneticAlgorithm.ga_could_be_restored_after)

        class BOWithoutSMAC(SMACBOKernelCombination):
            """Run information is the same and SMAC is not required."""
            def __init__(self):
                GlobalOptimizer.__init__(self)

        registered = global_optimizer._registered_global_optimizers
        is_registered = bo_id in registered
        if not is_registered:
            register_global_optimizer(bo_id, BOWithoutSMAC)
        try:
            variables = [ContinuousVariable('var1', [-1, 2]),
                         ContinuousVariable('var2', [-2, 3])]
            save_file = "save_file"
            # save file of the terminated run of other optimizer
            bo = get_global_optimizer(bo_id)
            X = [[var.resample() for var in variables] for _ in range(20)]
            Y = [rosenbrock(x) for x in X]
            run_info = bo.run_info
            for i in [0, 10]:
                i_best = int(np.argmin(Y[:i + 10]))
                run_info = bo._update_run_info(
                    run_info, x_best=X[i_best], y_best=Y[i_best],
                    X=X[i: i + 10], Y=Y[i: i + 10], n_eval=10)
            self.assertFalse(run_info.result.success)
            bo.save(run_info, save_file)

            calls = []

            def f(x):
                calls.append(x)
                return rosenbrock(x)

            ga = get_global_optimizer("Genetic_algorithm")
            self.assertTrue(ga.valid_restore_file(save_file))
            maxeval = 60
            res = ga.optimize(f, variables, maxeval=maxeval,
                              restore_file=save_file, save_file=save_file)
            self.assertEqual(len(calls), maxeval - 20)
            self.assertEqual(res.n_eval, maxeval)
            self.assertTrue(res.y <= min(Y))
            # the next resume is the usual resume of genetic algorithm
            self.assertTrue(ga.valid_restore_file(save_file))
            self.assertEqual(ga.load(save_file).result.n_eval, maxeval)
        finally:
            if not is_registered:
                del registered[bo_id]

#    def test_bo_restore(self):
#        bo = get_global_optimizer("GPyOpt_Bayesian_optimization")
#        f = rosenbrock
#        variables = [ContinuousVariable('var1', [1e-15, 10]),
#                     ContinuousVariable('var2', [1e-15, 2])]
#        save_file = "save_file"
#        report_file = "report_file"
#        res1 = bo.optimize(f, variables, maxiter=5, verbose=1,
#                           report_file=report_file,
#                           save_file=save_file)
#
#        res2 = bo.optimize(f, variables, maxiter=10, verbose=1,
#                           report_file=report_file,
#                           restore_file=save_file)
#
#        res3 = bo.optimize(f, variables, maxiter=5, verbose=1,
#                           report_file=report_file,
#                           restore_file=save_file)
#
#        self.assertEqual(res1.y, res3.y, msg=f"{res1}\n{res3}")
#        self.assertTrue(res1.y >= res2.y, msg=f"{res1}\n{res2}")

    def test_ls_restore(self):
        for opt in all_local_optimizers():
            print(opt.id)
            def f(x):
                y = rosenbrock(x)
                return y
            # positive domain because we check log scaling optimizations too
            variables = [ContinuousVariable('var1', [10, 20]),
                         ContinuousVariable('var2', [1, 2])]
            x0 = [var.resample() for var in variables]
            save_file = "save_file"
            report_file = "report_file"
            res1 = opt.optimize(f, variables, x0=x0, maxiter=5, verbose=1,
                                report_file=report_file,
                                save_file=save_file)
            res2 = opt.optimize(f, variables, x0=x0, maxiter=5, verbose=1,
                                report_file=report_file,
                                restore_file=save_file,
                                restore_points_only=True)
            res3 = opt.optimize(f, variables, x0=x0, maxiter=5, verbose=1,
                                report_file=report_file,
                                restore_file=save_file)
            res4 = opt.optimize(f, variables, x0=x0, maxiter=10, verbose=1,
                                report_file=report_file,
                                restore_file=save_file,
                                restore_points_only=True)
            self.assertEqual(res1.y, res3.y)
            self.assertTrue(res1.y >= res2.y)
            self.assertTrue(res2.y >= res4.y)
            for res in [res1, res2, res3, res4]:
                self.assertEqual(res.y, f(res.x))

    def test_gs_and_ls_restore(self):
        param_file = os.path.join(DATA_PATH, "PARAMS", 'another_test_params')
        base_out_dir = os.path.join(DATA_PATH, "test_gs_and_ls_restore_output")
        sys.argv = ['gadma', '-p', param_file, '-o', base_out_dir]
        settings, _ = get_settings_test()
        settings.linked_snp_s = False
        settings.silence = True
        out_dir = os.path.join(DATA_PATH, 'some_not_existed_dir')
        shared_dict = gadma.shared_dict.SharedDictForCoreRun(multiprocessing=False)
        if os.path.exists(out_dir):
            rmdir(out_dir)
        for ls_opt in ["None", "BFGS_log", "Powell"]: #all_local_optimizers():
            if os.path.exists(settings.output_directory):
                rmdir(settings.output_directory)
            if os.path.exists(out_dir):
                rmdir(out_dir)
            settings.local_optimizer = ls_opt #.id
            settings.local_maxiter = 1
            settings.silence = True
            core_run = CoreRun(0, shared_dict, settings)
            res1 = core_run.run()

            restore_settings = copy.copy(settings)
            restore_settings.output_directory = out_dir
            restore_settings.resume_from = settings.output_directory

            restore_core_run = CoreRun(0, shared_dict, restore_settings)
            res2 = restore_core_run.run()

            self.assertEqual(res1.y, res2.y)
            if os.path.exists(out_dir):
                rmdir(out_dir)

            restore_settings.only_models = True
            restore_core_run = CoreRun(0, shared_dict, restore_settings)
            res3 = restore_core_run.run()

            self.assertTrue(res3.y >= res2.y, f"{ls_opt}: {res3.y} < {res2.y}")
        if os.path.exists(settings.output_directory):
            rmdir(settings.output_directory)
        if os.path.exists(out_dir):
            rmdir(out_dir)

    def test_restore_finished_run(self):
        finished_run_dir = os.path.join(DATA_PATH, 'my_example_run')
        # Check for save_files
        ga = get_global_optimizer("Genetic_algorithm")
        for i in range(3):
            save_file_1 = os.path.join(finished_run_dir, str(i+1),
                                       "save_file_1_1")
            save_file_2 = os.path.join(finished_run_dir, str(i+1),
                                       "save_file_2_1")

            self.assertTrue(ga.valid_restore_file(save_file_1))
            self.assertTrue(ga.valid_restore_file(save_file_2))

        params_file = 'params'
        outdir = os.path.join(DATA_PATH, 'resume_dir')
        if check_dir_existence(outdir):
            shutil.rmtree(outdir)
        with open(params_file, 'w') as fl:
            fl.write("Linked SNP's: False\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxeval: 10\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file,
                    '--output', outdir]
        try:
            core.main()
            check_output_files(
                test=self,
                outdir=outdir,
                engine="moments",
                num_runs=3,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=True,
                draw_model_iter=True,
                mu_L_available=False,
                aic_or_claic="aic",
            )
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)

    def test_restore_without_file_for_initial_structure(self):
        """
        Run is not resumed when the saved file for the initial structure is
        missing: the error is raised before any optimization.
        """
        import tempfile
        finished_run_dir = os.path.join(DATA_PATH, 'my_example_run')
        dirname = tempfile.mkdtemp()
        resume_dir = os.path.join(dirname, 'my_example_run')
        outdir = os.path.join(dirname, 'resumed')
        shutil.copytree(finished_run_dir, resume_dir)
        # initial structure is [2, 1] and final one is [3, 1]
        os.remove(os.path.join(resume_dir, "1", "save_file_2_1"))
        sys.argv = ['gadma', '--resume', resume_dir, '--output', outdir]
        try:
            settings, _ = get_settings_test()
            settings.silence = True
            shared_dict = gadma.shared_dict.SharedDictForCoreRun(
                multiprocessing=False)
            core_run = CoreRun(1, shared_dict, settings)
            with self.assertRaises(ValueError) as context:
                core_run.run()
            message = str(context.exception)
            self.assertTrue("initial structure is [2, 1]" in message,
                            msg=message)
            self.assertTrue("[3, 1]" in message, msg=message)
            # the run with all files is resumed
            core_run = CoreRun(2, shared_dict, settings)
            core_run.run()
        finally:
            shutil.rmtree(dirname)

    def test_restore_models_from_finished_run(self):
        finished_run_dir = os.path.join(DATA_PATH, 'my_example_run')
        params_file = 'params'
        with open(params_file, 'w') as fl:
            fl.write("Stuck generation number: 2\n"
                     "Only models: True\n"
                     "Projections: [4,4]\n"
                     "Theta0: 1\n"
                     "Relative parameters: True\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxeval: 10\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file,
                    '--only_models']
        try:
            gadma.PIL_available = False
            core.main()
            check_output_files(
                test=self,
                outdir=finished_run_dir + '_resumed',
                engine="moments",
                num_runs=3,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=True,
                draw_model_iter=True,
                mu_L_available=False,
                aic_or_claic=None,
            )
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)
            gadma.PIL_available = True

    def test_restore_with_different_options_1(self):
        finished_run_dir = os.path.join(DATA_PATH, 'my_example_run')
        params_file = 'params'
        with open(params_file, 'w') as fl:
            fl.write("Stuck generation number: 2\n"
                     "Symmetric migrations: True\n"
                     "Only sudden: True\n"
                     "Theta0: None\n"
                     "Mutation rate: 1.25e-8\n"
                     "Sequence length: 4.04e6\n"
                     "Split fractions: False\n"
                     "Projections: 4,4\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxeval: 10\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            core.main()
            check_output_files(
                test=self,
                outdir=finished_run_dir + '_resumed',
                engine="moments",
                num_runs=3,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=True,
                draw_model_iter=True,
                mu_L_available=True,
                aic_or_claic=None,
            )
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
        try:
            settings, _ = gadma.cli.arg_parser.get_settings()
            shared_dict = SharedDictForCoreRun(multiprocessing=False)
            obj = CoreRun(1, shared_dict, settings)
            obj.run(settings.get_optimizers_init_kwargs())
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)

    def test_restore_with_different_options_2(self):
        finished_run_dir = os.path.join(DATA_PATH, 'my_example_run')
        params_file = 'params'
        with open(params_file, 'w') as fl:
            fl.write("Stuck generation number: 2\n"
                     "Engine: dadi\n"
                     "Projections: 4,4\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxeval: 10\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            core.main()
            check_output_files(
                test=self,
                outdir=finished_run_dir + '_resumed',
                engine="dadi",
                num_runs=3,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=True,
                draw_model_iter=True,
                mu_L_available=False,
                aic_or_claic=None,
            )
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)

    def test_restore_with_different_options_failure(self):
        finished_run_dir = os.path.join(DATA_PATH, 'my_example_run')
        params_file = 'params'
        with open(params_file, 'w') as fl:
            fl.write("Stuck generation number: 2\n"
                     "Projections: 4,4\n"
                     "Initial structure: 1, 1\n" # is not allowed to be changed
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxeval: 10\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            self.assertRaises(ValueError, core.main)
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)

    def test_restore_with_new_migration_masks(self):
        finished_run_dir = os.path.join(DATA_PATH,
                                        'my_example_run_one_structure')
        params_file = 'params'
        output_3 = "some_output"
        with open(params_file, 'w') as fl:
            fl.write("Stuck generation number: 2\n"
                     "Projections: 4,4\n"
                     "Migration masks: [[0, 1], [0, 0]]\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxeval: 10\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]

        try:
            settings, _ = get_settings_test()
            core.main()
            check_output_files(
                test=self,
                outdir=finished_run_dir + '_resumed',
                engine="moments",
                num_runs=3,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=True,
                draw_model_iter=True,
                mu_L_available=False,
                aic_or_claic=None,
            )

            # call corerun for cover case when there is o extra file in resume
            shared_dict = SharedDictForCoreRun(multiprocessing=False)
            core_run = CoreRun(0, shared_dict, settings)
            core_run.get_run_options()

            with open(params_file, 'w') as fl:
                fl.write("Migration masks: None\n"
                         "time_to_print_summary: 0.016\n")
            sys.argv = ['gadma', '--resume', finished_run_dir + "_resumed", 
                        '-p', params_file]
            core.main()
            check_output_files(
                test=self,
                outdir=finished_run_dir + '_resumed_resumed',
                engine="moments",
                num_runs=3,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=True,
                draw_model_iter=True,
                mu_L_available=False,
                aic_or_claic=None,
            )

            with open(params_file, 'w') as fl:
                fl.write("Migration masks: [[0]]\n")
            sys.argv = ['gadma', '--resume', finished_run_dir + "_resumed", 
                        '-p', params_file, '-o', output_3]
            self.assertRaises(ValueError, core.main)

            if check_dir_existence(output_3):
                shutil.rmtree(output_3)

            with open(params_file, 'w') as fl:
                fl.write("Symmetric migrations: True\n"
                         "Migration masks: [[[0, 0], [1, 0]]]\n")
            sys.argv = ['gadma', '--resume', finished_run_dir + "_resumed", 
                        '-p', params_file, '-o', output_3]
            self.assertRaises(ValueError, get_settings_test)

            if check_dir_existence(output_3):
                shutil.rmtree(output_3)

            with open(params_file, 'w') as fl:
                fl.write("Migration masks: [[[0, 0, 0], [1, 0, 0]]]\n")
            sys.argv = ['gadma', '--resume', finished_run_dir + "_resumed", 
                        '-p', params_file, '-o', output_3]
            self.assertRaises(ValueError, get_settings_test)

        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            if check_dir_existence(finished_run_dir + '_resumed_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed_resumed')
            if check_dir_existence(output_3):
                shutil.rmtree(output_3)

            os.remove(params_file)

    def test_restore_failure_with_new_migration_masks(self):
        finished_run_dir = os.path.join(DATA_PATH,
                                        'my_example_run')
        params_file = 'params'
        with open(params_file, 'w') as fl:
            fl.write("Stuck generation number: 2\n"
                     "Projections: 4,4\n"
                     "Migration masks: [[0, 1], [0, 0]]\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxeval: 10\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            self.assertRaises(ValueError, core.main)
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)
