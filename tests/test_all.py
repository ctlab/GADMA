import unittest

from .test_data import YRI_CEU_DATA
from .test_cli import get_settings_test
from gadma import *
import gadma
from gadma.core import SharedDictForCoreRun
from gadma.utils import StdAndFileLogger
import dadi
import copy
import pickle
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

    def test_bo_restore(self):
        bo = get_global_optimizer("GPyOpt_Bayesian_optimization")
        f = rosenbrock
        variables = [ContinuousVariable('var1', [1e-15, 10]),
                     ContinuousVariable('var2', [1e-15, 2])]
        save_file = "save_file"
        report_file = "report_file"
        res1 = bo.optimize(f, variables, maxiter=5, verbose=1,
                           report_file=report_file,
                           save_file=save_file)

        res2 = bo.optimize(f, variables, maxiter=10, verbose=1,
                           report_file=report_file,
                           restore_file=save_file)

        res3 = bo.optimize(f, variables, maxiter=5, verbose=1,
                           report_file=report_file,
                           restore_file=save_file)

        self.assertEqual(res1.y, res3.y, msg=f"{res1}\n{res3}")
        self.assertTrue(res1.y >= res2.y, msg=f"{res1}\n{res2}")

    def test_ls_restore(self):
        for opt in all_local_optimizers():
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
        base_out_dir = "test_gs_and_ls_restore_output"
        sys.argv = ['gadma', '-p', param_file, '-o', base_out_dir]
        settings, _ = get_settings_test()
        settings.linked_snp_s = False
        settings.silence = True
        out_dir = 'some_not_existed_dir'
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

            self.assertTrue(res3.y >= res2.y)
        if os.path.exists(settings.output_directory):
            rmdir(settings.output_directory)

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
                     "local_maxiter: 1\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file,
                    '--output', outdir]
        try:
            gadma.matplotlib_available = False
            core.main()
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)
            gadma.matplotlib_available = True

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
                     "local_maxiter: 1\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file,
                    '--only_models']
        try:
            gadma.PIL_available = False
            core.main()
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
                     "local_maxiter: 1\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            core.main()
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
                     "local_maxiter: 1\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            gadma.PIL_available = False
            gadma.moments_available = False
            core.main()
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)
            gadma.PIL_available = True
            gadma.moments_available = True

    def test_restore_with_different_options_failure(self):
        finished_run_dir = os.path.join(DATA_PATH, 'my_example_run')
        params_file = 'params'
        with open(params_file, 'w') as fl:
            fl.write("Stuck generation number: 2\n"
                     "Projections: 4,4\n"
                     "Initial structure: 1, 1\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxiter: 1\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            gadma.PIL_available = False
            gadma.moments_available = False
            self.assertRaises(ValueError, core.main)
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)
            gadma.PIL_available = True
            gadma.moments_available = True

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
                     "local_maxiter: 1\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]

        try:
            settings, _ = get_settings_test()
            core.main()

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
                     "local_maxiter: 1\n")
        sys.argv = ['gadma', '--resume', finished_run_dir, '-p', params_file]
        try:
            gadma.PIL_available = False
            gadma.moments_available = False
            self.assertRaises(ValueError, core.main)
        finally:
            if check_dir_existence(finished_run_dir + '_resumed'):
                shutil.rmtree(finished_run_dir + '_resumed')
            os.remove(params_file)
            gadma.PIL_available = True
            gadma.moments_available = True
