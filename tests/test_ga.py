import unittest
from gadma import *
import numpy as np
import pytest
from .test_optimizers import get_func, get_1pop_sim_example_1
from .test_optimizers import get_1pop_sim_example_2, get_2pop_sim_example_1
from .test_data import YRI_CEU_DATA
import gadma
from gadma.utils import custom_generator, WeightedMetaArray
from gadma.optimizers import GeneticAlgorithm, register_global_optimizer
from gadma.data.data import VCFDataHolder
import functools
import importlib
import pickle
import os
import sys
import timeit
import types
import copy

EXAMPLE_FOLDER = os.path.join(os.path.dirname(__file__), "test_data")
EXAMPLE_DATA = os.path.join(EXAMPLE_FOLDER, "DATA", "sfs", "YRI_CEU.fs")
PREPROCESSED_DATA = os.path.join(EXAMPLE_FOLDER, "DATA", "vcf_ld", "preprocessed_data_YRI_CEU.bp")

EXAMPLE_DATA_LD = VCFDataHolder(
    vcf_file=None,
    popmap_file=None,
    population_labels=["YRI", "CEU"],
    preprocessed_data=PREPROCESSED_DATA
)


def remember_results(model_func):
    """
    Returns the same function of demographic model that remembers its
    results. Model is simulated a lot of times for the same values of
    parameters when CLAIC is evaluated several times.
    """
    results = {}

    @functools.wraps(model_func)
    def wrapper(params, ns, *args):
        key = (tuple(float(el) for el in params),
               tuple(int(el) for el in ns), repr(args))
        if key not in results:
            results[key] = model_func(params, ns, *args)
        return copy.deepcopy(results[key])
    return wrapper


class TestGlobalOptimizers(unittest.TestCase):
    def test_registered_global_optimizers_fails(self):
        self.assertRaises(ValueError, get_global_optimizer, 'some strange_id')
        ex_id = 'Genetic_algorithm'
        opt = get_global_optimizer(ex_id)
        self.assertRaises(ValueError, register_global_optimizer,
                          ex_id, opt.__class__)
        self.assertRaises(ValueError, register_global_optimizer, 'id_ok', list)


class TestGeneticAlg(unittest.TestCase):
    def check_diff(self, diff, ind, msg):
        self.assertTrue(not diff[ind], msg=f"Arrays are equal "
                                           f"at index {ind}. " + msg)
        diff[ind] = True
        self.assertTrue(np.all(diff), msg=f"Arrays are not equal at some"
                                          f" index except {ind}. " + msg)

    def test_initialization(self):
        get_global_optimizer('Genetic_algorithm')

        self.assertRaises(ValueError, GeneticAlgorithm,
                          random_type='custom', custom_rand_gen=None)

    def test_random(self):
        random_types = ['uniform', 'resample', 'custom']
        ga = get_global_optimizer('Genetic_algorithm')

        variables = []
        variables.append(PopulationSizeVariable('n', domain=[0, 1]))
        variables.append(MigrationVariable('m', domain=[0, 1]))
        variables.append(SelectionVariable('s', domain=[0, 1]))
        variables.append(FractionVariable('f', domain=[0, 1]))
        variables.append(TimeVariable('t', domain=[1e-15, 1]))
        variables.append(DynamicVariable('var_disc',
                                         domain=[2, 0, 'Exp']))
        variables[0].domain = [90, 100]

        for r_type in random_types:
            ga.random_type = r_type
            ga.custom_rand_gen = None
            if r_type == 'custom':
                self.assertRaises(TypeError, ga.randomize, variables,
                                  random_type=r_type, custom_rand_gen=None)
            else:
                ga.randomize(variables, random_type=r_type,
                             custom_rand_gen=None)
            ga.randomize(variables, random_type=r_type,
                         custom_rand_gen=custom_generator)
        self.assertRaises(ValueError, ga.randomize, variables,
                          random_type='some_unknown_type')

    def test_mutation(self):
        mut_types = ['uniform', 'gaussian', 'resample']
        ga = get_global_optimizer('Genetic_algorithm')
        ga.cur_mut_strength = 1.0

        n_var = 5
        variables = []
        for i in range(n_var - 1):
            variables.append(ContinuousVariable('var%d' % i, domain=[0, 1]))
        variables.append(DiscreteVariable('var_disc_%d' % i,
                                          domain=[2, 3, 'Ex']))
        variables[0].domain = [90, 100]
        x_list = [var.resample() for var in variables]
        x_list[1] = 1e-20
        x_arr = WeightedMetaArray(x_list)
        x_arr.__str__()

        x_mut = x_arr
        while x_mut[1] != 0:
            x_mut = ga.mutation_by_ind(x_arr, variables, 1,
                                       mutation_type='gaussian')
        for ind in range(n_var):
            for mut_type in mut_types:
                with self.subTest(mut_type=mut_type,
                                  mut='full_mutation'):
                    ga.mut_strength = 1.0
                    x_mut = ga.mutation(x_arr, variables, mut_type, True)
                    self.assertTrue(np.all(x_mut != x_arr), msg=str((x_mut, x_arr)))
                with self.subTest(mut_type=mut_type, index=ind,
                                  mut='mutation_by_ind'):
                    msg = f"(mut. type is {mut_type}, index is {ind})"
                    x_mut = ga.mutation_by_ind(x_arr, variables, ind,
                                               mutation_type=mut_type,
                                               one_fifth_rule=True)
                    self.assertTrue(isinstance(x_mut, WeightedMetaArray),
                                    msg=f"Mutation returned not WeightedMetaArray "
                                        f"object ({x_mut.__class__}). " + msg)

                    self.assertIsNot(x_arr, x_mut)
                    self.check_diff(x_arr == x_mut, ind, msg)
                    self.check_diff(x_arr.weights == x_mut.weights, ind, msg)
                    self.assertEqual((x_mut.weights - x_arr.weights)[ind], 1, msg=msg)

                    x_mut2 = ga.mutation_by_ind(x_list, variables, ind,
                                                mutation_type=mut_type,
                                                one_fifth_rule=False)
                    self.assertIsInstance(x_mut2, WeightedMetaArray, msg=msg)
                    self.assertTrue((x_mut2.weights == x_mut.weights).all(), msg=msg)
                    self.assertEqual(x_mut2.metadata[-1], 'm')

        for _ in range(3):
            x_arr = ga.mutation_by_ind(x_arr, variables, 2,
                                       mutation_type=mut_type,
                                       one_fifth_rule=False)
        self.assertEqual(x_arr.weights[2], 4, msg=msg)

        # fails
        for ind in range(n_var):
            self.assertRaises(ValueError, ga.mutation_by_ind, x_arr,
                              variables, ind, mutation_type='bad_type')
        self.assertRaises(ValueError, ga.mutation, x_arr,
                          variables, mutation_type='bad_type')

    def test_crossover(self):
        cross_types = ['uniform', 'k_point']
        ks = range(1, 5)
        ga = get_global_optimizer('Genetic_algorithm')

        n_var = 20
        variables = []
        for i in range(n_var - 1):
            variables.append(ContinuousVariable('var%d' % i, domain=[0, 1]))
        variables.append(DiscreteVariable('var_disc_%d' % i,
                                          domain=[2, 3, 'Ex']))
        variables[0].domain = [90, 100]
        par1 = [var.resample() for var in variables]
        par2 = [var.resample() for var in variables]

        for cross_type in cross_types:
            for k in ks:
                with self.subTest(cross_type=cross_type, k=k):
                    msg = f"(crossover type is {cross_type}, k is {k}.)"
                    ch = ga.crossover(par1, par2, variables,
                                      crossover_type=cross_type, k=k)
                    self.assertIsNot(ch, par1, msg=msg)
                    self.assertIsNot(ch, par2, msg=msg)
                    self.assertTrue(np.any(ch != par1), msg=msg)
                    self.assertTrue(np.any(ch != par2), msg=msg)

                    self.assertEqual(ch.metadata, 'c', msg=msg)

                    chs = ga.crossover(par1, par2, variables,
                                       crossover_type=cross_type, k=k,
                                       one_child=False)
                    self.assertEqual(len(chs), 2)

        # fails
        self.assertRaises(ValueError, ga.crossover, par1, par2,
                          variables, crossover_type='bad_type')

    def test_selection(self):
        def f(x):
            return np.sum(x)

        def f_inf(x):
            return np.inf

        sel_types = ['roulette_wheel', 'rank']
        ga = get_global_optimizer('Genetic_algorithm')

        n_var = 5
        variables = []
        for i in range(n_var):
            variables.append(ContinuousVariable('var%d' % i, domain=[0, 1]))
        variables[0].domain = [90, 100]
        X_gen = [[var.resample() for var in variables] for _ in range(20)]
        for sel_type in sel_types:
            X_gen_new, Y_gen_new = ga.selection(f, variables, X_gen,
                                                selection_type=sel_type)
            X_gen_new, Y_gen_new = ga.selection(f, variables, X_gen,
                                                selection_type=sel_type)
            X_gen_new, Y_gen_new = ga.selection(f_inf, variables, X_gen,
                                                selection_type=sel_type)
            X_gen_new, Y_gen_new = ga.selection(f_inf, variables, X_gen,
                                                selection_type=sel_type,
                                                selection_random=True)
        # fails
        self.assertRaises(ValueError, ga.selection, f,
                          variables, X_gen, selection_type='bad_type')

    def test_opt_without_report_file(self):
        def f(x):
            return np.sum(x)

        n_var = 5
        variables = []
        for i in range(n_var):
            variables.append(ContinuousVariable('var%d' % i, domain=[0, 1]))
        x = [[var.resample() for var in variables] for _ in range(20)]
        ga = get_global_optimizer("Genetic_algorithm")
        run_info = ga._create_run_info()
        run_info.result.x = []
        ga.write_report(variables, run_info, report_file=None)

    def run_example(self, engine_id, example_func, not_bayesopt=False):
        args = ()
        if engine_id == 'dadi':
            args = ([4, 6, 8],)
        f, variables = example_func(engine_id, args)

        report_file = 'report_file'
        save_file = 'save_file'  # TODO
        eval_file = 'eval_file'

        for opt in all_global_optimizers():
            if not_bayesopt and opt.id == 'GPyOpt_Bayesian_optimization':
                continue  # TODO categorical parameters
            with self.subTest(optimizer=opt.id):
                open(report_file, 'w').close()
                open(eval_file, 'w').close()
                res = opt.optimize(f, variables, args=args,
                                   num_init=2, maxeval=10, maxiter=4,
                                   report_file=report_file, verbose=1,
                                   save_file=save_file, eval_file=eval_file)
                self.assertEqual(res.y, f(res.x, *args))
                self.assertTrue(os.stat(eval_file).st_size > 0)
                self.assertTrue(os.stat(report_file).st_size > 0)
                int_lines = -1
                with open(eval_file) as fl:
                    for line in fl:
                        int_lines += 1
                # As GA stops when got maxeval n_eval could be higher
                self.assertTrue(int_lines <= 20)

       # def test_1pop_example_1(self):
       #     for engine in all_engines():
       #         with self.subTest(engine=engine.id):
       #             self.run_example(engine.id, get_1pop_sim_example_1)
       #
       # def test_1pop_example_2(self):
       #     for engine in all_engines():
       #         with self.subTest(engine=engine.id):
       #             self.run_example(engine.id, get_1pop_sim_example_2,
       #                              not_bayesopt=True)
       #
       # def test_2pop_example_1(self):
       #     for engine in all_engines():
       #         with self.subTest(engine=engine.id):
       #             self.run_example(engine.id, get_2pop_sim_example_1)
       #
       # def test_yri_ceu(self):
       #     nu1F = PopulationSizeVariable('nu1F')
       #     nu2B = PopulationSizeVariable('nu2B')
       #     nu2F = PopulationSizeVariable('nu2F')
       #     m = MigrationVariable('m')
       #     Tp = TimeVariable('Tp')
       #     T = TimeVariable('T')
       #     Dyn = DynamicVariable('Dyn')
       #
       #     dm = EpochDemographicModel()
       #     dm.add_epoch(Tp, [nu1F])
       #     dm.add_split(0, [nu1F, nu2B])
       #     dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', Dyn])
       #
       #     data = SFSDataHolder(YRI_CEU_DATA, projections=(6, 6))
       #     engine = get_engine('moments')
       #     engine.set_model(dm)
       #     engine.set_data(data)
       #
       #     ga = get_global_optimizer("Genetic_algorithm")
       #     ga.maximize = True
       #     def f(x):
       #         y = engine.evaluate(x)
       #         return y
       #     res = ga.optimize(f, dm.variables, verbose=0, maxeval=30)
       #     # print(res)
       #
       #     var2value = engine.model.var2value(res.x)
       #     engine.model.fix_dynamics(res.x)
       #     x0 = [var2value[var] for var in engine.model.variables]
       #     ls = get_local_optimizer("BFGS")
       #     ls.maximize = True
       #     # print(ls.optimize(f, engine.model.variables, x0, verbose=0, maxiter=1))
       #
       #     def callback(x, y):
       #         pass
       #     res = ga.optimize(f, dm.variables, verbose=10, maxeval=30,
       #                       report_file='report_file', save_file='save_file',
       #                       eval_file='eval_file', callback=callback)
       #
       # def test_run_gadma_test(self):
       #     sys.argv = ['gadma', '--test']
       #     gadma.core.main()

    def test_missed_lines(self):
        ga = get_global_optimizer("Genetic_algorithm")
        ga.cur_mut_rate = 0.1
        ga._sample_mut_rate(mode='uniform')

        # check_x
        variables = [ContinuousVariable('v', domain=[0, 1])]
        self.assertRaises(ValueError, ga.check_x,
                          variables, [-1], raises=True)
        variables = [DiscreteVariable('v', domain=[1, 2])]
        self.assertRaises(ValueError, ga.check_x,
                          variables, [1.5])

        # is_stopped with impr_gen=None
        ga.is_stopped(10, 1)

    def test_restore_file(self):
        ga = get_global_optimizer("Genetic_algorithm")
        output_file = os.path.join(EXAMPLE_FOLDER, "save_file")
        run_info = ga._create_run_info()
        run_info.result.x = []
        try:
            with open(output_file, 'wb') as f:
                pickle.dump([1, 2, 3], f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            bad_run_info = copy.deepcopy(run_info)
            bad_run_info.result = ""
            with open(output_file, 'wb') as f:
                pickle.dump(bad_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            bad_run_info = types.SimpleNamespace(result=1, n_iter=0)
            with open(output_file, 'wb') as f:
                pickle.dump(bad_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            bad_run_info = copy.deepcopy(run_info)
            bad_run_info.result.n_eval = "not_int"
            with open(output_file, 'wb') as f:
                pickle.dump(bad_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            bad_run_info = copy.deepcopy(run_info)
            bad_run_info.result.X_out = 0
            with open(output_file, 'wb') as f:
                pickle.dump(bad_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            bad_run_info = copy.deepcopy(run_info)
            bad_run_info.result.Y = 0
            with open(output_file, 'wb') as f:
                pickle.dump(bad_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            bad_run_info = copy.deepcopy(run_info)
            bad_run_info.cur_mut_strength = "not_float"
            with open(output_file, 'wb') as f:
                pickle.dump(bad_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            bad_run_info = copy.deepcopy(run_info)
            bad_run_info.gen_times = 0
            with open(output_file, 'wb') as f:
                pickle.dump(bad_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump(run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), True)

            # additional tests when GA is restored from BO
            try:
                bo = get_global_optimizer("SMAC_BO_combination")
            except ValueError:  # Smac is not available
                return
            bo_run_info = bo._create_run_info()
            bo_run_info.result.n_iter = 100

            # if we save solid run_info ga should fail to restore it
            with open(output_file, 'wb') as f:
                pickle.dump(bo_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            # if we make a dict with opt names it should work
            dict_bo_run_info = {bo.id: bo_run_info}
            with open(output_file, 'wb') as f:
                pickle.dump(dict_bo_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), True)

            # if we broke this run_info
            bad_dict_bo_run_info = copy.deepcopy(dict_bo_run_info)
            bad_dict_bo_run_info[bo.id].result.n_iter = "some"
            with open(output_file, 'wb') as f:
                pickle.dump(bad_dict_bo_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

            # but if there will be a run_info from GA it will be restored
            dict_run_info = copy.deepcopy(dict_bo_run_info)
            dict_run_info[ga.id] = run_info
            with open(output_file, 'wb') as f:
                pickle.dump(dict_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), True)

            # and the last not obvious thing
            # if there is valid BO run info and broken run info for GA
            # the result will be FALSE
            bad_dict_run_info = copy.deepcopy(dict_run_info)
            bad_dict_run_info[ga.id].result.n_iter = "some"
            with open(output_file, 'wb') as f:
                pickle.dump(bad_dict_run_info, f)
            self.assertEqual(ga.valid_restore_file(output_file), False)

        finally:
            os.remove(output_file)


class TestInference(unittest.TestCase):

    @pytest.mark.timeout(0)
    def test_inference_ga(self):
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                if engine.id == "dadi":
                    args = ([4, 6, 8],)
                else:
                    args = ()
                filename = f"demographic_model_{engine.id}_YRI_CEU.py"
                location = os.path.join(EXAMPLE_FOLDER, "MODELS", filename)
                spec = importlib.util.spec_from_file_location("module",
                                                              location)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules['module'] = module
                func = getattr(module, 'model_func')

                if engine.id in ["dadi", "moments"]:
                    data = engine.base_module.Spectrum.from_file(EXAMPLE_DATA)
                elif engine.id == 'momi2':
                    data = engine.base_module.sfs_from_dadi(EXAMPLE_DATA)
                elif engine.id == "momentsLD":
                    data = engine.read_data(EXAMPLE_DATA_LD)
                else:
                    raise ValueError(f"Add new engine {engine.id} here for "
                                     "correct data reading in tests.")

                num_init = 10
                p_ids = gadma.cli.get_par_labels_from_file(location)
                variables = gadma.cli.get_variables(parameter_identifiers=p_ids,
                                                    lower_bound=None,
                                                    upper_bound=None)
                X_init = [[var.resample() for var in variables]
                          for _ in range(num_init)]
                Y_init = list()
                def f():
                    for x in X_init:
                        if engine.id == 'dadi':
                            numerics = engine.base_module.Numerics
                            get_model_func = numerics.make_extrap_log_func(func)
                            func_args = (data.sample_sizes, *args)
                            get_ll_func = engine.base_module.Inference.ll_multinom
                        elif engine.id == 'moments':
                            get_model_func = func
                            func_args = (data.sample_sizes, *args)
                            get_ll_func = engine.base_module.Inference.ll_multinom
                        elif engine.id == 'momi2':
                            get_model_func = func
                            func_args = ()
                            def get_ll_func(data, model):
                                model.set_data(data, length=1e8)
                                return model.log_likelihood()
                        elif engine.id == "momentsLD":
                            get_model_func = func
                            rhos = np.array(4 * 10000 * np.array(engine.r_bins))
                            theta = 0.001
                            get_ll_func = engine.base_module.Inference.ll_over_bins
                        else:
                            raise ValueError(f"Add new engine {engine.id} here"
                                             "for correct evaluation in tests.")
                        try:
                            if engine.id == "momentsLD":
                                model = get_model_func(x, rhos, theta)
                                model = moments.LD.Inference.remove_normalized_lds(model)
                                means, varcovs = moments.LD.Inference.remove_normalized_data(
                                    data['means'],
                                    data['varcovs'],
                                    num_pops=model.num_pops,
                                    normalization=0)
                                y = get_ll_func(means, model, varcovs)
                            else:
                                model = get_model_func(x, *func_args)
                                y = get_ll_func(data, model)
                            Y_init.append(y)
                        except AttributeError as e:
                            Y_init.append(-np.inf)

                time = 2 * timeit.timeit(f, number=1) / num_init
                optimize_ga(data, func, engine.id, args=args,
                            p_ids=p_ids, maxtime_per_eval=0.1,
                            num_init=num_init, X_init=X_init, Y_init=None,
                            gen_size=10, ga_maxiter=2, ls_maxiter=1,
                            verbose=1, callback=None,
                            save_file='save_file', eval_file='eval_file',
                            mutation_rate=1.5e-8,
                            fixed_ancestral_size=10000)

    #                    report_file='report_file')

    def test_inference_claic_funcs(self):
        dirname = os.path.join(EXAMPLE_FOLDER, "DATA",
                               "sfs", 'YRI_CEU_test_boots')
        for engine in all_engines():
            if engine.id == "momentsLD":
                continue
            projections = (3, 3)
            data = engine.read_data(SFSDataHolder(os.path.join(EXAMPLE_FOLDER,
                                                               "DATA", "sfs",
                                                               'YRI_CEU.fs'),
                                                  projections=projections))
            boots = gadma.Inference.load_data_from_dir(dirname, engine.id,
                                                       projections=projections)

            p0 = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]
            filename = os.path.join(
                EXAMPLE_FOLDER, "MODELS", f'demographic_model_{engine.id}_YRI_CEU.py')
            spec = importlib.util.spec_from_file_location("module", filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # CLAIC is evaluated several times for the same values
            func = remember_results(module.model_func)

            if engine.id == 'dadi':
                pts = [4, 6, 8]
                c1 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     engine.id, pts=pts)
                c2 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     pts=pts)
                # old interface
                c3 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     pts=pts, eps=1e-2)
                c4 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     pts, 1e-2)
                c5 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     pts, eps=1e-2)

                self.assertEqual(c1, c2)
                self.assertEqual(c2, c3)
                self.assertEqual(c3, c4)
                self.assertEqual(c4, c5)
                # fails
                self.assertRaises(Exception, gadma.Inference.get_claic_score,
                                  func, boots, p0, data)
                self.assertRaises(Exception, gadma.Inference.get_claic_score,
                                  func, boots, p0, data, 'moments', pts)
                self.assertRaises(Exception, gadma.Inference.get_claic_score,
                                  func, boots, p0, data, None, 1e-2)
            elif engine.id == 'moments':
                pts = [4, 6, 8]
                c1 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     engine.id, pts=pts)
                c2 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     engine.id)
                # old interface
                c3 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     pts=None, eps=1e-2)
                c4 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     None, 1e-2)
                c5 = gadma.Inference.get_claic_score(func, boots, p0, data,
                                                     None, eps=1e-2)

                self.assertEqual(c1, c2)
                self.assertEqual(c2, c3)
                self.assertEqual(c3, c4)
                self.assertEqual(c4, c5)

                # fails
                self.assertRaises(Exception, gadma.Inference.get_claic_score,
                                  func, boots, p0, data, pts=pts)
            # else:
            # c1 = gadma.Inference.get_claic_score(func, boots, p0, data,
            #                                 engine.id, pts=None)

    def test_claic_uses_bootstrapped_data(self):
        """
        CLAIC = 2 * trace(J * H^-1) - 2 * LL, where hessian H is evaluated
        for the original data and J is build from gradients that are
        evaluated for each of bootstrapped data.
        """
        dirname = os.path.join(EXAMPLE_FOLDER, "DATA",
                               "sfs", 'YRI_CEU_test_boots')
        projections = (3, 3)
        p0 = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]
        pts = [4, 6, 8]
        eps = 1e-2
        for engine in all_engines():
            if engine.id not in ["dadi", "moments"]:
                continue
            module = engine.base_module
            data = engine.read_data(SFSDataHolder(EXAMPLE_DATA,
                                                  projections=projections))
            boots = gadma.Inference.load_data_from_dir(dirname, engine.id,
                                                       projections=projections)
            self.assertTrue(len(boots) > 1)
            filename = os.path.join(
                EXAMPLE_FOLDER, "MODELS",
                f'demographic_model_{engine.id}_YRI_CEU.py')
            spec = importlib.util.spec_from_file_location("module", filename)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            # CLAIC is evaluated several times for the same values
            func = remember_results(model_module.model_func)

            kwargs = {"pts": pts} if engine.id == "dadi" else {}

            def get_claic(some_boots):
                return gadma.Inference.get_claic_score(
                    func, some_boots, p0, data, engine.id, eps=eps, **kwargs
                )

            claic = get_claic(boots)

            # 1. Compare with value that is evaluated directly
            Godambe = importlib.import_module(f"{module.__name__}.Godambe")
            if hasattr(Godambe, "get_grad"):
                get_grad = Godambe.get_grad
            else:
                get_grad = Godambe._get_grad
            ns = data.sample_sizes
            if engine.id == "dadi":
                func_ex = module.Numerics.make_extrap_log_func(func)
            else:
                func_ex = func

            def ll_func(params, some_data):
                if engine.id == "dadi":
                    model = func_ex(params, ns, pts)
                else:
                    model = func_ex(params, ns)
                return module.Inference.ll_multinom(model, some_data)

            x0 = np.array(p0)
            H = - Godambe.get_hess(ll_func, x0, eps, args=[data])
            J = np.zeros((len(x0), len(x0)))
            for boot in boots:
                grad = get_grad(ll_func, x0, eps,
                                args=[module.Spectrum(boot)])
                J += np.outer(grad, grad)
            J /= len(boots)
            expected = 2 * np.trace(np.dot(J, np.linalg.inv(H)))
            expected -= 2 * ll_func(x0, data)
            self.assertTrue(np.isclose(claic, expected),
                            msg=f"{engine.id}: {claic} != {expected}")

            # 2. Value should be different for another bootstrapped data
            for other_boots in [boots[:1], boots[1:]]:
                other_claic = get_claic(other_boots)
                self.assertFalse(
                    np.isclose(claic, other_claic),
                    msg=f"{engine.id}: CLAIC is the same ({claic}) for "
                        "different bootstrapped data."
                )

    def test_claic_for_parameters_without_influence(self):
        """
        Some parameters could have no influence on the likelihood, e.g.
        fraction of split when sizes of populations are changed suddenly.
        Hessian is singular in that case. CLAIC should be evaluated and be
        equal to CLAIC of the model without such parameters.
        """
        dirname = os.path.join(EXAMPLE_FOLDER, "DATA",
                               "sfs", 'YRI_CEU_test_boots')
        projections = (4, 4)
        eps = 1e-3
        values = {"s1": 0.5, "t1": 0.3, "nu11": 1.5, "nu12": 0.8,
                  "m1_12": 1.0}
        names_with_influence = ["t1", "nu11", "nu12", "m1_12"]
        for engine in all_engines():
            if engine.id not in ["dadi", "moments"]:
                continue
            module = engine.base_module
            engine.set_data(SFSDataHolder(EXAMPLE_DATA,
                                          projections=projections))
            data = engine.data
            boots = gadma.Inference.load_data_from_dir(dirname, engine.id,
                                                       projections=projections)
            model = StructureDemographicModel(
                initial_structure=[1, 1], final_structure=[1, 1],
                has_migs=True, has_sels=False, has_dom=False,
                has_dyns=False, sym_migs=True, frac_split=True,
                has_p_misid=False,
            )
            engine.set_model(model)
            self.assertEqual(sorted(values.keys()),
                             sorted([var.name for var in model.variables]))
            x0 = [values[var.name] for var in model.variables]
            if engine.id == "dadi":
                args = ([8, 12, 16],)
            else:
                args = (engine.default_dt_fac,)
            ll = engine.evaluate(x0, *args)

            # fraction of split has no influence
            other_values = dict(values)
            other_values["s1"] = 0.1
            other_x0 = [other_values[var.name] for var in model.variables]
            self.assertEqual(ll, engine.evaluate(other_x0, *args))

            claic, used_eps = gadma.utils.get_claic_score(
                engine, x0, boots, args=args, eps=eps, log_likelihood=ll,
                return_eps=True
            )
            self.assertEqual(used_eps, eps)
            self.assertTrue(claic is not None)

            # Evaluate the same without fraction of split
            Godambe = importlib.import_module(f"{module.__name__}.Godambe")
            if hasattr(Godambe, "get_grad"):
                get_grad = Godambe.get_grad
            else:
                get_grad = Godambe._get_grad
            ns = data.sample_sizes

            def ll_func(params, some_data):
                new_values = dict(values)
                new_values.update(zip(names_with_influence, params))
                x = [new_values[var.name] for var in model.variables]
                sfs = engine.simulate(x, ns, None, None, *args)
                return module.Inference.ll_multinom(sfs, some_data)

            p0 = np.array([values[name] for name in names_with_influence])
            H = - Godambe.get_hess(ll_func, p0, eps, args=[data])
            J = np.zeros((len(p0), len(p0)))
            for boot in boots:
                grad = get_grad(ll_func, p0, eps,
                                args=[module.Spectrum(boot)])
                J += np.outer(grad, grad)
            J /= len(boots)
            expected = 2 * np.trace(np.dot(J, np.linalg.inv(H))) - 2 * ll
            self.assertTrue(np.isclose(claic, expected),
                            msg=f"{engine.id}: {claic} != {expected}")
