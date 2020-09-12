import unittest
from gadma import *
import numpy as np
from .test_optimizers import get_func, get_1pop_sim_example_1
from .test_optimizers import get_1pop_sim_example_2, get_2pop_sim_example_1
from .test_data import YRI_CEU_DATA

class TestGeneticAlg(unittest.TestCase):
    def check_diff(self, diff, ind, msg):
        self.assertTrue(not diff[ind], msg=f"Arrays are equal "
                                           f"at index {ind}. " + msg)
        diff[ind] = True
        self.assertTrue(np.all(diff), msg=f"Arrays are not equal at some"
                                          f" index except {ind}. " + msg)

    def test_initialization(self):
        get_global_optimizer('Genetic_algorithm')

    def test_mutation(self):
        mut_types = ['uniform', 'gaussian', 'resample']
        ga = get_global_optimizer('Genetic_algorithm')
        ga.cur_mut_strength = 1.0

        n_var = 5
        variables = []
        for i in range(n_var):
            variables.append(ContinuousVariable('var%d' % i, domain=[0,1]))
        variables[0].domain = [90, 100]
        x_list = [var.resample() for var in variables]
        x_arr = WeightedMetaArray(x_list)


        for ind in range(n_var):
            for mut_type in mut_types:
                with self.subTest(mut_type=mut_type, index=ind,
                                  mut='full_mutation'):
                    ga.mut_strength = 1.0
                    x_mut = ga.mutation(x_arr, variables, mut_type, True)
                    self.assertTrue(np.all(x_mut != x_arr))
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
            x_arr = ga.mutation_by_ind(x_arr, variables, 3,
                                       mutation_type=mut_type,
                                       one_fifth_rule=True)
        self.assertEqual(x_arr.weights[3], 4, msg=msg)

    def test_crossover(self):
        cross_types = ['uniform', 'k_point']
        ks = range(1, 5)
        ga = get_global_optimizer('Genetic_algorithm')

        n_var = 20
        variables = []
        for i in range(n_var):
            variables.append(ContinuousVariable('var%d' % i, domain=[0,1]))
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

    def test_selection(self):
        def f(x):
            return np.sum(x)

        sel_types = ['roulette_wheel', 'rank']
        ga = get_global_optimizer('Genetic_algorithm')

        n_var = 5
        variables = []
        for i in range(n_var):
            variables.append(ContinuousVariable('var%d' % i, domain=[0,1]))
        variables[0].domain = [90, 100]
        X_gen = [[var.resample() for var in variables] for _ in range(20)]
        for sel_type in sel_types:
            X_gen_new, Y_gen_new = ga.selection(f, variables, X_gen,
                                                selection_type=sel_type)


    def run_example(self, engine_id, example_func):
        args = ()
        if engine_id == 'dadi':
            args = ([40,50,60],)
        f, variables = example_func(engine_id, args)

        for opt in all_global_optimizers():
            with self.subTest(optimizer=opt.id):
                res = opt.optimize(f, variables, 
                                   args=args, num_init=20, maxeval=35)
                self.assertEqual(res.y, f(res.x, *args))

    def test_1pop_example_1(self):
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                self.run_example(engine.id, get_1pop_sim_example_1)

    def test_1pop_example_2(self):
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                self.run_example(engine.id, get_1pop_sim_example_2)

    def test_2pop_example_1(self):
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                self.run_example(engine.id, get_2pop_sim_example_1)

    def test_yri_ceu(self):
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', Dyn])

        data = SFSDataHolder(YRI_CEU_DATA)
        engine = get_engine('moments')
        engine.set_model(dm)
        engine.set_data(data)

        ga = get_global_optimizer("Genetic_algorithm")
        ga.maximize = True
        def f(x):
            y = engine.evaluate(x)
            return y
        res = ga.optimize(f, dm.variables, verbose=0, maxeval=30)
        #print(res)

        engine.model.fix_dynamics(res.x)
        x0 = res.x[np.array(engine.model.is_fixed) == False]
        ls = get_local_optimizer("BFGS")
        ls.maximize = True
        #print(ls.optimize(f, engine.model.variables, x0, verbose=0, maxiter=1))
