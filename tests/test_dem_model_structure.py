import unittest
from gadma import *
import itertools

TEST_STRUCTURES = [(1,), (2,), (3,),
                   (1,1), (2,1), (1,2), (3, 2),
                   (1,2,1), (1,1,1), (1, 3, 4)]

class TestModelStructure(unittest.TestCase):
    def _test_initialization(self):
        for structure in TEST_STRUCTURES:
            for create_migs, create_sels, create_dyns, sym_migs in\
                    list(itertools.product([False, True],repeat=4)):
                dm = DemographicModel.from_structure(structure, create_migs,
                                                     create_sels, create_dyns,
                                                     sym_migs)
                n_par = 2 * (len(structure) - 1)  # for splits variables 
                for i, str_val in enumerate(structure):
                    if i == 0:
                        # for each interval (except first) there is time,
                        # size of population, selection and dynamic
                        n_par += (str_val - 1) * (2 + int(create_sels)\
                                 + int(create_dyns))
                    else:
                        # for other intervals there are also migrations
                        n_pop = i + 1
                        n_migs = int(create_migs) * (n_pop * (n_pop - 1))
                        if sym_migs:
                            n_migs /= 2
                        n_par += str_val * (n_pop * (1 + int(create_dyns)\
                                 + int(create_sels)) + n_migs + 1)
                msg = f"Parameters are not equal for dem model with structure "\
                      f"{structure} and create_migs ({create_migs}), "\
                      f"create_sels ({create_sels}), create_dyns ({create_dyns}), "\
                      f"sym_migs ({sym_migs})"
                self.assertEqual(len(dm.variables), n_par, msg=msg)

    def _test_likelihood(self):
        for structure in TEST_STRUCTURES:
#            print(structure)
            for create_migs, create_sels, create_dyns, sym_migs in\
                    list(itertools.product([False, True],repeat=4)):
                create_sels = False
                model_generator = lambda structure: DemographicModel.from_structure(structure, create_migs,
                                                     create_sels, create_dyns,
                                                     sym_migs)
 
                dm = model_generator(structure)
                variables = dm.variables
                x = [var.resample() for var in variables]
#                print(dm.var2value(x))

                for engine in all_engines():
                    engine.set_model(dm)
                    if engine.id == 'dadi':
                        sizes = [20 for _ in range(len(structure))]
                        args = ([5, 10, 15],)  # pts
                    else:
                        sizes = [4 for _ in range(len(structure))]
                        args = ()
                    # simulate data
                    data = engine.simulate(x, sizes, *args)
                    engine.set_data(data)
#                    print(data)
#                    print(type(data))

                    # get ll of data
                    ll_true = engine.evaluate(x, *args)

                    # increase structure
                    for i in range(len(structure)):
                        new_structure = list(copy.copy(structure))
                        new_structure[i] += 1
                        msg = f"Increase structure from {structure} to "\
                              f"{new_structure} for engine {engine.id}. "\
                              f"create_migs: {create_migs}, "\
                              f"create_sels: {create_sels}, "\
                              f"create_dyns: {create_dyns}, "\
                              f"sym_migs: {sym_migs}"
                        print(msg)
                        new_dm, new_X = increase_structure(dm, new_structure,
                                                           [x], model_generator)
                        engine.set_model(new_dm)
                        #print("!!!", dm.var2value(x), new_dm.var2value(new_X[0]))
                        new_ll = engine.evaluate(new_X[0], *args)
                        self.assertTrue(np.allclose(ll_true, new_ll),
                                        msg=f"{ll_true} != {new_ll} : " + msg)

