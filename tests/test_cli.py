import unittest
import sys
import os
import numpy as np
import copy

from gadma.cli.arg_parser import ArgParser, get_settings
from gadma.cli import SettingsStorage
from gadma import *

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


class TestCLI(unittest.TestCase):
    def test_argparser(self):
        parser = ArgParser()
        parser.format_help()

        sys.argv = ['gadma', '--test']
        get_settings()

        sys.argv = ['gadma']
        self.assertRaises(SystemExit, get_settings)

    def test_settings_storage(self):
        self.assertEqual(len(list(all_local_optimizers())), 5)
        settings = SettingsStorage()

        some_strange_attr = "some_strange_attr"
        self.assertRaises(ValueError, settings.__setattr__,
                          some_strange_attr, 1)

        # default values of None
        settings.theta0 = None

        # integers
        settings.verbose = 1
        settings.n_elitism = np.int8(2)
        self.assertRaises(ValueError, settings.__setattr__,
                          'number_of_repeats', 1.5)
        self.assertRaises(ValueError, settings.__setattr__,
                          'sequence_length', -10)
        # lists of integers
        settings.pts = [10, 20, 30]
        settings.projections = '10, 20'
        self.assertRaises(ValueError,  settings.__setattr__,
                          'initial_structure', [1.0, 2])
        self.assertRaises(ValueError,  settings.__setattr__,
                          'initial_structure', '1.0, -2')
        self.assertRaises(ValueError,  settings.__setattr__,
                          'initial_structure', 1.5)

        # floats
        settings.p_mutation = np.float32(0.5)
        settings.mean_mutation_strength = 0.2
        settings.eps = 2
        self.assertRaises(ValueError,  settings.__setattr__,
                          'p_random', 2)
        self.assertRaises(ValueError,  settings.__setattr__,
                          'vmin', 'some_string')
        # lists of floats
        settings.lower_bound = [0.3, 1, 2]
        settings.upper_bound = '0.4, 3,5'
        settings.fractions = [0.3, 0.4, 0.2]
        self.assertRaises(ValueError, settings.__setattr__,
                          'fractions', ['some_str', 0.1, 0.3])
        self.assertRaises(ValueError, settings.__setattr__,
                          'fractions', [2, 0.3, 0.4])
        self.assertRaises(ValueError, settings.__setattr__,
                          'fractions', '-2, 0.3, 0.4')
        self.assertRaises(ValueError, settings.__setattr__,
                          'fractions', 2)


        # bools
        settings.linked_snp_s = False
        self.assertRaises(ValueError,  settings.__setattr__,
                          'no_migrations', 4)

        # equal length
        settings.lower_bound = [0.3, 2, 4]
        settings.upper_bound = [1.4, 3, 5]
        print(settings.upper_bound)
        self.assertRaises(ValueError, settings.__setattr__,
                          'upper_bound', [1.4, 3])
        print(settings.upper_bound)
        # self.assertRaises(ValueError, settings.__setattr__,
        #                   'upper_bound', [0, 3, 5])
        self.assertRaises(ValueError, settings.__setattr__,
                          'lower_bound', [0.3, 2])
        # self.assertRaises(ValueError, settings.__setattr__,
        #                   'lower_bound', [2, 3, 5])

        # fractions
        settings.fractions = [0.5, 0.5, 0.5, 0.5]
        self.assertRaises(ValueError, settings.__setattr__,
                          'fractions', '0.3, 0.4')
        self.assertRaises(ValueError, settings.__setattr__,
                          'fractions', [0.5, 0.5, 0.5])
        old_fracs = copy.copy(settings.fractions)
        settings.size_of_generation = None
        settings.n_elitism = 10
        self.assertEqual(old_fracs, settings.fractions)
        settings.size_of_generation = 10
        self.assertEqual(settings.size_of_generation, 10)
        self.assertEqual(settings.fractions[0], 1.0)
        self.assertNotEqual(old_fracs, settings.fractions)
        settings.p_mutation = 0.6
        self.assertEqual(settings.fractions[1], settings.p_mutation)


    def test_old_param_file(self):
        old_param_file = os.path.join(DATA_PATH, 'example_params_old')
        settings = SettingsStorage()
        settings.from_file(old_param_file)



        
       
        
