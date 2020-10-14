import unittest
import sys
import os
import numpy as np
import copy
import shutil

from gadma.cli.arg_parser import ArgParser, get_settings
from gadma.cli import SettingsStorage
from gadma import *

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


class TestCLI(unittest.TestCase):
    def test_argparser(self):
        parser = ArgParser()
        parser.format_help()

        old_argv = copy.copy(sys.argv)
        try:
            sys.argv = ['gadma', '--test']
            settings, _ = get_settings()
            settings.inner_data

            sys.argv = ['gadma']
            self.assertRaises(SystemExit, get_settings)

            param_file = os.path.join(DATA_PATH, 'another_test_params')
            another_fs = os.path.join(DATA_PATH, 'YRI_CEU.fs')
            sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir',
                        '-i', another_fs]
            settings, _ = get_settings()
            self.assertEqual(settings.output_directory, abspath('some_dir'))
            self.assertEqual(settings.input_file, abspath(another_fs))
            sys.argv = ['gadma', '-p', param_file, '-o', 'tests',
                        '-i', another_fs]
            self.assertRaises(RuntimeError, get_settings)

            param_file = os.path.join(DATA_PATH,
                                      'another_test_params_without_base')
            sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
            self.assertRaises(AttributeError, get_settings)
            sys.argv = ['gadma', '-p', param_file, '-i', another_fs]
            self.assertRaises(AttributeError, get_settings)

            param_file = os.path.join(DATA_PATH,
                                      'another_test_params_bad1')
            sys.argv = ['gadma', '-p', param_file]
            self.assertRaises(AttributeError, get_settings)

            param_file = os.path.join(DATA_PATH,
                                      'another_test_params_bad2')
            sys.argv = ['gadma', '-p', param_file]
            self.assertRaises(AttributeError, get_settings)

        finally:
            sys.argv = old_argv

    def test_settings_storage(self):
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
                          'initial_structure', '1, -2')
        self.assertRaises(ValueError,  settings.__setattr__,
                          'initial_structure', 1.5)
        self.assertRaises(ValueError,  settings.__setattr__,
                          'final_structure', {2: 1.5})

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
        self.assertRaises(ValueError, settings.__setattr__,
                          'upper_bound', [1.4, 3])
        # self.assertRaises(ValueError, settings.__setattr__,
        #                   'upper_bound', [0, 3, 5])
        self.assertRaises(ValueError, settings.__setattr__,
                          'lower_bound', [0.3, 2])
        # self.assertRaises(ValueError, settings.__setattr__,
        #                   'lower_bound', [2, 3, 5])
        settings.initial_structure = [1, 1]
        settings.final_structure = [3, 2]
        self.assertRaises(ValueError, settings.__setattr__,
                          'final_structure', [1, 3, 2])
        settings.number_of_populations = 2
        self.assertRaises(ValueError, settings.__setattr__,
                          'number_of_populations', 3)



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
        settings.fractions = [0.2, 0.3, 0.3]

        # number of populations and length of lists in other order
        settings = SettingsStorage()
        settings.number_of_populations = 2
        self.assertRaises(ValueError, settings.__setattr__,
                          'initial_structure', [1])

        # files and dirs
        self.assertRaises(ValueError, settings.__setattr__,
                          'directory_with_bootstrap', 'not_existing_dir')
        self.assertRaises(ValueError, settings.__setattr__,
                          'input_file', 'not_existing_file')
        settings.parameter_identifiers = 'nu, n, t, f, s'
        self.assertRaises(ValueError, settings.__setattr__,
                          'parameter_identifiers', 'e, t')

        settings.const_for_mutation_strength = 1.5
        settings.const_for_mutation_rate = 1.04
        self.assertRaises(ValueError, settings.__setattr__,
                          'const_for_mutation_strength', 2.5)
        self.assertRaises(ValueError, settings.__setattr__,
                          'const_for_mutation_rate', 5)

        # vmin
        settings.vmin = 1e-15
        self.assertRaises(ValueError, settings.__setattr__,
                          'vmin', 0)
        self.assertRaises(ValueError, settings.__setattr__,
                          'vmin', -1e-15)

        # check that pts is working in all engines
        for engine in all_engines():
            settings.engine = engine.id
            settings.pts = [20, 30, 40]
            settings.engine = engine.id

        # units of time in drawing
        settings.units_of_time_in_drawing = 'YeArs'
        self.assertRaises(ValueError, settings.__setattr__,
                          'units_of_time_in_drawing', 'ctrange_value')
        settings.const_of_time_in_drawing = 0.01
        self.assertEqual(settings.units_of_time_in_drawing, 'thousand years')
        settings.const_of_time_in_drawing = 100
        self.assertEqual(settings.const_of_time_in_drawing, 1.0)
        self.assertEqual(settings.units_of_time_in_drawing, 'generations')

        # min and max bound of variables
        settings.min_n = 0.1
        settings.max_n = 1000
        self.assertRaises(ValueError, settings.__setattr__,
                          'min_n', 0)
        self.assertTrue(
            (PopulationSizeVariable('v').domain == [0.1, 1000]).all())
        settings.min_t = 1e-4
        settings.max_t = 10
        self.assertRaises(ValueError, settings.__setattr__,
                          'min_t', -1)
        self.assertTrue((TimeVariable('v').domain == [1e-4, 10]).all())
        settings.min_m = 0
        settings.max_m = 5
        self.assertTrue((MigrationVariable('v').domain == [0, 5]).all())

        # get initial structure when number of populations is known
        settings = SettingsStorage()
        self.assertEqual(settings.initial_structure, None)
        settings.number_of_populations = 2
        self.assertTrue(settings.initial_structure == [1, 1])

    def test_old_param_file(self):
        old_param_file = os.path.join(DATA_PATH, 'example_params_old')
        settings = SettingsStorage()
        settings.from_file(old_param_file)

    def test_another_param_file(self):
        param_file = os.path.join(DATA_PATH,
                                  'another_test_params')
        out_dir = os.path.join(os.path.dirname(__file__), 'output_dir')
        if check_dir_existence(out_dir):
            shutil.rmtree(out_dir)

        old_argv = copy.copy(sys.argv)
        try:
            sys.argv = ['gadma', '-p', param_file]
            core.main()
        finally:
            if check_dir_existence(out_dir):
                shutil.rmtree(out_dir)
            sys.argv = old_argv

    def test_saved_settings_storage(self):
        param_file = os.path.join(DATA_PATH, 'another_test_params')
        saved_params_file = os.path.join(DATA_PATH, 'params_file')
        saved_extra_params_file = os.path.join(DATA_PATH, 'extra_params_file')

        settings1 = SettingsStorage.from_file(param_file)
        settings1.to_files(saved_params_file, saved_extra_params_file)
        settings2 = SettingsStorage.from_file(saved_params_file,
                                              saved_extra_params_file)

        self.assertEqual(settings1, settings2)
