import unittest
import sys
import os
import numpy as np
import copy
import shutil

import gadma
from gadma.cli.arg_parser import ArgParser, get_settings,\
    check_required_settings
from gadma.cli import SettingsStorage, get_variables
from gadma.cli import settings as default_settings
from gadma.utils import StdAndFileLogger
from gadma.utils import PopulationSizeVariable, MigrationVariable,\
    TimeVariable, FractionVariable, ContinuousVariable
from gadma import *
from gadma.core.shared_dict import SharedDict, SharedDictForCoreRun

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


def get_settings_test():
    settings, args = get_settings()
    check_required_settings(settings)
    return settings, args


class TestCLI(unittest.TestCase):
    def test_argparser(self):
        parser = ArgParser()
        parser.format_help()

        old_argv = copy.copy(sys.argv)
        created_params_file = os.path.join(DATA_PATH, "PARAMS", "created_params")

        try:
            sys.argv = ['gadma', '--test']
            settings, _ = get_settings_test()
            settings.inner_data

            sys.argv = ['gadma']
            self.assertRaises(SystemExit, get_settings_test)

            param_file = os.path.join(DATA_PATH, "PARAMS", 'another_test_params')
            another_fs = os.path.join(DATA_PATH, "DATA", "sfs", 'YRI_CEU.fs')
            sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir',
                        '-i', another_fs]
            settings, _ = get_settings_test()

            sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir',
                        '-i', another_fs, '--only_models']
            settings, _ = get_settings_test()

            self.assertEqual(settings.output_directory, abspath('some_dir'))
            self.assertEqual(settings.input_file, abspath(another_fs))
            sys.argv = ['gadma', '-p', param_file, '-o', 'tests',
                        '-i', another_fs]
            self.assertRaises(RuntimeError, get_settings_test)

            param_file = os.path.join(DATA_PATH, "PARAMS",
                                      'another_test_params_without_base')
            sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
            self.assertRaises(AttributeError, get_settings_test)
            sys.argv = ['gadma', '-p', param_file, '-i', another_fs]
            self.assertRaises(AttributeError, get_settings_test)

            param_file = os.path.join(DATA_PATH, "PARAMS",
                                      'another_test_params_bad1')
            sys.argv = ['gadma', '-p', param_file]
            self.assertRaises(AttributeError, get_settings_test)

            param_file = os.path.join(DATA_PATH, "PARAMS",
                                      'another_test_params_bad2')
            sys.argv = ['gadma', '-p', param_file]
            self.assertRaises(AttributeError, get_settings_test)

            param_file = os.path.join(DATA_PATH, "PARAMS",
                                      'another_test_params_bad3')
            sys.argv = ['gadma', '-p', param_file]
            self.assertRaises(AttributeError, get_settings_test)

            dir_without_run = os.path.join(DATA_PATH, "DATA", "sfs",
                                           "YRI_CEU_test_boots")
            dir_with_run = os.path.join(DATA_PATH, "my_example_run")
            with open(created_params_file, 'w') as fl:
                with open(os.path.join(DATA_PATH, "PARAMS",
                                       "another_test_params")) as gl:
                    for line in gl:
                        fl.write(line)
                fl.write(f"Resume from: {dir_without_run}")
            sys.argv = ['gadma', '-p', created_params_file, '--resume',
                        dir_with_run]
            self.assertRaises(ValueError, get_settings_test)
            sys.argv = ['gadma', '-p', created_params_file, '--only_models']
            setting, _ = get_settings_test()
            self.assertEqual(setting.resume_from, None)
            self.assertEqual(setting.only_models, False)
        finally:
            sys.argv = old_argv

    def test_generation_of_bounds_constrain(self):
        params_file = os.path.join(DATA_PATH, "PARAMS", 'example_params_file')
        settings = SettingsStorage.from_file(params_file)
        settings.get_model()

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
        settings.fractions = default_settings.fractions

        # bools
        settings.linked_snp_s = False
        self.assertRaises(ValueError,  settings.__setattr__,
                          'no_migrations', 4)

        # equal length
        settings.lower_bound = [0.3, 2, 4]
        settings.upper_bound = [1.4, 3, 5]
        self.assertRaises(ValueError, settings.__setattr__,
                          'upper_bound', [1.4, 3])
        self.assertRaises(ValueError, settings.__setattr__,
                          'upper_bound', [0, 3, 5])
        self.assertRaises(ValueError, settings.__setattr__,
                          'lower_bound', [0.3, 2])
        self.assertRaises(ValueError, settings.__setattr__,
                          'lower_bound', [2, 3, 5])
        settings.initial_structure = [1, 1]
        settings.final_structure = [3, 2]
        self.assertRaises(ValueError, settings.__setattr__,
                          'final_structure', [1, 3, 2])
        self.assertRaises(ValueError, settings.__setattr__,
                          'initial_structure', "1, some")
        # number of populations
        settings.number_of_populations = 2
        self.assertRaises(ValueError, settings.__setattr__,
                          'number_of_populations', 3)

        # migration mask
        settings.migration_masks = [[0, 0], [1, 0]]
        self.assertRaises(ValueError, settings.__setattr__,
                          'migration_masks', [[[0, "some"], [1, 0]]])
        self.assertRaises(ValueError, settings.__setattr__,
                          'migration_masks', "some")
        self.assertRaises(ValueError, settings.__setattr__,
                          'migration_masks', [[[0], [1, 0]]])
        self.assertRaises(ValueError, settings.__setattr__,
                          'migration_masks', [1, 0])
        self.assertRaises(ValueError, settings.__setattr__,
                          'migration_masks', [[1]])


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

        # units of time on pictures and no time for generation TODO
        settings = SettingsStorage()
        self.assertRaises(ValueError, settings.__setattr__,
                          'units_of_time_in_drawing', "not_years")
        settings.units_of_time_in_drawing = "years"
        self.assertEqual(settings.units_of_time_in_drawing, "generations")

        settings.time_for_generation = 1
        unit = "thousand years"
        settings.units_of_time_in_drawing = unit
        self.assertEqual(settings.units_of_time_in_drawing, unit)

        # custom filename without model_func function
        self.assertRaises(ValueError, settings.__setattr__,
                          "custom_filename",
                          os.path.join(DATA_PATH, "MODELS", "no_model_func.py"))

        # intial_structure after custom file
        settings.custom_filename = os.path.join(
            DATA_PATH, "MODELS", "small_1pop_dem_model_moments.py")
        settings.initial_structure = [1, 1]
        settings.final_structure = [2, 1]

        # custom file with not callable model_func
        path = os.path.join(DATA_PATH, "MODELS",
                            'small_1pop_dem_model_without_function.py')
        self.assertRaises(ValueError, settings.__setattr__,
                          'custom_filename', path)

        # files and dirs
        settings = SettingsStorage()
        self.assertRaises(ValueError, settings.__setattr__,
                          'directory_with_bootstrap', 'not_existing_dir')
        self.assertRaises(ValueError, settings.__setattr__,
                          'input_file', 'not_existing_file')
        settings.parameter_identifiers = 'nu, t, f, s'
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
        # check errors for bad engines
        settings.model_plot_engine = "demes"
        settings.sfs_plot_engine = "dadi"
        self.assertRaises(ValueError,  settings.__setattr__,
                          'engine', 'demes')
        self.assertRaises(ValueError,  settings.__setattr__,
                          'sfs_plot_engine', 'demes')
        self.assertRaises(ValueError,  settings.__setattr__,
                          'model_plot_engine', 'dadi')

        # units of time in drawing
        settings.time_for_generation = 1.0
        settings.units_of_time_in_drawing = 'YeArs'
        self.assertRaises(ValueError, settings.__setattr__,
                          'units_of_time_in_drawing', 'strange_value')
        settings.const_of_time_in_drawing = 0.001
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
            list(PopulationSizeVariable('v').domain) == [0.1, 1000])

        PopulationSizeVariable.default_domain = [1e-2, 100]  # going back

        settings.min_t = 1e-4
        settings.max_t = 10
        self.assertRaises(ValueError, settings.__setattr__,
                          'min_t', -1)
        self.assertTrue(list(TimeVariable('v').domain) == [1e-4, 10])
        settings.min_m = 0
        settings.max_m = 5
        self.assertTrue(list(MigrationVariable('v').domain) == [0, 5])

        no_lin = [0, "Exp"]
        settings.dynamics = no_lin
        self.assertRaises(ValueError, settings.__setattr__,
                          'dynamics', [-1, "Exp"])
        self.assertEqual(list(DynamicVariable('d').domain), no_lin)
        settings.dynamics = "0, Exp"
        self.assertEqual(list(DynamicVariable('d').domain), no_lin)
        settings.dynamics = ["Sud", "Lin", "Exp"]

        # get model with parameters when there is no pop ids
        settings = SettingsStorage()
        settings.custom_filename = os.path.join(
            DATA_PATH, "MODELS", "small_1pop_dem_model_no_ids.py")
        self.assertRaises(ValueError, settings.get_model)
        settings.lower_bound = [1e-2, 1e-2, 1e-15, 1e-15]
        settings.upper_bound = [100, 100, 5, 5]
        dm = settings.get_model()
        settings = SettingsStorage()
        settings.custom_filename = os.path.join(
            DATA_PATH, "MODELS", "small_1pop_dem_model_no_ids_2.py")
        self.assertRaises(ValueError, settings.get_model)
        settings.lower_bound = [1e-2, 1e-2, 1e-15, 1e-15]
        settings.upper_bound = [100, 100, 5, 5]
        dm = settings.get_model()

        # get initial structure when number of populations is known
        settings = SettingsStorage()
        self.assertEqual(settings.initial_structure, None)
        settings.number_of_populations = 2
        self.assertTrue(settings.initial_structure == [1, 1])

    def test_old_param_file(self):
        old_param_file = os.path.join(DATA_PATH, "PARAMS",
                                      'example_params_old')
        settings = SettingsStorage()
        settings.from_file(old_param_file)

    def test_another_param_file(self):
        param_file = os.path.join(DATA_PATH, "PARAMS",
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
        param_file = os.path.join(DATA_PATH, "PARAMS", 'another_test_params')
        saved_params_file = os.path.join(DATA_PATH, "PARAMS", 'params_file')
        saved_extra_params_file = os.path.join(DATA_PATH, "PARAMS",
                                               'extra_params_file')

        settings1 = SettingsStorage.from_file(param_file)
        settings1.pts = None
        settings1.to_files(saved_params_file, saved_extra_params_file)
        settings2 = SettingsStorage.from_file(saved_params_file,
                                              saved_extra_params_file)
        self.assertEqual(settings1, settings2)

        self.assertTrue(not settings1.__eq__(param_file))

        super(SettingsStorage, settings2).__setattr__('new_attr', 1)
        self.assertNotEqual(settings1, settings2)
        settings2 = SettingsStorage.from_file(saved_params_file,
                                              saved_extra_params_file)
        super(SettingsStorage, settings1).__setattr__('new_attr', 1)
        self.assertNotEqual(settings1, settings2)
        settings1 = SettingsStorage.from_file(saved_params_file,
                                              saved_extra_params_file)

        settings1.lower_bound = np.array([0, 0, 0, 0])
        settings1.pts = (10, 20, 30)
        self.assertNotEqual(settings1, settings2)

        settings1 = SettingsStorage.from_file(saved_params_file,
                                              saved_extra_params_file)
        settings2 = copy.deepcopy(settings1)
        settings1.pts = (10, 20, 30)
        settings2.pts = (5, 10, 20)
        self.assertNotEqual(settings1, settings2)

        settings1.fractions = np.array([0.3, 0.1, 0.1])
        gadma.settings.fractions = np.array(gadma.settings.fractions)
        self.assertTrue(isinstance(gadma.settings.fractions, np.ndarray))
        settings1.fractions = np.array(gadma.settings.fractions)
        self.assertTrue(isinstance(settings1.fractions, list))
        gadma.settings.fractions = settings1.fractions

    def test_migration_masks_failure(self):
        options = [
            ["Custom filename: demographic_model_moments_YRI_CEU.py\n",
             "Migration masks: [[0, 1], [1, 0]]\n"],
            ["Initial structure: 1, 1\n",
             "Final structure: 2, 1\n",
             "Migration masks: [[0, 1], [1, 0]]\n"]]
        output = "test_migration_masks_failure"
        params_file = 'params'
        for opts in options:
            with open(params_file, 'w') as fl:
                fl.write("Input file: YRI_CEU.fs\n"
                         f"Output directory: {output}\n")
                for line in opts:
                    fl.write(line)
            sys.argv = ['gadma', '-p', params_file]
            try:
                gadma.PIL_available = False
                gadma.moments_available = False
                self.assertRaises(ValueError, core.main)
            finally:
                if check_dir_existence(output):
                    shutil.rmtree(output)
                os.remove(params_file)
                gadma.PIL_available = True
                gadma.moments_available = True

    def test_inbreeding_run(self):
        params_file = 'params'
        outdir = os.path.join(DATA_PATH, 'inbreed_dir')
        if check_dir_existence(outdir):
            shutil.rmtree(outdir)
        with open(params_file, 'w') as fl:
            fl.write("Input file: tests/test_data/DATA/sfs/YRI_CEU.fs\n"
                     "Linked SNP's: False\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxiter: 1\n"
                     "inbreeding: True\n"
                     "engine: dadi")
        sys.argv = ['gadma', '-p', params_file, '--output', outdir]
        try:
            gadma.matplotlib_available = False
            core.main()
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)
            gadma.matplotlib_available = True

    def test_inbreeding_fail_run_with_moments(self):
        params_file = 'params'
        outdir = os.path.join(DATA_PATH, 'resume_dir')
        if check_dir_existence(outdir):
            shutil.rmtree(outdir)
        with open(params_file, 'w') as fl:
            fl.write("Input file: tests/test_data/DATA/sfs/YRI_CEU.fs\n"
                     "Linked SNP's: False\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxiter: 1\n"
                     "inbreeding: True\n"
                     "engine: moments")
        sys.argv = ['gadma', '-p', params_file,
                    '--output', outdir]
        try:
            gadma.matplotlib_available = False
            self.assertRaises(ValueError, core.main)
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)
            gadma.matplotlib_available = True


    def test_logging_to_stderr(self):
        saved_stderr = sys.stderr
        sys.stderr = StdAndFileLogger("log_file", stderr=True)
        sys.stderr.write("Something in stderr")
        sys.stderr = saved_stderr
        os.remove("log_file")

    # TODO move to core tests
    def test_shared_dict(self):
        d = SharedDict()
        self.assertTrue(d.default_key(1) is None)
        self.assertEqual(d.get_value(10, None), 10)

        d.get_models_for_process_in_group("process", "group")
        d.get_best_model_in_group("group")
        d.get_best_model_for_process_in_group("process", "group")
        d.add_model_for_process("process", "group", "model")
        d.get_best_model_in_group("group", key=lambda x: 1)

        d = SharedDictForCoreRun()
        d.update_best_model_for_process(1, 'log-likelihood', 'engine',
                                        [1, 2, 3], {'log-likelihood': -10})
        d.update_best_model_for_process(2, 'log-likelihood', 'engine',
                                        [2, 3, 4],
                                        {'log-likelihood': -10, 'AIC': 100})
        d.get_models_in_group('log-likelihood', align_y_dict=True)
        d.get_models_in_group('log-likelihood', align_y_dict=False)

    def test_get_variables_function(self):

        def check(variables):
            self.assertIsInstance(variables[0], PopulationSizeVariable)
            self.assertIsInstance(variables[1], PopulationSizeVariable)
            self.assertIsInstance(variables[2], PopulationSizeVariable)
            self.assertIsInstance(variables[3], MigrationVariable)
            self.assertIsInstance(variables[4], TimeVariable)
            self.assertIsInstance(variables[5], FractionVariable)

        p_ids = ['nu1', 'nu2', 'n', 'm', 't', 'p']
        lower_bound = [1e-2, 1e-2, 1e-5, 0, 0, 0]
        upper_bound = [10, 1, 3, 4, 6, 1]

        variables = get_variables(p_ids, lower_bound, upper_bound)
        check(variables)
        for var, lb, ub in zip(variables, lower_bound, upper_bound):
            self.assertEqual(var.domain[0], lb)
            self.assertEqual(var.domain[1], ub)

        variables = get_variables(p_ids, None, None)
        check(variables)
        for var in variables:
            self.assertEqual(list(var.domain),
                             list(var.__class__.default_domain))

        variables = get_variables(None, lower_bound, upper_bound)
        for var, lb, ub in zip(variables, lower_bound, upper_bound):
            self.assertIsInstance(var, ContinuousVariable)
            self.assertEqual(var.domain[0], lb)
            self.assertEqual(var.domain[1], ub)

        self.assertRaises(AssertionError, get_variables, None, None, None)
        self.assertRaises(AssertionError, get_variables, p_ids,
                          None, upper_bound)
        self.assertRaises(AssertionError, get_variables, p_ids,
                          lower_bound, None)

class TestSomeHandsOn(unittest.TestCase):
    def test_example_1(self):
        sys.argv = ['gadma', '--test']
        core.main()

    def test_example_2(self):
        snp_data = os.path.join(DATA_PATH, "DATA", "sfs", 'data.txt')
        outdir = os.path.join(DATA_PATH, 'resume_dir')
        params_file = 'params'
        if check_dir_existence(outdir):
            shutil.rmtree(outdir)
        with open(params_file, 'w') as fl:
            fl.write("global_maxiter: 1\n"
                     "local_maxiter: 1\n"
                     "Population labels: YRI\n")

        sys.argv = ['gadma', '-i', snp_data, '-p', params_file, '-o', outdir,
                    '--only_models']
        try:
            core.main()
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)
            gadma.matplotlib_available = True
