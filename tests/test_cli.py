import unittest
import sys
import os
import numpy as np
import copy
import shutil
import pytest
from pathlib import Path

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
from gadma.cli import arg_parser

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


def get_settings_test():
    settings, args = get_settings()
    check_required_settings(settings)
    return settings, args


def check_output_files(test, outdir, engine, num_runs, model_plot, comp_plot, custom_model, gen_code_iter, draw_model_iter, mu_L_available, aic_or_claic=None):
    from gadma import moments_available, dadi_available, momi_available, demes_available
    def check_file_exists(directory, filename):
        msg = f"Expect file or directory named {filename} in the output directory {directory}, but it is not there.\n"\
              f"Listdir of output directory:\n{os.listdir(directory)}"
        test.assertTrue(os.path.exists(os.path.join(directory, filename)), msg)

    def check_output_for_engines(directory, engines_list, prefix=""):
        for eng in engines_list:
            if eng != "demes":
                check_file_exists(directory=directory, filename=prefix + f"best_logLL_model_{eng}_code.py")
            else:
                check_file_exists(directory=directory, filename=prefix + f"best_logLL_model_{eng}_code.py.yml")
            if aic_or_claic is not None:
                if eng != "demes":
                    check_file_exists(directory=directory, filename=prefix + f"best_{aic_or_claic}_model_{eng}_code.py")
                else:
                    check_file_exists(directory=directory, filename=prefix + f"best_{aic_or_claic}_model_{eng}_code.py.yml")

    def check_pictures_output(directory, full_prefix=""):
        if model_plot:
            check_file_exists(directory=directory, filename=full_prefix + "_model.pdf")
        if comp_plot:
            check_file_exists(directory=directory, filename=full_prefix + "_data_comp.pdf")
        if model_plot and comp_plot and gadma.PIL_available:
            check_file_exists(directory=directory, filename=full_prefix + ".png")
                

    check_file_exists(directory=outdir, filename="params_file")
    check_file_exists(directory=outdir, filename="extra_params_file")
    check_file_exists(directory=outdir, filename="GADMA.log")
    check_engines = [engine]
    if not custom_model:
        if dadi_available and engine != "dadi":
            check_engines.append("dadi")
        if moments_available and engine != "moments":
            check_engines.append("moments")
        if momi_available and mu_L_available and engine != "momi2":
            check_engines.append("momi2")
        if demes_available:
            check_engines.append("demes")
    else:
        if engine == "dadi" and demes_available:
            check_engines.append("demes")

    check_output_for_engines(directory=outdir, engines_list=check_engines)

    for num in range(num_runs):
        check_file_exists(directory=outdir, filename=str(num + 1))
        check_file_exists(directory=os.path.join(outdir, str(num + 1)), filename="GADMA_GA.log")
        check_file_exists(directory=os.path.join(outdir, str(num + 1)), filename="eval_file")
        check_output_for_engines(directory=os.path.join(outdir, str(num + 1)), engines_list=check_engines, prefix="current_")
        check_output_for_engines(directory=os.path.join(outdir, str(num + 1)), engines_list=check_engines, prefix="final_")
        check_pictures_output(directory=os.path.join(outdir, str(num + 1)), full_prefix="final_best_logLL_model")
        # check output for iterations
        if gen_code_iter:
            check_file_exists(directory=os.path.join(outdir, str(num + 1)), filename="code")
            for eng in check_engines:
                check_file_exists(directory=os.path.join(outdir, str(num + 1), "code"), filename=eng)
                ext = "py" if eng != "demes" else "yml"
                check_file_exists(directory=os.path.join(outdir, str(num + 1), "code", eng), filename=f"iteration_0.{ext}")
        if draw_model_iter:
            check_file_exists(directory=os.path.join(outdir, str(num + 1)), filename="pictures")
            check_pictures_output(directory=os.path.join(outdir, str(num + 1), "pictures"), full_prefix="iteration_0")

    check_pictures_output(directory=outdir, full_prefix="best_logLL_model")


class TestCLI(unittest.TestCase):
    def tearDown(self):
        if Path("./some_dir").exists():
            shutil.rmtree("./some_dir")

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
            # А тут setting дважды, поэтому он ниже просто перезаписывается

            sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir',
                        '-i', another_fs, '--only_models']
            settings, _ = get_settings_test()

            self.assertEqual(settings.output_directory, abspath('some_dir'))
            self.assertEqual(settings.input_data, abspath(another_fs))
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
        settings.verbose = 1.0
        settings.n_elitism = np.int8(2)
        self.assertRaises(ValueError, settings.__setattr__,
                          'n_elitism', -1)
        self.assertRaises(ValueError, settings.__setattr__,
                          'number_of_repeats', 1.5)
        # sequence length
        self.assertRaises(ValueError, settings.__setattr__,
                          'sequence_length', -10)
        self.assertRaises(ValueError, settings.__setattr__,
                          'sequence_length', "something")
        self.assertRaises(ValueError, settings.__setattr__,
                          'sequence_length', {"1": 100, "2": "something"})

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

        # input data as vcf file
        vcf_file = os.path.join(DATA_PATH, "DATA", "vcf",
                                "out_of_africa_chr22_sim.vcf")
        popmap_file = os.path.join(DATA_PATH, "DATA", "vcf",
                                   "out_of_africa_chr22_sim.popmap")
        settings.input_data = f"{vcf_file}, {popmap_file}"
        self.assertRaises(AssertionError, settings.__setattr__,
                          'input_data', f"{vcf_file}, {popmap_file}, {vcf_file}")
        self.assertRaises(AssertionError, settings.__setattr__,
                          'input_data', f"{popmap_file}, {vcf_file}")
        self.assertRaises(AssertionError, settings.__setattr__,
                          'input_data', f"{vcf_file}")

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
        #self.assertEqual(settings.units_of_time_in_drawing, "generations")

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
                          'input_data', 'not_existing_file')

        # par ids
        settings.parameter_identifiers = 'nu, t, f, s'
        self.assertRaises(ValueError, settings.__setattr__,
                          'parameter_identifiers', 'e, t')
        # repeats of names
        self.assertRaises(ValueError, settings.__setattr__,
                          'parameter_identifiers', ['n', 't', 's', 'n'])


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
        self.assertRaises(ValueError,  settings.__setattr__,
                          'engine', 'demes')
        self.assertRaises(ValueError,  settings.__setattr__,
                          'model_plot_engine', 'dadi')
        # check that momi transforms to momi2
        if gadma.momi_available:
            settings.engine = "momi"
            self.assertEqual(settings.engine, "momi2")
            self.engine = "moments"

        # check error for wrong dadi extrapolation
        settings.dadi_extrapolation = "make_extrap_func"
        self.assertRaises(ValueError, settings.__setattr__,
                          'dadi_extrapolation', 'something_else')
        settings.dadi_extrapolation = "make_extrap_log_func"

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

        TimeVariable.default_domain = [1e-15, 5]  # going back
        MigrationVariable.default_domain = [0, 10]  # going back

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

        # get model for model from gadma
        settings = SettingsStorage()
        settings.custom_filename = os.path.join(
            DATA_PATH, "MODELS", "simple_gadma_model.py")
        dm = settings.get_model()

        # get initial structure when number of populations is known
        settings = SettingsStorage()
        self.assertEqual(settings.initial_structure, None)
        settings.number_of_populations = 2
        self.assertTrue(settings.initial_structure == [1, 1])

        # check that we cannot infer structure model when pop num is >3
        settings = SettingsStorage()
        settings.number_of_populations = 4
        self.assertRaises(ValueError, check_required_settings, settings)

        # Bayesian optimization
        if smac_available:
            settings = SettingsStorage()
            settings.custom_filename =  os.path.join(
                DATA_PATH, "MODELS", "demographic_model_dadi_3pops.py")
            settings.global_optimizer = "SMAC_BO_combination"
            if dadi_available:
                settings.engine = "dadi"
                settings.number_of_populations = 2
                check_required_settings(settings)
            if momi_available:
                settings.engine = "momi"
                settings.number_of_populations = 5
                check_required_settings(settings)
            if moments_LD_available:
                settings.engine = "momentsLD"
                settings.number_of_populations = 5
                check_required_settings(settings)
            if moments_available:
                settings.engine = "moments"
                settings.number_of_populations = 5
                settings.num_init_const = 10
                check_required_settings(settings)

        # check is_valid function of settings_storage
        try:
            settings = SettingsStorage()
            settings.engine = "moments"
            settings.output_directory = "Some_out_dir"
            settings.mutation_rate = 1e-5
            settings.sequence_length = 1e6
            settings.model_plot_engine = "demes"
            vcf_file = os.path.join(DATA_PATH, "DATA", "vcf", 'out_of_africa_chr22_sim.vcf')
            popmap = os.path.join(DATA_PATH, "DATA", "vcf", 'out_of_africa_chr22_sim.popmap')
            settings.input_data = f"{vcf_file}, {popmap}"

            settings.is_valid()
            
            # structure model
            settings.selection = False
            settings.dominance = True
            settings.is_valid()
            self.assertFalse(settings.selection)
            self.assertFalse(settings.dominance)

            settings.ancestral_state_misid_error = True
            settings.outgroup = False
            settings.is_valid()
            self.assertFalse(settings.ancestral_state_misid_error)
            settings.outgroup = True

            if gadma.momi_available:
                settings.ancestral_state_misid_error = True
                settings.engine = "momi2"
                self.assertRaises(ValueError, settings.is_valid)
                settings.engine = "moments"
                settings.ancestral_state_misid_error = False

                settings.engine = "momi2"
                settings.mutation_rate = None
                self.assertRaises(RuntimeError, settings.is_valid)
                settings.mutation_rate = 1e-5
                settings.sequence_length = None
                self.assertRaises(RuntimeError, settings.is_valid)
                settings.sequence_length = {"chr22": 100000, "chr1": 1000000}
                self.assertRaises(ValueError, settings.is_valid)
                settings.sequence_length = 1e6
                settings.model_plot_engine = "momi2"
                settings.units_of_time_in_drawing = "genetic units"
                self.assertRaises(RuntimeError, settings.is_valid)
                settings.model_plot_engine = "demes"
                settings.units_of_time_in_drawing = "years"
                settings.engine = "moments"
                settings.dynamics = ["Sud", "Lin", "Exp"]

            settings.ld_kwargs = {'cM': False, 'report': True}
            settings.is_valid()
            settings.ld_kwargs = None
            settings.is_valid()

            settings.number_of_populations = 1
            settings.lower_bound_of_first_split = 100
            self.assertRaises(ValueError, settings.is_valid)
            settings.lower_bound_of_first_split = None
            settings.upper_bound_of_first_split = 100
            self.assertRaises(ValueError, settings.is_valid)
            settings.upper_bound_of_first_split = None
            settings.lower_bound_of_second_split = 100
            self.assertRaises(ValueError, settings.is_valid)
            settings.lower_bound_of_second_split = None
            settings.upper_bound_of_second_split = 100
            self.assertRaises(ValueError, settings.is_valid)
            settings.upper_bound_of_second_split = None
            del settings.number_of_populations

            settings.number_of_populations = 3
            settings.lower_bound_of_first_split = 100
            settings.upper_bound_of_first_split = 10
            self.assertRaises(ValueError, settings.is_valid)
            settings.lower_bound_of_first_split = None
            settings.upper_bound_of_first_split = None
            settings.lower_bound_of_second_split = 100
            settings.upper_bound_of_second_split = 10
            self.assertRaises(ValueError, settings.is_valid)
            settings.lower_bound_of_second_split = None
            settings.upper_bound_of_second_split = None
            del settings.number_of_populations
            
            settings.fixed_ancestral_size = 10000
            settings.sequence_length = None
            self.assertRaises(ValueError, settings.is_valid)
            settings.sequence_length = 1e6
            settings.mutation_rate = None
            self.assertRaises(ValueError, settings.is_valid)
            settings.mutation_rate = 1e-5

            settings.ancestral_size_as_parameter = False
            settings.is_valid()
            self.assertTrue(settings.ancestral_size_as_parameter)
            settings.fixed_ancestral_size = None
            settings.ancestral_size_as_parameter = False

            # custom model
            settings.custom_filename = os.path.join(DATA_PATH, "MODELS", 'demographic_model_moments_YRI_CEU.py')
            settings.is_valid()
            self.assertEqual(settings.model_plot_engine, "moments")

            settings.inbreeding = True
            settings.is_valid()
            self.assertFalse(settings.inbreeding)

            settings.ancestral_state_misid_error = True
            settings.is_valid()
            self.assertFalse(settings.ancestral_state_misid_error)

            settings.engine = "moments"
            settings.model_plot_engine = "demes"

            settings.engine = "momentsLD"
            settings.custom_filename = os.path.join(DATA_PATH, "MODELS", 'demographic_model_momentsLD_YRI_CEU.py')
            settings.is_valid()  # will give warning that no plot will be generated

            vcf_file = os.path.join(DATA_PATH, "DATA", "vcf", 'data.vcf')
            popmap = os.path.join(DATA_PATH, "DATA", "vcf", 'popmap')
            settings.input_data = f"{vcf_file}, {popmap}"
            settings.sequence_length = 1e6
            self.assertRaises(ValueError, settings.is_valid)
        finally:
            if check_dir_existence("Some_out_dir"):
                shutil.rmtree("Some_out_dir")

    def test_settings_for_custom_gadma_model(self):
        """
        Custom model could be built by user as GADMA model. It gets theta0,
        mutation and recombination rates and time for generation from
        settings, its own values are kept if there are no such settings.
        """
        import tempfile
        model_file = os.path.join(DATA_PATH, "MODELS",
                                  "simple_gadma_model.py")

        def get_values(model):
            return (model.theta0, model.mutation_rate,
                    model.recombination_rate, model.gen_time)

        # 1. Model from file
        settings = SettingsStorage()
        settings.custom_filename = model_file
        dm = settings.get_model()
        self.assertTrue(isinstance(dm, EpochDemographicModel))
        self.assertEqual(get_values(dm), (None, None, None, None))
        settings.theta0 = 0.5
        settings.time_for_generation = 25
        self.assertEqual(get_values(settings.get_model()),
                         (0.5, None, None, 25))
        settings.mutation_rate = 1.25e-8
        settings.recombination_rate = 1e-8
        self.assertEqual(get_values(settings.get_model()),
                         (0.5, 1.25e-8, 1e-8, 25))

        # size of ancestral population is known then
        settings = SettingsStorage()
        settings.custom_filename = model_file
        settings.engine = "moments"
        settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                           "small_1pop.fs")
        settings.theta0 = 0.5
        engine = get_engine("moments")
        engine.set_data(settings.inner_data)
        engine.set_model(settings.get_model())
        engine.data_holder = settings.data_holder
        args = settings.get_engine_args()
        x = [1.5, 0.2]
        self.assertTrue(engine.evaluate(x, *args) is not None)
        self.assertTrue(np.isclose(engine.get_N_ancestral(x, *args),
                                   engine.get_theta(x, *args) / 0.5))

        # 2. Model has its own values
        work_dir = tempfile.mkdtemp("gadma_test_dir")
        try:
            own_model_file = os.path.join(work_dir, "gadma_model.py")
            with open(model_file) as f:
                code = f.read()
            self.assertTrue("EpochDemographicModel()" in code)
            with open(own_model_file, "w") as f:
                f.write(code.replace(
                    "EpochDemographicModel()",
                    "EpochDemographicModel(theta0=0.7, mutation_rate=2e-8, "
                    "gen_time=30)"
                ))
            settings = SettingsStorage()
            settings.custom_filename = own_model_file
            self.assertEqual(get_values(settings.get_model()),
                             (0.7, 2e-8, None, 30))
            settings.theta0 = 0.5
            settings.recombination_rate = 1e-8
            self.assertEqual(get_values(settings.get_model()),
                             (0.5, 2e-8, 1e-8, 30))
        finally:
            shutil.rmtree(work_dir)

        # 3. Function of the model is set directly
        def model_func():
            nu1 = PopulationSizeVariable("nu1")
            t1 = TimeVariable("t1")
            model = EpochDemographicModel(mutation_rate=2e-8)
            model.add_epoch(time_arg=t1, size_args=[nu1])
            return model

        for bounds in [None, ([1e-2, 1e-15], [100, 5])]:
            settings = SettingsStorage()
            settings.model_func = model_func
            if bounds is not None:
                settings.parameter_identifiers = ["nu1", "t1"]
                settings.lower_bound, settings.upper_bound = bounds
            self.assertEqual(get_values(settings.get_model()),
                             (None, 2e-8, None, None))
            settings.theta0 = 0.5
            settings.time_for_generation = 25
            self.assertEqual(get_values(settings.get_model()),
                             (0.5, 2e-8, None, 25))

    def test_options_of_engines_in_new_process(self):
        """
        Options of engines are kept in their classes. A new process could
        start with the default values of them, so they should be set again
        for the run that is made there.
        """
        from gadma.engines import DadiEngine, MomentsLdEngine
        from gadma.core import core

        def get_options():
            return (DadiEngine.extrapolation, MomentsLdEngine.ld_kwargs,
                    MomentsLdEngine.n_processes)

        def set_options(options):
            (DadiEngine.extrapolation, MomentsLdEngine.ld_kwargs,
             MomentsLdEngine.n_processes) = options

        defaults = get_options()
        options = ("make_extrap_func", {"r_bins": "[0, 1e-6, 1e-5]"}, 3)
        self.assertNotEqual(defaults, options)
        settings = SettingsStorage()
        core_run_cls = core.CoreRun
        try:
            settings.dadi_extrapolation = options[0]
            settings.ld_kwargs = options[1]
            settings.number_of_processes = options[2]
            self.assertEqual(get_options(), options)

            # the state of a new process
            set_options(defaults)
            settings.ensure_engines_options()
            self.assertEqual(get_options(), options)

            # run in a new process
            options_in_run = []

            class Run(object):
                def __init__(self, index, shared_dict, settings):
                    pass

                def run(self, initial_kwargs):
                    options_in_run.append(get_options())

            core.CoreRun = Run
            set_options(defaults)
            core.job(1, None, settings)
            self.assertEqual(options_in_run, [options])
        finally:
            core.CoreRun = core_run_cls
            set_options(defaults)

    def test_run_with_killed_process(self):
        """
        Main process stops with an error when the process of some run is
        killed, for example, by the system when there is not enough memory.
        Pool of processes waits for the result of such run forever.
        """
        import signal
        import subprocess
        import tempfile
        import time
        if not hasattr(signal, "SIGKILL") or not hasattr(os, "killpg"):
            self.skipTest("Process could not be killed in this system.")
        dirname = tempfile.mkdtemp()
        model_file = os.path.join(dirname, "model.py")
        params_file = os.path.join(dirname, "params")
        err_filename = os.path.join(dirname, "stderr")
        with open(model_file, "w") as fl:
            fl.write(
                "import os\n"
                "import signal\n"
                "import multiprocessing\n"
                "import moments\n"
                "\n\n"
                "def model_func(params, ns):\n"
                "    nuB, nuF, TB, TF = params\n"
                "    # process of the run is killed at the first evaluation\n"
                "    name = multiprocessing.current_process().name\n"
                "    if name != 'MainProcess':\n"
                "        os.kill(os.getpid(), signal.SIGKILL)\n"
                "    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])\n"
                "    fs = moments.Spectrum(sts)\n"
                "    fs.integrate(tf=TB, Npop=[nuB])\n"
                "    fs.integrate(tf=TF, Npop=[nuF])\n"
                "    return fs\n"
            )
        data_file = os.path.join(DATA_PATH, "DATA", "sfs", "small_1pop.fs")
        with open(params_file, "w") as fl:
            fl.write(f"Input data: {data_file}\n"
                     f"Output directory: {os.path.join(dirname, 'out')}\n"
                     "Engine: moments\n"
                     f"Custom filename: {model_file}\n"
                     "Number of repeats: 2\n"
                     "Number of processes: 2\n"
                     "Silence: True\n")
        code = ("import sys\n"
                "from gadma.core import main\n"
                f"sys.argv = ['gadma', '-p', {params_file!r}]\n"
                "main()\n")
        env = dict(os.environ)
        repo_dir = os.path.dirname(os.path.dirname(gadma.__file__))
        env["PYTHONPATH"] = os.pathsep.join(
            [repo_dir] + [el for el in [env.get("PYTHONPATH")] if el])
        try:
            with open(err_filename, "w") as err_file:
                # all processes of the run are in one group to stop them
                # together
                proc = subprocess.Popen(
                    [sys.executable, "-c", code], stdout=subprocess.DEVNULL,
                    stderr=err_file, env=env, start_new_session=True
                )
                try:
                    returncode = proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    returncode = None
                finally:
                    # pool could start new processes instead of the stopped
                    # ones, so we stop the group until it is empty
                    for _ in range(100):
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except OSError:
                            break
                        time.sleep(0.1)
                    proc.wait()
            with open(err_filename) as fl:
                err = fl.read()
            self.assertIsNotNone(
                returncode, msg="Main process waits for the killed run.")
            self.assertEqual(returncode, 1, msg=err[-2000:])
            self.assertTrue("killed by signal SIGKILL" in err,
                            msg=err[-2000:])
        finally:
            shutil.rmtree(dirname)

    def test_summary_with_failed_evaluation(self):
        """
        Summary of the runs is printed when the best point of some run is
        the point where evaluation of log-likelihood failed.
        """
        import contextlib
        import io
        import tempfile
        from datetime import datetime
        from unittest import mock
        from gadma.core import print_runs_summary
        from gadma.engines.dadi_moments_common import DadiOrMomentsEngine
        dirname = tempfile.mkdtemp()

        def get_summary(settings, models):
            """Returns lines of summary for the best models of the runs."""
            shared_dict = SharedDictForCoreRun(multiprocessing=False)
            for i, model in enumerate(models):
                shared_dict._put_new_model_for_process(
                    i + 1, "log-likelihood", model)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                print_runs_summary(datetime.now(), shared_dict, settings)
            return stream.getvalue().split("\n")

        try:
            for engine_id in ["moments", "dadi"]:
                settings = SettingsStorage()
                settings.output_directory = os.path.join(dirname, engine_id)
                settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                                   "YRI_CEU.fs")
                settings.projections = [4, 4]
                settings.engine = engine_id
                settings.initial_structure = [1, 1]
                settings.final_structure = [1, 1]
                # values could be translated to physical units
                settings.mutation_rate = 1.25e-8
                settings.sequence_length = 4.04e6
                settings.read_data()
                engine = get_engine(engine_id)
                engine.set_data(settings.inner_data)
                engine.data_holder = settings.data_holder
                engine.set_model(settings.get_model())
                args = settings.get_engine_args()

                def get_x():
                    return [var.resample() for var in engine.model.variables]

                # usual best point of a run
                y = None
                while y is None:
                    x = get_x()
                    y = engine.evaluate(x, *args)
                good_model = (engine, x, {"log-likelihood": y})

                # 1. numerics of engine failed for all entries of SFS
                x = get_x()
                original_simulate = engine.simulate

                def masked_simulate(*args, **kwargs):
                    sfs = original_simulate(*args, **kwargs)
                    sfs.mask[:] = True
                    return sfs

                with mock.patch.object(engine, "simulate",
                                       side_effect=masked_simulate):
                    self.assertIsNone(engine.evaluate(x, *args))
                failed_models = [(engine, x, {"log-likelihood": -np.inf})]
                # 2. error that could happen in dadi
                if engine_id == "dadi":
                    x = get_x()
                    error = AttributeError(
                        "'MaskedArray' object has no attribute 'folded'")
                    with mock.patch.object(DadiOrMomentsEngine, "evaluate",
                                           side_effect=error):
                        self.assertIsNone(engine.evaluate(x, *args))
                    self.assertIsNone(engine.get_theta(x, *args))
                    failed_models.append(
                        (engine, x, {"log-likelihood": -np.inf}))

                for failed_model in failed_models:
                    for models in [[good_model, failed_model],
                                   [failed_model]]:
                        lines = get_summary(settings, models)
                        line = [line for line in lines
                                if line.startswith(f"Run {len(models)}")]
                        self.assertEqual(len(line), 1, msg=str(lines))
                        self.assertTrue("-inf" in line[0], msg=line[0])
        finally:
            shutil.rmtree(dirname)

    def test_bounds_of_split_time_in_settings(self):
        """
        Bounds of split times could be used for the model with structure
        only and when time could be translated to generations.
        """
        try:
            settings = SettingsStorage()
            settings.output_directory = "Some_out_dir"
            settings.engine = "moments"
            settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                               "YRI_CEU.fs")
            # there is no error without bounds
            settings.is_valid()

            # bounds are in generations, so it is not possible to use them
            # when size of ancestral population could not be obtained
            for name in ["lower_bound_of_first_split",
                         "upper_bound_of_first_split"]:
                setattr(settings, name, 1000)
                self.assertRaises(ValueError, settings.is_valid)
                setattr(settings, name, None)
            settings.upper_bound_of_first_split = 1000
            settings.mutation_rate = 1.25e-8
            self.assertRaises(ValueError, settings.is_valid)
            settings.sequence_length = 4.04e6
            settings.is_valid()
            settings.mutation_rate = None
            self.assertRaises(ValueError, settings.is_valid)
            settings.theta0 = 0.37976
            settings.is_valid()
            settings.theta0 = None
            self.assertRaises(ValueError, settings.is_valid)
            settings.ancestral_size_as_parameter = True
            settings.is_valid()
            settings.ancestral_size_as_parameter = False

            # it is unknown what parameters of custom model are split times
            settings.theta0 = 0.37976
            settings.is_valid()
            settings.custom_filename = os.path.join(
                DATA_PATH, "MODELS", 'demographic_model_moments_YRI_CEU.py')
            self.assertRaises(ValueError, settings.is_valid)
            settings.upper_bound_of_first_split = None
            settings.is_valid()
        finally:
            if check_dir_existence("Some_out_dir"):
                shutil.rmtree("Some_out_dir")

    def test_units_of_time_without_time_for_generation(self):
        """
        If time for generation is not set then time could be measured either
        in generations or in genetic units.
        """
        from gadma.core.draw_and_generate_code import \
            get_Nanc_gen_time_and_units
        outdir = "Some_out_dir_for_units"
        data_file = os.path.join(DATA_PATH, "DATA", "sfs", "small_1pop.fs")

        def get_settings_and_messages(units=None, **kwargs):
            settings = SettingsStorage()
            settings.engine = "moments"
            settings.model_plot_engine = "demes"
            settings.output_directory = outdir
            settings.input_data = data_file
            for name, value in kwargs.items():
                setattr(settings, name, value)
            if units is not None:
                settings.units_of_time_in_drawing = units
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                settings.is_valid()
            messages = [str(warning.message) for warning in caught
                        if "Time for generation" in str(warning.message)]
            return settings, messages

        try:
            # 1. Size of ancestral population is unknown: genetic units
            settings, messages = get_settings_and_messages()
            self.assertEqual(settings.units_of_time_in_drawing,
                             "genetic units")
            self.assertEqual(len(messages), 1)
            self.assertTrue("will be in genetic units" in messages[0],
                            msg=messages[0])

            # 2. Size of ancestral population is known: generations
            settings, messages = get_settings_and_messages(
                mutation_rate=1e-8, sequence_length=1e6
            )
            self.assertEqual(settings.units_of_time_in_drawing,
                             "generations")
            self.assertEqual(len(messages), 1)
            self.assertTrue("will be in generations" in messages[0],
                            msg=messages[0])

            # 3. It is known but parameters are relative: genetic units
            settings, messages = get_settings_and_messages(
                mutation_rate=1e-8, sequence_length=1e6,
                relative_parameters=True
            )
            self.assertEqual(settings.units_of_time_in_drawing,
                             "genetic units")

            # 4. Units that could be used are not changed
            for units in ["generations", "genetic units"]:
                settings, messages = get_settings_and_messages(
                    units=units, mutation_rate=1e-8, sequence_length=1e6
                )
                self.assertEqual(settings.units_of_time_in_drawing, units)
                self.assertEqual(len(messages), 0)

            # 5. Nothing is changed if time for generation is set
            settings, messages = get_settings_and_messages(
                units="thousand years", time_for_generation=25
            )
            self.assertEqual(settings.units_of_time_in_drawing,
                             "thousand years")
            self.assertEqual(len(messages), 0)
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)

        # Check values that are used for drawing and generation of code
        class EngineWithNanc(object):
            def get_N_ancestral(self, values, *args):
                return 12345

        engine = EngineWithNanc()
        settings = SettingsStorage()
        settings.engine = "moments"
        # no time for generation
        settings.units_of_time_in_drawing = "generations"
        self.assertEqual(get_Nanc_gen_time_and_units([], engine, settings),
                         (12345, 1.0, "generations"))
        # size of ancestral population is not changed for genetic units
        settings.units_of_time_in_drawing = "genetic units"
        self.assertEqual(get_Nanc_gen_time_and_units([], engine, settings),
                         (12345, 0.5, "genetic units"))
        settings.time_for_generation = 25
        self.assertEqual(get_Nanc_gen_time_and_units([], engine, settings),
                         (12345, 0.5, "genetic units"))
        settings.units_of_time_in_drawing = "thousand years"
        self.assertEqual(get_Nanc_gen_time_and_units([], engine, settings),
                         (12345, 0.025, "thousand years"))
        # time for generation is not used if time is in generations
        settings.units_of_time_in_drawing = "generations"
        self.assertEqual(get_Nanc_gen_time_and_units([], engine, settings),
                         (12345, 1.0, "generations"))

    def test_old_param_file(self):
        # ignore warnings about deprecation and renaming
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.settings_storage', lineno=900)
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.settings_storage', lineno=906)

        old_param_file = os.path.join(DATA_PATH, "PARAMS",
                                      'example_params_old')
        settings = SettingsStorage()
        settings.from_file(old_param_file)

    @pytest.mark.timeout(0)
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
            check_output_files(
                test=self,
                outdir=out_dir,
                engine="dadi",
                num_runs=1,
                model_plot=True,
                comp_plot=True,
                custom_model=True,
                gen_code_iter=True,
                draw_model_iter=True,
                mu_L_available=False,
                aic_or_claic="claic",
            )
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
                fl.write("Input data: YRI_CEU.fs\n"
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
        data_file = os.path.join(DATA_PATH, "DATA", "sfs", "small_1pop.fs")
        if check_dir_existence(outdir):
            shutil.rmtree(outdir)
        with open(params_file, 'w') as fl:
            fl.write(f"Input data: {data_file}\n"
                     "Projections: 6\n"
                     "Linked SNP's: False\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxiter: 1\n"
                     "inbreeding: True\n"
                     "engine: dadi\n"
                     "units_of_time_in_drawing: genetic units")
        sys.argv = ['gadma', '-p', params_file, '--output', outdir]
        try:
            gadma.core.main()
            check_output_files(
                test=self,
                outdir=outdir,
                engine="dadi",
                num_runs=1,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=False,
                draw_model_iter=False,
                mu_L_available=False,
                aic_or_claic="aic",
            )
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)

    def test_inbreeding_fail_run_with_moments(self):
        params_file = 'params'
        outdir = os.path.join(DATA_PATH, 'out_dir')
        if check_dir_existence(outdir):
            shutil.rmtree(outdir)
        with open(params_file, 'w') as fl:
            fl.write("Input data: tests/test_data/DATA/sfs/YRI_CEU.fs\n"
                     "Linked SNP's: False\n"
                     "Silence: True\n"
                     "global_maxiter: 2\n"
                     "local_maxiter: 1\n"
                     "inbreeding: True\n"
                     "engine: moments")
        sys.argv = ['gadma', '-p', params_file,
                    '--output', outdir]
        try:
            self.assertRaises(ValueError, core.main)
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)

    def test_reading_from_cmd_and_checks(self):
        # checks recombination rate warning
        params_file = 'params'
        outdir = os.path.join(DATA_PATH, 'out_dir')
        if check_dir_existence(outdir):
            shutil.rmtree(outdir)
        with open(params_file, 'w') as fl:
            fl.write("Input data: tests/test_data/DATA/sfs/YRI_CEU.fs\n"
                     "Silence: True\n"
                     "engine: moments\n"
                     "recombination rate: 1e-8")
        sys.argv = ['gadma', '-p', params_file,
                    '--output', outdir]
        try:
            settings_storage, args = arg_parser.get_settings()
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)
        
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

    def test_engine_in_shared_dict(self):
        """
        Engine that is kept in the shared dictionary together with a model
        has additional information (e.g. theta) for this model only, not for
        every point that the engine of the run has evaluated.
        """
        import tempfile
        import warnings
        from gadma.utils import WeightedMetaArray
        dirname = tempfile.mkdtemp()
        try:
            for engine_id in ["moments", "dadi"]:
                settings = SettingsStorage()
                settings.output_directory = os.path.join(dirname, engine_id)
                settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                                   "YRI_CEU.fs")
                settings.projections = [4, 4]
                settings.engine = engine_id
                settings.initial_structure = [1, 1]
                settings.final_structure = [1, 1]
                settings.read_data()
                engine = get_engine(engine_id)
                engine.set_data(settings.inner_data)
                engine.data_holder = settings.data_holder
                engine.set_model(settings.get_model())
                args = settings.get_engine_args()
                x, y = None, None
                n_points = 10
                for _ in range(n_points):
                    new_x = [var.resample() for var in engine.model.variables]
                    new_y = engine.evaluate(new_x, *args)
                    if new_y is not None and (y is None or new_y > y):
                        x, y = new_x, new_y
                self.assertEqual(len(engine.saved_add_info), n_points)
                self.assertTrue(y is not None)
                theta = engine.get_theta(x, *args)

                y_dict = {"log-likelihood": y}
                shared_dict = SharedDictForCoreRun(multiprocessing=False)
                shared_dict._put_new_model_for_process(
                    1, "log-likelihood", (engine, x, y_dict))
                x_with_metadata = WeightedMetaArray(x)
                x_with_metadata.metadata = "m"
                shared_dict.update_best_model_for_process(
                    2, "log-likelihood", engine, x_with_metadata, y_dict)
                shared_dict.add_model_for_process(
                    3, "log-likelihood", engine, x, y_dict)
                # engine of the run is not changed
                self.assertEqual(len(engine.saved_add_info), n_points)

                models = shared_dict.get_models_in_group("log-likelihood",
                                                         align_y_dict=True)
                self.assertEqual(len(models), 3)
                for index, (saved_engine, saved_x, saved_y) in models:
                    msg = f"engine {engine_id}, run {index}"
                    self.assertEqual(len(saved_engine.saved_add_info), 1,
                                     msg=msg)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        saved_theta = saved_engine.get_theta(saved_x, *args)
                    # information for the model is taken, not evaluated
                    self.assertEqual(
                        [str(w.message) for w in caught
                         if "Additional evaluation" in str(w.message)], [],
                        msg=msg
                    )
                    self.assertEqual(saved_theta, theta, msg=msg)
        finally:
            shutil.rmtree(dirname)

    def test_get_variables_function(self):

        def check(variables):
            self.assertIsInstance(variables[0], PopulationSizeVariable)
            self.assertIsInstance(variables[1], PopulationSizeVariable)
            self.assertIsInstance(variables[2], PopulationSizeVariable)
            self.assertIsInstance(variables[3], MigrationVariable)
            self.assertEqual(variables[3].units, "genetic")
            self.assertIsInstance(variables[4], TimeVariable)
            self.assertEqual(variables[4].units, "physical")
            self.assertIsInstance(variables[5], FractionVariable)
            self.assertIsInstance(variables[6], GrowthRateVariable)
            self.assertEqual(variables[6].units, "universal")  # TODO it is not correct behavior
            self.assertIsInstance(variables[7], SelectionVariable)

        p_ids = ['nu1', 'nu2', 'n', 'm_gen', 't_phys', 'p', 'g_1', 'gamma']
        lower_bound = [1e-2, 1e-2, 1e-5, 0, 0, 0, -1e-3, 0]
        upper_bound = [10, 1, 3, 4, 10000, 1, 1e-3, 5]

        variables = get_variables(p_ids, lower_bound, upper_bound)
        check(variables)
        for var, lb, ub in zip(variables, lower_bound, upper_bound):
            self.assertEqual(var.domain[0], lb)
            self.assertEqual(var.domain[1], ub)

        variables = get_variables(p_ids, None, None)
        check(variables)
        for var in variables:
            if var.units != "physical":
                self.assertEqual(list(var.domain),
                                 list(var.__class__.default_domain))
            else:
                self.assertEqual(var.name, "t_phys")
                self.assertNotEqual(list(var.domain),
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
            check_output_files(
                test=self,
                outdir=outdir,
                engine="moments",
                num_runs=1,
                model_plot=True,
                comp_plot=True,
                custom_model=False,
                gen_code_iter=False,
                draw_model_iter=False,
                mu_L_available=False,
                aic_or_claic=None,
            )
        finally:
            if check_dir_existence(outdir):
                shutil.rmtree(outdir)
            os.remove(params_file)
            gadma.matplotlib_available = True
