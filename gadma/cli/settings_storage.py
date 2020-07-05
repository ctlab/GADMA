
import os
import ruamel.yaml
import numpy as np
from . import settings
from ..data import SFSDataHolder
from ..engines import get_engine, MomentsEngine
from ..models import StructureDemographicModel, CustomDemographicModel
from ..optimizers import get_local_optimizer, get_global_optimizer
from ..utils import ensure_dir_existence, ensure_file_existence

import importlib.util
import sys

HOME_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PARAM_TEMPLATE = os.path.join(HOME_DIR, "params_template")
EXTRA_PARAM_TEMPLATE = os.path.join(HOME_DIR, "extra_params_template")

CHANGED_IDENTIFIERS = {"use_moments_or_dadi": "engine",
                       "size_of_population_in_ga": "size_of_generation",
                       "fractions_in_ga": "fractions",
                       "epsilon": "eps",
                       "stop_iteration": "stuck_generation_number",
                       "name_of_local_optimization": "local_optimizer"}

DEPRECATED_IDENTIFIERS = ["multinom", "verbose", "flush_delay",
                          "epsilon_for_ls", "gtol", "maxiter",
                          "multinomial_mutation", "multinomial_crossing",
                          "random_n_a", "distribution", "std",
                          "mean_mutation_rate_for_hc",
                          "const_for_mutation_rate_for_hc",
                          "stop_iteration_for_hc"]

class SettingsStorage(object):

    def __setattr__(self, name, value):
        int_attrs = ['stuck_generation_number', 'sequence_length',
                     'print_models_code_every_n_iteration',
                     'draw_models_every_n_iteration', 'size_of_generation',
                     'number_of_repeats', 'number_of_processes',
                     'number_of_populations', 'time_to_print_summary']
        float_attrs = ['theta0', 'time_for_generation', 'eps',
                       'const_of_time_in_drawing', 'vmin', 'min_n', 'max_n',
                       'min_t', 'max_t', 'min_m', 'max_m']
        probs_attrs = ['mean_mutation_strength', 'mean_mutation_rate']
        bool_attrs = ['outgroup', 'linked_snp_s', 'only_sudden',
                      'no_migrations', 'silence', 'test']
        int_list_attrs = ['pts', 'initial_structure', 'final_structure',
                          'projections']
        float_list_attrs = ['lower_bound', 'upper_bound']
        probs_list_attrs = ['fractions']
        attrs_with_equal_len = ['initial_structure', 'final_structure',
                                'population_labels', 'projections']
        special_attrs = ['const_for_mutation_strength',
                         'const_for_mutation_rate', 'vmin',
                         'parameter_identifiers']
        exist_file_attrs = ['input_file', 'custom_filename']  # TODO
        empty_dir_attrs = ['output_directory']  # TODO

        data_holder_attrs = ['projections', 'outgroup',
                             'population_labels', 'sequence_length']
        bounds_attrs = ['min_n', 'max_n', 'min_t', 'max_t', 'min_m', 'max_m']
        bounds_lists = ['lower_bound', 'upper_bound', 'parameter_identifiers']
        missed_attrs = ['engine', 'local_optimizer', '_inner_data']

        if (name not in int_attrs and name not in float_attrs and
                name not in probs_attrs and name not in bool_attrs and
                name not in attrs_with_equal_len and
                name not in int_list_attrs and
                name not in probs_list_attrs and name not in special_attrs and
                name not in exist_file_attrs and
                name not in empty_dir_attrs and
                name not in data_holder_attrs and
                name not in bounds_attrs and name not in missed_attrs and
                name not in bounds_lists):
            raise ValueError(f"Setting {name} should be checked.")

        super(SettingsStorage, self).__setattr__(name, value)

        # -1. For structures it could be one number. We need to transfrom
        # it in list
        if name == 'initial_structure' or name == 'final_structure':
            if isinstance(value, (int, np.integer)):
                value = [value]
                super(SettingsStorage, self).__setattr__(name, value)

        # 0. If attribute is equal to the same from setting storage
        # then we let it go any way. It is because of None's in settings
        we_check = True
        print(name, value)
        if hasattr(settings, name):
            default_value = getattr(settings, name)
            if (isinstance(value, np.ndarray) or
                    isinstance(default_value, np.ndarray)):
                if (default_value == value).all():
                    we_check = False
            else:
                if default_value == value:
                    we_check = False

        # 1. Base checks
        # 1.1 Check is int (positive)
        if name in int_attrs and we_check:
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"Setting {name} ({value}) must be integer.")
            if value < 0:
                raise ValueError(f"Setting {name} ({value}) must be positive.")
        # 1.2 Check is float and probability
        if (name in float_attrs or name in probs_attrs) and we_check:
            if (not isinstance(value, (int, float, np.float)) and
                    not isinstance(value, (int, np.integer))):
                raise ValueError(f"Setting {name} ({value}) must be float.")
            if name in probs_attrs and (value < 0 or value > 1):
                raise ValueError(f"Setting {name} ({value}) must be between 0 "
                                 "and 1.")
        # 1.3 Check is bool
        if name in bool_attrs and we_check:
            if not isinstance(value, bool):
                raise ValueError(f"Setting {name} ({value}) must be boolean.")
        # 1.4 Check is list of ints
        if name in int_list_attrs and we_check:
            error = ValueError(f"Setting {name} ({value}) must be list of "
                               "integers.")
            if isinstance(value, str):
                try:
                    value = [int(x) for x in value.split(',')]
                    super(SettingsStorage, self).__setattr__(name, value)
                except:  # NOQA
                    raise error
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise error
            for val in value:
                if not isinstance(val, (int, np.integer)):
                    raise error
                if val < 0:
                    raise ValueError(f"Setting {name} ({value}) have positive"
                                     " elements.")
        # 1.5 Check is the list of floats and probabilities
        if ((name in probs_list_attrs or name in float_list_attrs) and
                we_check):
            if name in probs_list_attrs:
                error = ValueError(f"Setting {name} ({value}) must be list of"
                                   " probabilities.")
            else:
                error = ValueError(f"Setting {name} ({value}) must be list of"
                                   " floats.")
            if isinstance(value, str):
                try:
                    value = [float(x) for x in value.split(',')]
                    super(SettingsStorage, self).__setattr__(name, value)
                except:  # NOQA
                    raise error
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise error
            try:
                value = [float(x) for x in value]
                super(SettingsStorage, self).__setattr__(name, value)
            except:
                raise error
            if name in probs_list_attrs:
                for val in value:
                    if val < 0 or val > 1:
                        raise error
        # 1.6 Check that lengths of arrays are equal between lists
        # and equal to the number of populations
        if name in attrs_with_equal_len:
            attrs_list = attrs_with_equal_len
        elif name in bounds_attrs:
            attrs_list = bounds_attrs
        if ((name in attrs_with_equal_len or name in bounds_attrs) and
                we_check):
            for attr_name in attrs_list:
                attr_value = getattr(self, attr_name)
                if attr_value is None:
                    continue
                if len(value) != len(attr_value):
                    raise ValueError(f"Setting {name} ({value}) has different"
                                     f" length with another setting "
                                     f"{attr_name} ({attr_value}).")
            if (name not in bounds_attrs and
                    hasattr(self, 'number_of_populations') and
                    len(getattr(self, name)) != self.number_of_populations):
                raise ValueError(f"Length of {name} should be equal to "
                                 f"{self.number_of_populations}.")
        # 1.7 Population labels could be taken as one string
        if name == 'population_labels':
            if isinstance(value, str):
                value = value.split(',')
                super(SettingsStorage, self).__setattr__(name, value)

        # 1.8 Check for empty dirs
        if name in empty_dir_attrs:
            value = ensure_dir_existence(value, check_emptiness=True)
            super(SettingsStorage, self).__setattr__(name, value)
        # 1.9 Check file exist
        if name in exist_file_attrs:
            value = ensure_file_existence(value)
            super(SettingsStorage, self).__setattr__(name, value)

        # 1.10 Check that identifiers are good:
        if name == "parameter_identifiers":
            if isinstance(value, str):
                value = [x for x in value.split(",")]
            value = [x.strip() for x in value]
            for val in value:
                if val.lower()[0] not in settings.P_IDS:
                    raise ValueError("Each parameter identifier should start"
                                     " with symbol from the following list: "
                                     f"{settings.P_IDS.keys()}")
            super(SettingsStorage, self).__setattr__(name, value)
 
        # 2. Dependencies checks
        # 2.1 Const of mutation strength and rate
        if name in ['const_for_mutation_strength', 'const_for_mutation_rate']:
            if not isinstance(value, float):
                raise ValueError(f"Setting {name} ({value}) must be float.")
            if value < 1 or value > 2:
                raise ValueError(f"Setting {name} ({value}) must be between 1 "
                                 "and 2.")
        # 2.2 Vmin
        if name == 'vmin':
            if value <= 0:
                raise ValueError(f"Setting {name} ({value}) must be > 0.")

        # 3. Now change some other attributes according to new one
        # 3.1 If new_file we need to create data_holder
        if name == 'input_file':
            data_holder = SFSDataHolder(self.input_file,
                                        self.projections,
                                        self.outgroup,
                                        self.population_labels,
                                        self.sequence_length)
            super(SettingsStorage, self).__setattr__('data_holder',
                                                     data_holder)
        # 3.2 If we change some attributes of data_holder we need update it
        elif name in data_holder_attrs:
            if hasattr(self, 'data_holder'):
                setattr(self.data_holder, name, value)
        # 3.3 For engine we need check it exists
        elif name == 'engine':
            engine_obj = get_engine(value)
        # 3.4 For local_optimizer we need check it existence
        elif name == 'local_optimizer':
            optimizer_obj = get_local_optimizer(value)
        # 3.5 If we change engine or pts, we should check for warning if pts
        # would be ignored
        elif name in ['engine', 'pts']:
            if self.engine != 'dadi' and self.pts != settings.pts:
                Warning(f"Engine {value} does not need pts (for dadi only). "
                        "It will be ignored.")
        # 3.6 If we set number of populations, we can now check if length of
        # setted attributes are correct. We have already checked that they are
        # equal between each other
        elif name == 'number_of_populations':
            for attr_name in attrs_with_equal_len:
                if getattr(self, attr_name) is None:
                    continue
                if len(getattr(self, attr_name)) != self.number_of_populations:
                    raise ValueError(f"Length of {attr_name} should be equal "
                                     f"to {self.number_of_populations}.")
        # 3.7 If we set fractions or size of generation then we create/update
        # GA options that depend on these values.
        elif name == 'fractions' or name == 'size_of_generation':
            n_elitism = int(self.fractions[0] * self.size_of_generation)
            p_random = 1 - sum(self.fractions)
            super(SettingsStorage, self).__setattr__('n_elitism',
                                                     n_elitism)
            super(SettingsStorage, self).__setattr__('p_mutation',
                                                     self.fractions[1])
            super(SettingsStorage, self).__setattr__('p_crossover',
                                                     self.fractions[2])
            super(SettingsStorage, self).__setattr__('p_random',
                                                     p_random)
        # 3.8 Units of time in drawings
        elif (name == 'units_of_time_in_drawing' or
                name == 'const_of_time_in_drawing'):
            d = {'generations': 1, 'years': 1,
                 'thousands of years': 0.01, 'kya': 0.01}
            if name == 'units_of_time_in_drawing':
                if value not in d:
                    raise ValueError(f"Setting {name} (value) must be one of:"
                                     f" {d.keys()}")
                if (value != 'generations' and
                        not hasattr(self, 'time_for_generation')):
                    Warning(f"There is no time for one generation, time will "
                            "be in generations")
            else:
                for key, val in d.items():
                    if val == value:
                        super(SettingsStorage, self).__setattr__(
                            'units_of_time_in_drawing', key)
                        break
        # 3.9 Domain of variables
        elif name in bounds_attrs:
            if name == 'min_n':
                PopulationSizeVariable.default_domain[0] = value
            elif name == 'max_n':
                PopulationSizeVariable.default_domain[1] = value
            elif name == 'min_t':
                TimeVariable.default_domain[0] = value
            elif name == 'max_t':
                TimeVariable.default_domain[1] = value
            elif name == 'min_m':
                MigrationVariable.default_domain[0] = value
            elif name == 'max_m':
                MigrationVariable.default_domain[1] = value
            else:
                raise AttributeError("Check for supported variables")
            if name.endswith('n'):
                Warning(f"Domain of PopulationSizeVariable changed to "
                        f"{PopulationSizeVariable.default_domain}")
            if name.endswith('t'):
                Warning(f"Domain of TimeVariable changed to "
                        f"{TimeVariable.default_domain}")
            if name.endswith('m'):
                Warning(f"Domain of MigrationVariable changed to "
                        f"{MigrationVariable.default_domain}")

        # 3.10 If we set custom filename with model we should check it is
        # valid python code
        if name == "custom_filename":
            spec = importlib.util.spec_from_file_location("module", value)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules['module'] = module
            if hasattr(module, "model_func"):
                func_name = "model_func"
            else:
                raise ValueError("There is no such function `model_func` in "
                                 f" file {value}.")
            model_func_value = getattr(module, func_name)
            super(SettingsStorage, self).__setattr__("model_func",
                                                     model_func_value)

        if name in bounds_attrs and self.custom_filename is None:
            msg = f"Setting {name} is set before custom_filename is set."
            if self.initial_structure is not None:
                Warning(msg + " It will be ignored as initial structure of "
                        "model is set.")
            else:
                Warning(msg)
        # 3.11 Check for structure or custom filename and ignore some options
        if (name in ['initial_structure', 'final_structure'] and
                value is not None and self.custom_filename is not None):
            if self.lower_bound is not None and self.upper_bound is not None:
                Warning(f"Setting {name} will be ignored as the custom model"
                        " is already set.")                

    def __getattr__(self, name):
        try:
            return super(SettingsStorage, self).__getattr__(name)
        except AttributeError:
            if not hasattr(settings, name):
                raise AttributeError(f"There is no such attribute {name} "
                                     "for SettingsStorage.")
            value = getattr(settings, name)
            if value is not None:
                return value
            # If it is None then maybe we want to return some default value
            if name == "initial_structure" and self.custom_filename is None:
                if hasattr(self, "number_of_populations"):
                    return [settings.initial_structure_unit
                            for _ in range(self.number_of_populations)]
            elif name == "final_structure" and self.custom_filename is None:
                return self.initial_structure
            elif (name == "parameter_identifiers" and
                    self.custom_filename is not None):
                with open(self.custom_filename) as f:
                    for line in f:
                        if line.startswith("def model_func"):
                            break
                    try:
                        line = next(f)
                        p_ids = line.strip().split("=")[0].split(",")
                        self.__setattr__("parameter_identifiers", p_ids)
                        return p_ids
                    except ValueError:  # wrong list
                        pass
                    except StopIteration:  # no function
                        pass
            elif ((name == "lower_bound" or name == "upper_bound") and
                    self.custom_filename is not None):
                if self.parameter_identifiers is not None:
                    bound = list()
                    for p_id in self.parameter_identifiers:
                        domain = settings.P_IDS[p_id[0].lower()].default_domain
                        if name == "lower_bound":
                            bound.append(domain[0])
                        else:
                            bound.append(domain[1])
                    self.__setattr__(name, bound)
                    return bound
            return value

    def read_data(self):
        engine = get_engine(self.engine)
        data = engine.read_data(self.data_holder)
        self.projections = data.sample_sizes
        self.population_labels = data.pop_ids
        self.outgroup = not data.folded  # TODO check function
        if self.pts is None:
            max_n = max(self.projections)
            x = (int((max_n - 1) / 10) + 1) * 10
            self.pts = [x, x + 10, x + 20]
        self._inner_data = data
        self.number_of_populations = len(self.projections)
        return data

    @property
    def inner_data(self):
        if not hasattr(self, '_inner_data'):
            self._inner_data = self.read_data()
        return self._inner_data

    @staticmethod
    def from_file(param_file, extra_param_file=None):
        # Load all values
        if param_file is None and extra_params_file is None:
            return SettingsStorage()
        loaded_dict = {}
        if param_file is not None:
            with open(param_file) as fl:
                loaded_dict = ruamel.yaml.load(fl,
                                               ruamel.yaml.RoundTripLoader)
        if extra_param_file is not None:
            with open(extra_param_file) as fl:
                extra_dict = ruamel.yaml.load(fl,
                                              ruamel.yaml.RoundTripLoader)
            loaded_dict = {**loaded_dict, **extra_dict}

        # Create object
        settings_storage = SettingsStorage()
        for key in loaded_dict:
            attr_name = key.lower().strip()
            attr_name = attr_name.replace(" ", "_")
            attr_name = attr_name.replace("'", "_")
            attr_name = attr_name.replace("__", "_")
            print(attr_name)
            if attr_name in CHANGED_IDENTIFIERS:
                attr_name = CHANGED_IDENTIFIERS[attr_name]
                setting_name = attr_name.replace("_", " ").capitalize()
                Warning(f"Setting `{key}` is renamed in 2 version of GADMA to"
                        f" `{setting_name}`. It is successfully read.")
                
            if not hasattr(settings_storage, attr_name):
                if attr_name in DEPRECATED_IDENTIFIERS:
                    Warning(f"Setting `{key}` was deprecated in 2 version of "
                            "GADMA. If you have not set it in purpose, "
                            "ignore this warning.")
                else:
                    raise AttributeError(f"Unknown identifier: `{key}`.")
            settings_storage.__setattr__(attr_name, loaded_dict[key])
        return settings_storage

    def to_files(self, params_file, extra_params_file):
        # TODO comments with default values
        known_missed_attrs = ['data_holder', 'const_of_time_in_drawing']
        for filename, template in zip([params_file, extra_params_file],
                                      [PARAM_TEMPLATE, EXTRA_PARAM_TEMPLATE]):
            with open(template) as fl:
                readed_template = fl.read()
            loaded_template = ruamel.yaml.load(readed_template,
                                               ruamel.yaml.RoundTripLoader)
            for key_name in loaded_template:
                attr_name = key_name.lower()
                attr_name = attr_name.replace("' ", '_')
                attr_name = attr_name.replace(' ', '_')
                attr_name = attr_name.replace("'", '_')
                attr_name = attr_name.replace("__", "_")
                value = getattr(self, attr_name)
                # get rid of umpy as yaml could not serialize it
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, np.float):
                    value = float(value)
                if isinstance(value, np.integer):
                    value = int(value)
                if isinstance(value, np.bool):
                    value = bool(value)
                loaded_template[key_name] = value

            with open(filename, 'w') as fl:
                ruamel.yaml.dump(loaded_template, fl,
                                 Dumper=ruamel.yaml.RoundTripDumper)

    def get_global_optimizer(self):
        ga = get_global_optimizer("Genetic_algorithm")
        ga.gen_size = self.size_of_generation
        ga.n_elitism = self.n_elitism
        ga.p_mutation = self.p_mutation
        ga.p_crossover = self.p_crossover
        ga.p_random = self.p_random
        ga.eps = self.eps
        ga.n_stuck_gen = self.stuck_generation_number
        ga.maximize = True
        return ga

    def get_local_optimizer(self):
        ls = get_local_optimizer(self.local_optimizer)
        ls.maximize = True
        return ls

    def get_optimizers_kwargs(self):
        kwargs = {}
        kwargs['args'] = self.get_engine_args()
        kwargs['verbose'] = 1
        return kwargs

    def get_engine_args(self, engine_id=None):
        if engine_id == None:
            engine_id = self.engine
        if engine_id == 'dadi':
            args = (self.pts,)
        else:
            args = (MomentsEngine.default_dt_fac,)
        return args

    def get_model(self):
        gen_time = self.time_for_generation
        theta0 = self.theta0
        mut_rate = self.mutation_rate
        if (self.initial_structure is not None and
                self.final_structure is not None):
            create_migs = not self.no_migrations
            create_sels = False
            create_dyns = not self.only_sudden
            sym_migs = False
            return StructureDemographicModel(self.initial_structure,
                                             self.final_structure,
                                             create_migs, create_sels,
                                             create_dyns, sym_migs,
                                             gen_time, theta0, mut_rate)
        elif (self.custom_filename is not None and
                self.lower_bound is not None and
                self.upper_bound is not None):
            var_classes = list()
            if self.parameter_identifiers is not None:
                for p_id in self.parameter_identifiers:
                    var_classes.append(settings.P_IDS[p_id[0].lower()])
            else:
                for _ in self.lower_bound:
                    var_classes.append(ContinuousVariable)
            names = [f"var{i}" for i in range(len(var_classes))]
            if self.parameter_identifiers is not None:
                p_ids = self.parameter_identifiers
                if len(set(p_ids)) == len(p_ids):
                    names = p_ids
            variables = list()
            for low_bound, upp_bound, var_cls, name in zip(self.lower_bound,
                                                           self.upper_bound,
                                                           var_classes,
                                                           names):
                variables.append(var_cls(name, domain=[low_bound, upp_bound]))
            return CustomDemographicModel(self.model_func, variables,
                                          gen_time, theta0, mut_rate)
