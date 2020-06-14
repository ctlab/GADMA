
import os
import ruamel.yaml
import numpy as np
from . import settings
from ..data import SFSDataHolder
from ..engines import get_engine
from ..models import StructureDemographicModel
from ..optimizers import get_local_optimizer, get_global_optimizer
from ..utils import ensure_dir_existence, ensure_file_existence

HOME_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PARAM_TEMPLATE = os.path.join(HOME_DIR, "params_template")
EXTRA_PARAM_TEMPLATE = os.path.join(HOME_DIR, "extra_params_template")


class SettingsStorage(object):

    def __setattr__(self, name, value):
        int_attrs = ['stuck_generation_number', 'sequence_length',
                     'print_models_code_every_n_iteration',
                     'draw_models_every_n_iteration', 'size_of_generation',
                     'number_of_repeats', 'number_of_processes',
                     'time_to_print_summary']
        float_attrs = ['theta0', 'time_for_generation', 'eps',
                       'const_of_time_in_drawing', 'vmin', 'min_n', 'max_n',
                       'min_t', 'max_t', 'min_m', 'max_m']
        probs_attrs = ['mean_mutation_strength', 'mean_mutation_rate']
        bool_attrs = ['outgroup','linked_snp_s', 'only_sudden', 'no_migrations',
                      'silence', 'test']
        int_list_attrs = ['pts', 'initial_structure', 'final_structure',
                          'projections']
        probs_list_attrs = ['fractions']
        attrs_with_equal_len = ['initial_structure', 'final_structure',
                                'population_labels', 'projections']
        special_attrs = ['const_for_mutation_strength',
                         'const_for_mutation_rate', 'vmin']
        exist_file_attrs = ['input_file'] # TODO
        empty_dir_attrs = ['output_directory'] # TODO

        data_holder_attrs = ['projections', 'outgroup',
                             'population_labels', 'sequence_length']
        bounds_attrs = ['min_n', 'max_n', 'min_t', 'max_t', 'min_m', 'max_m']
        missed_attrs = ['engine', 'local_optimizer', '_inner_data']
        
        if (name not in int_attrs and name not in float_attrs and
                name not in probs_attrs and name not in bool_attrs and
                name not in attrs_with_equal_len and 
                name not in int_list_attrs and
                name not in probs_list_attrs and name not in special_attrs and
                name not in exist_file_attrs and
                name not in empty_dir_attrs and
                name not in data_holder_attrs and
                name not in bounds_attrs and name not in missed_attrs):
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
            if (not isinstance(value, (float, np.float)) and 
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
                except:
                    raise error
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise error
            for val in value:
                if not isinstance(val, (int, np.integer)):
                    raise error
                if val < 0:
                    raise ValueError(f"Setting {name} ({value}) have positive"
                                     " elements.")
        # 1.5 Check is the list of probabilities
        if name in probs_list_attrs and we_check:
            error = ValueError(f"Setting {name} ({value}) must be list of "
                               "probabilities.")
            if isinstance(value, str):
                try:
                    value = [float(x) for x in value.split(',')]
                    super(SettingsStorage, self).__setattr__(name, value)
                except:
                    raise error
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise error
            for val in value:
                if (not isinstance(val, (float, np.float)) or 
                        val < 0 or val > 1):
                    raise error
        # 1.6 Check that lengths of arrays are equal between lists
        # and equal to the number of populations
        if name in attrs_with_equal_len and we_check:
            for attr_name in attrs_with_equal_len:
                attr_value = getattr(self, attr_name)
                if attr_value is None:
                    continue
                if len(value) != len(attr_value):
                    raise ValueError(f"Setting {name} ({value}) has different"
                                     f" length with another setting "
                                     f"{attr_name} ({attr_value}).")
            if (hasattr(self, 'number_of_populations') and
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
            ensure_dir_existence(value, check_emptiness=True)
        # 1.9 Check file exist
        if name in exist_file_attrs:
            ensure_file_existence(value)

        # 2. Dependencies checks
        # 2.1 Const of metation strength and rate
        if name in ['const_for_mutation_strength', 'const_for_mutation_rate']:
            if not isinstance(value, float):
                raise ValueError(f"Setting {name} ({value}) must be float.")
            if value < 1 or value > 2:
                raise ValueError(f"Setting {name} (value) must be between 1 "
                                 "and 2.")
        # 2.2 Vmin
        if name == 'vmin':
            if value <= 0:
                raise ValueError(f"Setting {name} (value) must be > 0.")


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
        # 3.4 For local_optimizer we need check it exists
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
                if len(getattr(attr_name)) != self.number_of_populations:
                    raise ValueError(f"Length of {attr_name} should be equal "
                                     f"to {self.number_of_populations}.")
        # 3.7 If we set fractions or size of generation then we create/update
        # GA options that depend on these values.
        elif name ==  'fractions' or name == 'size_of_generation':
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

    
    def __getattr__(self, name):
        try:
            return super(SettingsStorage, self).__getattr__(name)
        except AttributeError:            
            if not hasattr(settings, name):
                if name == 'initial_structure':
                    try:
                        super(SettingsStorage, self).__getattr__(
                            'number_of_populations')
                    except AttributeError:
                        raise AttributeError("Setting number of populations "
                                             "must be specified.")
                    return [settings.initial_structure_unit
                            for _ in range(self.number_of_populations)]
                elif name == 'final_structure':
                    return self.initial_structure
                raise AttributeError(f"There is no such attribute {name} "
                                     "for SettingsStorage.")
            return getattr(settings, name)

    def read_data(self):
        engine = get_engine(self.engine)
        data = engine.read_data(self.data_holder)
        self.projections = data.sample_sizes
        self.population_labels = data.pop_ids
        self.outgroup = not data.folded  #TODO check function
        if self.pts is None:
            max_n = max(self.projections)
            x = (int((max_n - 1) / 10) + 1) * 10
            self.pts = [x, x + 10, x + 20]
        self._inner_data = data
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
            print(attr_name)
            if not hasattr(settings_storage, attr_name):
                raise AttributeError(f"Unknown identifier: {key}.")
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
        kwargs = self.get_engine_kwargs()
        kwargs['verbose'] = 1
        return kwargs

    def get_engine_kwargs(self):    
        kwargs = {}
        if self.engine == 'dadi':
            kwargs['args'] = (self.pts,)
        else:
            kwargs['args'] = ()
        return kwargs

    def get_model(self):
        create_migs = not self.no_migrations
        create_sels = False
        create_dyns = not self.only_sudden
        sym_migs = False
        gen_time = self.time_for_generation
        Nref = self.theta0
        return StructureDemographicModel(self.initial_structure,
                                         self.final_structure,
                                         create_migs, create_sels,
                                         create_dyns, sym_migs,
                                         gen_time, Nref)
