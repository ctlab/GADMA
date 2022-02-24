import os
import ruamel.yaml
import numpy as np
from . import settings
from .. import demes_available, momi_available
from ..data import SFSDataHolder, \
    VCFDataHolder, check_and_return_projections_and_labels
from ..engines import get_engine, MomentsEngine, MomentsLdEngine
from ..engines import all_engines, all_drawing_engines
from ..models import StructureDemographicModel, CustomDemographicModel
from ..optimizers import get_local_optimizer, get_global_optimizer
from ..optimizers import LinearConstrain
from ..utils import check_dir_existence, check_file_existence, abspath,\
    module_name_from_path, custom_generator, ensure_dir_existence
from ..utils import PopulationSizeVariable, TimeVariable, MigrationVariable,\
    ContinuousVariable, DynamicVariable, GrowthRateVariable,\
    SelectionVariable, FractionVariable, DemographicVariable
from ..data import extract_chromosomes_from_vcf
from ..data import create_recombination_maps_from_rate
from ..data import create_bed_files_and_extract_chromosomes
import warnings
import importlib.util
import sys
import copy
import numbers
import inspect
from keyword import iskeyword

HOME_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
PARAM_TEMPLATE = os.path.join(HOME_DIR, "params_template")
EXTRA_PARAM_TEMPLATE = os.path.join(HOME_DIR, "extra_params_template")

CHANGED_IDENTIFIERS = {"use_moments_or_dadi": "engine",
                       "size_of_population_in_ga": "size_of_generation",
                       "fractions_in_ga": "fractions",
                       "epsilon": "eps",
                       "stop_iteration": "stuck_generation_number",
                       "name_of_local_optimization": "local_optimizer",
                       "lower_bounds": "lower_bound",
                       "upper_bounds": "upper_bound",
                       "multinom": "ancestral_size_as_parameter",
                       "input_file": "input_data"}

DEPRECATED_IDENTIFIERS = ["flush_delay",
                          "epsilon_for_ls", "gtol", "maxiter",
                          "multinomial_mutation", "multinomial_crossing",
                          "distribution", "std",
                          "mean_mutation_rate_for_hc",
                          "const_for_mutation_rate_for_hc",
                          "stop_iteration_for_hc", "sfs_plot_engine"]


def get_variable_class(par_label):
    first_letter = par_label[0].lower()
    if first_letter not in settings.P_IDS:
        raise ValueError(f"Unknown type of parameter: {par_label}. Should "
                         "start with one of the following letters: "
                         f"{settings.P_IDS.keys()}")
    cls = settings.P_IDS[par_label[0].lower()]
    if isinstance(cls, list):
        assert first_letter == 'g'
        if par_label.lower().startswith("gamma"):
            return SelectionVariable
        return GrowthRateVariable
    return cls


def _get_variable(cls, par_label, domain, engine="dadi"):
    """
    Returns correct variable from its label.
    If engine is `dadi` or `moments` then variable will be in `genetic` units
    by default if othervise is not stated. If engine is something else (`momi`)
    then units are `physical` by default.

    If parameter label has `_phys` or `_gen` at the end then it is mark about
    the units. For example, engine is `dadi` but label is `nu_phys` that
    means that variable will be :class:`gadma.PopulationSizeVariable` in
    `physical` units.
    """
    if (cls == FractionVariable or cls is GrowthRateVariable or
            not issubclass(cls, DemographicVariable)):
        return cls(name=par_label, domain=domain)
    units = "genetic" if engine in [
        'dadi', 'moments',  "momentsLD"] else "physical"
    if par_label.lower().endswith("_phys"):
        units = "physical"
    if par_label.lower().endswith("_gen"):
        units = "genetic"
    return cls(name=par_label, units=units, domain=domain)


def get_variables(parameter_identifiers, lower_bound, upper_bound,
                  engine="dadi"):
    assert not (
        parameter_identifiers is None and
        lower_bound is None and
        upper_bound is None
    ), "Either par. identifiers nor lower + upper bounds should be set."
    assert (
        (lower_bound is None and upper_bound is None) or
        (lower_bound is not None and upper_bound is not None)
    ), "Both lower and upper bounds should be set or both equal to None. "\
       f"{lower_bound} {upper_bound}"
    if lower_bound is not None:
        domains = list(zip(lower_bound, upper_bound))
    else:
        domains = [None for _ in parameter_identifiers]
    units = "genetic" if engine in [
        "dadi", "moments", "momentsLD"] else "physical"
    if parameter_identifiers is not None:
        classes = [get_variable_class(p_id) for p_id in parameter_identifiers]
    else:
        warnings.warn("Any parameter identifiers were not found. Please "
                      "check the bounds and units of the parameters "
                      f"carefully. Units will be {units}.")
        classes = [ContinuousVariable for _ in lower_bound]
        parameter_identifiers = [f"var{i}" for i in range(len(lower_bound))]
    variables = []
    for cls, name, domain in zip(classes, parameter_identifiers, domains):
        variables.append(_get_variable(
            cls=cls,
            par_label=name,
            domain=domain,
            engine=engine,
        ))
    return variables


def get_par_labels_from_file(filename):
    with open(filename) as f:
        big_comment = False
        big_comment_str = ""
        found_func = False
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            if found_func and line.strip().startswith("'''"):
                if big_comment and big_comment_str == "'''":
                    big_comment = False
                    big_comment_str = ""
                else:
                    big_comment = True
                    if big_comment_str == "":
                        big_comment_str = "'''"
            elif found_func and line.strip().startswith('"""'):
                if big_comment and big_comment_str == '"""':
                    big_comment = False
                    big_comment_str = ""
                else:
                    big_comment = True
                    if big_comment_str == "":
                        big_comment_str = '"""'
            elif found_func and not big_comment:
                break
            if line.startswith("def model_func"):
                found_func = True
        try:
            p_ids = line.strip().split("=")[0].split(",")
            p_ids = [x.strip() for x in p_ids]
            for x in p_ids:
                get_variable_class(x)
                if not x.isidentifier() or iskeyword(x):
                    raise IndexError
            return p_ids
        except IndexError:  # two commas will create x = "" (x[0])
            pass
        except ValueError:  # not in P_IDS
            pass


class SettingsStorage(object):
    """
    Class to hold all settings of GADMA run. All default values of settings
    are defined in :mod:`gadma.cli.settings`.
    """
    def __setattr__(self, name, value):
        """
        Sets attribute. The default values of all attributes are in
        :mod:`gadma.cli.settings`. If value is equal to default no checks are
        done. Otherwise value is checked at least for type. If name is not
        known setting then error is raised.

        :param name: Name of attribute.
        :param value: Value of the attribute.
        """
        int_attrs = ['stuck_generation_number', 'verbose',
                     'print_models_code_every_n_iteration', 'n_elitism',
                     'draw_models_every_n_iteration', 'size_of_generation',
                     'number_of_repeats', 'number_of_processes',
                     'number_of_populations', 'global_maxiter',
                     'global_maxeval', 'local_maxiter', 'local_maxeval',
                     'num_init_const', "region_len", 'fixed_ancestral_size']
        float_attrs = ['theta0', 'time_for_generation', 'eps',
                       'const_of_time_in_drawing', 'vmin', 'min_n', 'max_n',
                       'min_t', 'max_t', 'min_m', 'max_m',
                       'upper_bound_of_first_split',
                       'upper_bound_of_second_split',
                       'const_for_mutation_strength',
                       'const_for_mutation_rate',
                       'mutation_rate', 'recombination_rate',
                       'time_to_print_summary']
        probs_attrs = ['mean_mutation_strength', 'mean_mutation_rate',
                       'p_mutation', 'p_crossover', 'p_random']
        bool_attrs = ['outgroup', 'linked_snp_s', 'only_sudden',
                      'no_migrations', 'silence', 'test', 'random_n_a',
                      'relative_parameters', 'only_models',
                      'symmetric_migrations', 'split_fractions',
                      'generate_x_transform', 'global_log_transform',
                      'local_log_transform', 'inbreeding', 'selection',
                      'ancestral_size_as_parameter']
        int_list_attrs = ['pts', 'initial_structure', 'final_structure',
                          'projections']
        float_list_attrs = ['lower_bound', 'upper_bound']
        probs_list_attrs = ['fractions']
        attrs_with_equal_len = ['initial_structure', 'final_structure',
                                'population_labels', 'projections']
        special_attrs = ['const_for_mutation_strength',
                         'const_for_mutation_rate', 'vmin',
                         'parameter_identifiers', 'migration_masks']
        exist_file_attrs = ['input_data', 'custom_filename',
                            'bed_file', "preprocessed_data"]
        exist_dir_attrs = ['directory_with_bootstrap',
                           'resume_from', 'recombination_maps']
        empty_dir_attrs = ['output_directory']
        data_holder_attrs = ['projections', 'outgroup',
                             'population_labels', 'sequence_length',
                             "preprocessed_data", "recombination_maps",
                             'bed_files_dir']
        bounds_attrs = ['min_n', 'max_n', 'min_t', 'max_t', 'min_m', 'max_m',
                        'dynamics']
        bounds_lists = ['lower_bound', 'upper_bound', 'parameter_identifiers']
        missed_attrs = ['engine', 'global_optimizer', 'local_optimizer',
                        '_inner_data', '_bootstrap_data', 'X_init', 'Y_init',
                        'model_func', 'get_engine_args', 'data_holder',
                        'units_of_time_in_drawing', 'resume_from_settings',
                        'dadi_available', 'moments_available',
                        'model_plot_engine', 'demes_available',
                        'demesdraw_available',
                        'kernel', 'acquisition_function', 'sequence_length']
        dict_attrs = ['ld_kwargs']

        super_hasattr = True
        setattr_at_the_end = True
        try:
            super(SettingsStorage, self).__getattr__(name)
        except AttributeError:
            super_hasattr = False
        if (name not in int_attrs and name not in float_attrs and
                name not in probs_attrs and name not in bool_attrs and
                name not in attrs_with_equal_len and
                name not in int_list_attrs and
                name not in probs_list_attrs and name not in special_attrs and
                name not in exist_file_attrs and
                name not in exist_dir_attrs and
                name not in empty_dir_attrs and
                name not in data_holder_attrs and
                name not in bounds_attrs and name not in missed_attrs and
                name not in bounds_lists and not super_hasattr and
                name not in dict_attrs):
            raise ValueError(f"Setting {name} should be checked.")

        # -1. For structures it could be one number. We need to transfrom
        # it in list
        if name in int_list_attrs or name in float_list_attrs:
            if isinstance(value, numbers.Real):
                value = [value]
                self.__setattr__(name, value)
                return
        if name == 'population_labels' or name == 'parameter_identifiers':
            if isinstance(value, str):
                if len(value.split(',')) == 1:
                    value = [value.strip()]
                    self.__setattr__(name, value)
                    return

        if isinstance(value, str) and value.lower() == 'none':
            value = None
        # get rid of numpy as yaml could not serialize it
        if isinstance(value, np.ndarray) and name != "_inner_data":
            value = value.tolist()

        # 0. If attribute is equal to the same from setting storage
        # then we let it go any way. It is because of None's in settings
        we_check = True
        if value is None:
            we_check = False
        if hasattr(settings, name):
            default_value = getattr(settings, name)
            if (isinstance(value, np.ndarray) or
                    isinstance(default_value, np.ndarray)):
                if (np.array(default_value) == np.array(value)).all():
                    we_check = False
            else:
                if default_value == value:
                    we_check = False
                    try:
                        existed = object.__getattribute__(self, name)
                        if existed != value:
                            delattr(self, name)
                    except AttributeError:
                        pass
                    setattr_at_the_end = False

        # 1. Base checks
        # 1.1 Check is int (positive)
        if name in int_attrs and we_check:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if (not isinstance(value, numbers.Integral) or
                    isinstance(value, bool)):
                raise ValueError(f"Setting {name} ({value}) must be integer.")
            if value < 0:
                raise ValueError(f"Setting {name} ({value}) must be positive.")
        # 1.2 Check is float and probability
        if (name in float_attrs or name in probs_attrs) and we_check:
            if (not isinstance(value, numbers.Real) or
                    isinstance(value, bool)):
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
                except:  # NOQA
                    raise error
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise error
            for val in value:
                if not isinstance(val, numbers.Integral):
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
                    self.__setattr__(name, value)
                except:  # NOQA
                    raise error
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise error
            try:
                value = [float(x) for x in value]
            except:  # NOQA
                raise error
            if name in probs_list_attrs:
                for val in value:
                    if val < 0 or val > 1:
                        raise error
        # 1.6 Check that lengths of arrays are equal between lists
        # and equal to the number of populations
        if name in attrs_with_equal_len:
            attrs_list = attrs_with_equal_len
        elif name in bounds_lists:
            attrs_list = bounds_lists
        if ((name in attrs_with_equal_len or name in bounds_lists) and
                we_check):
            if (name not in bounds_lists and
                    hasattr(self, 'number_of_populations') and
                    len(value) != self.number_of_populations):
                raise ValueError(f"Length of {name} should be equal to "
                                 f"{self.number_of_populations}.")
            for attr_name in attrs_list:
                if attr_name == name:
                    continue
                attr_value = getattr(self, attr_name)
                if attr_value is None:
                    continue
                if len(value) != len(attr_value):
                    raise ValueError(f"Setting {name} ({value}) has different"
                                     f" length with another setting "
                                     f"{attr_name} ({attr_value}).")
        # 1.7 Population labels could be taken as one string
        if name == 'population_labels':
            if isinstance(value, str):
                value = [x.strip() for x in value.split(',')]
                self.__setattr__(name, value)

        # 1.8.-1 Sequence length could be integer or dict
        if name == "sequence_length" and value is not None:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            is_int = isinstance(value, numbers.Integral)
            is_dict = isinstance(value, dict)
            if not (is_int or is_dict):
                raise ValueError(f"Setting {name} should be integer or "
                                 "dictionary {chrom: length}")
            if is_int and value < 0:
                raise ValueError(f"Setting {name} ({value}) must be positive.")
            new_val = {}
            if is_dict:
                for key, val in value.items():
                    if not isinstance(val, numbers.Integral):
                        raise ValueError(
                            f"Setting {name} should contain integer values "
                            "when it is a dictionary (got {value})."
                        )
                    new_val[str(key)] = val
                value = new_val

        # 1.8.0 expand path of files:
        if (value is not None and
                (name in empty_dir_attrs or
                 name in exist_file_attrs or name in exist_dir_attrs)):
            abspath_value = []
            for val in value.split(","):
                abspath_value.append(abspath(val.strip()))
            value = ", ".join(abspath_value)
        # 1.8 Check for empty dirs (NOW IN ARG_PARSER)
        # if name in empty_dir_attrs and value is not None:
        #     value = ensure_dir_existence(value, check_emptiness=True)
        # 1.9 Check file and dir exist
        if name in exist_file_attrs and value is not None:
            for val in value.split(","):
                val = val.strip()
                if not check_file_existence(val):
                    raise ValueError(f"Setting {name} should be set to existed"
                                     f" file. File {val} does not exist.")
        if name in exist_dir_attrs and value is not None:
            if not check_dir_existence(value):
                raise ValueError(f"Setting {name} should be set to existed "
                                 f"directory. Dir {value} does not exist.")

        # 1.10 Check that identifiers are good:
        if name == "parameter_identifiers" and value is not None:
            if isinstance(value, str):
                value = [x.strip() for x in value.split(",")]
            value = [x.strip() for x in value]
            for val in value:
                get_variable_class(val)
            # Check that identifiers are different
            if not len(set(value)) == len(value):
                raise ValueError("Parameter identifiers should contain unique "
                                 "names of the parameters. Names are not "
                                 f"unique, there are repeats: {value}")

        # 2. Dependencies checks
        # 2.1 Const of mutation strength and rate
        if name in ['const_for_mutation_strength', 'const_for_mutation_rate']:
            if value < 1 or value > 2:
                raise ValueError(f"Setting {name} ({value}) must be between 1 "
                                 "and 2.")
        # 2.2 Vmin
        if name == 'vmin':
            if value is not None and value <= 0:
                raise ValueError(f"Setting {name} ({value}) must be > 0.")

        # 3. Now change some other attributes according to new one
        # 3.1 If new_file we need to create data_holder
        if name == 'input_data':
            files = [name.strip() for name in value.split(",")]
            if len(files) > 1:
                n_files = len(files)
                assert n_files <= 2, "Input file could take maximum two "\
                                     "files: SFS file or VCF file with popmap"\
                                     f" file. Got {n_files} files: {files}"
                extension = files[0][files[0].rfind('.'):].lower()
                assert extension == ".vcf", "Input file was set to two files "\
                                            "but the first one does not have "\
                                            f".vcf extension: {value}"
            else:
                files = [value]
                extension = value[value.rfind('.'):].lower()
            if extension == ".vcf":
                assert len(files) != 1, "VCF file should be set together with"\
                                        " popmap file. Popmap file is missed,"\
                                        " please set it as: vcf_file, "\
                                        "popmap_file"
                vcf_file, popmap_file = files
                data_holder = VCFDataHolder(
                    vcf_file=vcf_file,
                    popmap_file=popmap_file,
                    projections=self.projections,
                    outgroup=self.outgroup,
                    population_labels=self.population_labels,
                    sequence_length=self.sequence_length,
                    recombination_maps=self.recombination_maps,
                    preprocessed_data=self.preprocessed_data
                )
            else:
                data_holder = SFSDataHolder(
                    sfs_file=value,
                    projections=self.projections,
                    outgroup=self.outgroup,
                    population_labels=self.population_labels,
                    sequence_length=self.sequence_length
                )
            super(SettingsStorage, self).__setattr__('data_holder',
                                                     data_holder)

        # 3.2 If we change some attributes of data_holder we need update it
        elif name in data_holder_attrs:
            if hasattr(self, 'data_holder'):
                setattr(self.data_holder, name, value)
        # 3.3 For engine we need check it exists
        elif name == 'engine':
            engine = get_engine(value)
            if not engine.can_evaluate:
                raise ValueError(f"Engine {value} cannot evaluate "
                                 f"log-likelihood. Available engines are: "
                                 f"{[engine.id in all_engines()]}")
        elif name == 'model_plot_engine':
            engine = get_engine(value)
            if not engine.can_draw_model:
                raise ValueError(f"Engine {value} cannot draw model plots. "
                                 f"Available engines are: "
                                 f"{[engine.id in all_drawing_engines()]}")
        # 3.4 For local and global optimizer we need check existence
        elif name == 'global_optimizer':
            get_global_optimizer(value)
        elif name == 'local_optimizer':
            get_local_optimizer(value)
        # 3.5 If we change engine or pts, we should check for warning if pts
        # would be ignored
        elif name in ['engine', 'pts']:
            if self.engine != 'dadi' and self.pts != settings.pts:
                warnings.warn(f"Engine {self.engine} does not need pts (for "
                              "dadi only). It will be used only for generated"
                              " code for dadi if any.")
        # 3.6 If we set number of populations, we can now check if length of
        # setted attributes are correct. We have already checked that they are
        # equal between each other
        elif name == 'number_of_populations':
            for attr_name in attrs_with_equal_len:
                if getattr(self, attr_name) is None:
                    continue
                if len(getattr(self, attr_name)) != value:
                    raise ValueError(f"Length of {attr_name} should be equal "
                                     f"to {self.number_of_populations}.")
        # 3.7 If we set fractions or size of generation then we create/update
        # GA options that depend on these values.
        elif name == 'fractions' or name == 'size_of_generation':
            fractions = value
            gen_size = value
            if name == 'fractions':
                name = '_fractions'
                if len(fractions) not in [3, 4]:
                    raise ValueError("Length of fractions ({value}) must be "
                                     "equal to 3 (old,mut,cros) or 4 "
                                     "(old,mut,cros,rand). Got length of "
                                     f"{len(value)}")
                if len(fractions) == 3:
                    if sum(value) > 1:
                        raise ValueError('Sum of fractions (when 3 fractions'
                                         ' are setted) must be not greater '
                                         f'than 1. ({value})')
                gen_size = self.size_of_generation
            if name == 'size_of_generation':
                fractions = self.fractions
            if fractions is not None:
                if gen_size is not None:
                    n_elitism = int(fractions[0] * gen_size)
                    self.__setattr__('n_elitism', n_elitism)
                if len(fractions) == 3:
                    p_random = 1 - sum(fractions)
                else:
                    p_random = fractions[3]
                self.__setattr__('p_mutation', fractions[1])
                self.__setattr__('p_crossover', fractions[2])
                self.__setattr__('p_random', p_random)
                if name == '_fractions' and len(value) == 3:
                    value.append(p_random)
        # 3.8 Units of time in drawings
        elif (name == 'units_of_time_in_drawing' or
                name == 'const_of_time_in_drawing'):
            d = {'generations': 1, 'years': 1, 'thousand years': 0.001,
                 'thousands of years': 0.001, 'kya': 0.001}
            if name == 'units_of_time_in_drawing':
                value = value.lower()
                if value not in d:
                    raise ValueError(f"Setting {name} ({value}) must be one "
                                     f"of: {d.keys()}")
                if (value != 'generations' and
                        self.time_for_generation is None):
                    value = 'generations'
                    warnings.warn(f"There is no time for one generation, time"
                                  " will be in generations.")  # TODO move
                object.__setattr__(self, 'const_of_time_in_drawing', d[value])
            else:
                if d[self.units_of_time_in_drawing] != value:
                    found = False
                    for key, val in d.items():
                        if val == value:
                            self.__setattr__('units_of_time_in_drawing', key)
                            found = True
                            break
                    if not found:
                        warnings.warn("No such constant for time drawing. It "
                                      "will be equal to 1 and time will be in"
                                      " generations.")
                        self.__setattr__('units_of_time_in_drawing',
                                         'generations')
                        return

        # 3.9 Domain of variables
        elif name in bounds_attrs:
            if name in ['min_n', 'min_t'] and value <= 0:
                raise ValueError(f"Lower bound {name} should be greater "
                                 "than 0.")
            domain_changed = True
            cls = get_variable_class(name.split("_")[-1])
            if cls == DynamicVariable:
                if isinstance(value, str):
                    value = [x.strip() for x in value.split(",")]
                    for i in range(len(value)):
                        if value[i].isdigit():
                            value[i] = int(value[i])

            old_domain = np.array(cls.default_domain)
            if name.startswith('min'):
                cls.default_domain = [value, cls.default_domain[1]]
            elif name.startswith('max'):
                cls.default_domain = [cls.default_domain[0], value]
            else:
                assert name == "dynamics"
                for dyn in value:
                    if dyn not in cls._help_dict:
                        raise ValueError(f"Unknown dynamic {value}. Available "
                                         "dynamics are: "
                                         f"{list(cls._help_dict.keys())}")
                cls.default_domain = value

            domain_changed = len(old_domain) != len(cls.default_domain) or\
                np.any(old_domain != np.array(cls.default_domain))
            if domain_changed:
                if name.endswith('n'):
                    warnings.warn(f"Domain of PopulationSizeVariable changed "
                                  f"to {cls.default_domain}")
                if name.endswith('t'):
                    warnings.warn(f"Domain of TimeVariable changed to "
                                  f"{cls.default_domain}")
                if name.endswith('m'):
                    warnings.warn(f"Domain of MigrationVariable changed to "
                                  f"{cls.default_domain}")
                if name == "dynamics":
                    warnings.warn(f"Domain of DynamicVariable changed to "
                                  f"{cls.default_domain}")

        # 3.10 If we set custom filename with model we should check it is
        # valid python code
        if name == "custom_filename" and value is not None:
            module_name = module_name_from_path(value)
            spec = importlib.util.spec_from_file_location(module_name, value)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module
            if hasattr(module, "model_func"):
                func_name = "model_func"
            else:
                raise ValueError("There is no such function `model_func` in "
                                 f" file {value}.")
            model_func_value = getattr(module, func_name)
            if not callable(model_func_value):
                raise ValueError(f"Function {func_name} should be callable.")

#        if name in bounds_attrs and self.custom_filename is None:
#            msg = f"Setting {name} is set before custom_filename is set."
#            if self.initial_structure is not None:
#                warnings.warn(msg + " It will be ignored as initial structure"
#                              " of model is set.")
#            else:
#                warnings.warn(msg)

        # 3.11 Check for structure or custom filename and ignore some options
        if (name in ['initial_structure', 'final_structure'] and
                value is not None and self.custom_filename is not None):
            if self.lower_bound is not None and self.upper_bound is not None:
                warnings.warn(f"Setting {name} will be ignored as the custom "
                              "model is already set.")

        if ((name == 'lower_bound' and self.upper_bound is not None) or
                (name == 'upper_bound' and self.lower_bound is not None)):
            if name == 'lower_bound':
                lower_bound = value
                upper_bound = self.upper_bound
            else:
                lower_bound = self.lower_bound
                upper_bound = value
            for low, upp in zip(lower_bound, upper_bound):
                if low > upp:
                    raise ValueError(f"Lower bound ({self.lower_bound}) "
                                     f"should be less than upper bound "
                                     f"({self.upper_bound}) element-wise.")

        # 3.12 Check migration masks
        if name == 'migration_masks' and value is not None:
            if np.array(value).shape == (2, 2):
                value = [value]
            transformed_value = []
            for i, mask in enumerate(value):
                new_mask = []
                if not isinstance(mask, list):
                    raise ValueError("Migration masks option should be set to "
                                     "a list with lists (masks).")
                mask_size = None
                for line in mask:
                    if not isinstance(line, list):
                        raise ValueError("Each mask in Migration masks should "
                                         f"be a list. Mask number {i}: {mask}")
                    if mask_size is None:
                        mask_size = len(line)
                    if mask_size != len(line):
                        raise ValueError("Each mask in Migration masks should "
                                         "be a 2x2 or 3x3 list. Mask number "
                                         f"{i} has different lengths: {mask}")
                    for el in line:
                        if el not in [0, 1]:
                            raise ValueError("Masks should consist of 0 and 1"
                                             f" only. Mask number {i}: {mask}")
                    new_mask.append(list(line))
                transformed_value.append(new_mask)
            value = transformed_value

        # 3.13 If we have ld_kwargs we put them as attribute to momentsLD
        if name == 'ld_kwargs':
            MomentsLdEngine.ld_kwargs = value
        if name == 'number_of_processes':
            MomentsLdEngine.n_processes = value

        if setattr_at_the_end:
            super(SettingsStorage, self).__setattr__(name, value)
            # assert(self.__getattr__(name) == value)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # If it is None then maybe we want to return some default value
            if name == "initial_structure" and self.custom_filename is None:
                if hasattr(self, "number_of_populations"):
                    return [settings.initial_structure_unit
                            for _ in range(self.number_of_populations)]
            elif name == "final_structure" and self.custom_filename is None:
                return self.initial_structure
            elif (name == "parameter_identifiers" and
                    self.custom_filename is not None):
                p_ids = get_par_labels_from_file(self.custom_filename)
                if p_ids is not None:
                    object.__setattr__(self,
                                       "parameter_identifiers", p_ids)
                return p_ids
            elif ((name == "lower_bound" or name == "upper_bound") and
                  (self.custom_filename is not None or
                   self.model_func is not None)):
                if self.parameter_identifiers is not None:
                    bound = list()
                    for p_id in self.parameter_identifiers:
                        var = _get_variable(
                            cls=get_variable_class(p_id),
                            par_label=p_id,
                            domain=None,
                            engine=self.engine
                        )
                        if name == "lower_bound":
                            bound.append(var.domain[0])
                        else:
                            bound.append(var.domain[1])
                    return bound
            if hasattr(settings, name):
                return getattr(settings, name)
            raise AttributeError(f"There is no such attribute {name} "
                                 "for SettingsStorage.")

    def __eq__(self, other):
        known_missed_attrs = ['data_holder', 'model_func',
                              'units_of_time_in_drawing', 'bootstrap_data',
                              'inner_data', 'number_of_populations',
                              'test', 'generate_x_transform', 'resume_from',
                              'output_directory']
        if not isinstance(other, self.__class__):
            return False
        for attr_name in set(dir(self) + dir(other)):
            if attr_name in known_missed_attrs:
                continue
            if attr_name.startswith("_"):
                continue
            try:
                other_value = getattr(other, attr_name)
            except AttributeError:
                return False
            try:
                self_value = getattr(self, attr_name)
            except AttributeError:
                return False

            if callable(self_value) or callable(other_value):
                continue

            if (isinstance(self_value, np.ndarray) or
                    isinstance(other_value, np.ndarray)):
                if not np.all(np.array(self_value) == np.array(other_value)):
                    return False
            elif (isinstance(self_value, tuple) or
                    isinstance(other_value, tuple)):
                try:
                    if not list(self_value) == list(other_value):
                        return False
                except TypeError:
                    return False
            else:
                if not self_value == other_value:
                    return False
        return True

    @property
    def fractions(self):
        if not hasattr(self, '_fractions'):
            return settings.fractions
        fracs = self._fractions
        if hasattr(self, 'size_of_generation'):
            if self.size_of_generation is not None:
                p_1 = float(self.n_elitism) / self.size_of_generation
            else:
                p_1 = fracs[0]
        fracs = [p_1, self.p_mutation, self.p_crossover, self.p_random]
        return fracs

    def _prepare_for_moments_ld(self):
        # if we have momentsLD then we should create bed files and rec maps
        we_have_rec_rate = self.recombination_rate is not None
        we_have_rec_maps = (self.recombination_maps is not None and
                            os.path.isdir(self.recombination_maps))
        we_have_rec = we_have_rec_rate or we_have_rec_maps
        print(we_have_rec)
        if isinstance(self.data_holder, VCFDataHolder):
            if self.bed_files_dir is None:
                try:
                    path = os.path.join(self.output_directory, "_bed_files")
                    create_bed_files_and_extract_chromosomes(
                        data_holder=self.data_holder,
                        output_dir=path,
                        region_len=self.region_len,
                    )
                    self.bed_files_dir = path
                except Exception as e:
                    if (self.engine == "momentsLD" and
                            self.preprocessed_data is None):
                        raise ValueError(
                            "Failed to generate bed files for region "
                            "identification due to the following error:\n"
                            f"{str(e)}"
                        ) from e
            else:
                check_dir_existence(self.bed_file_dir)
            # If we have recombination rate only then we have to create rec map
            if not we_have_rec_maps and we_have_rec_rate:
                try:
                    path = os.path.join(
                        self.output_directory, "_recombination_maps"
                    )
                    create_recombination_maps_from_rate(
                        data_holder=self.data_holder,
                        output_dir=path,
                        recombination_rate=self.recombination_rate,
                    )
                    self.recombination_maps = path
                except Exception:
                    if self.engine == "momentsLD":
                        raise ValueError(
                            "Failed to generate recombination maps files "
                            f"due to the following error: {str(e)}"
                        ) from e
        else:
            assert self.engine != "momentsLD", "Set VCF data for momentsLD."

    def read_data(self):
        """
        Reads data with engine. Attribute of`engine` and `data_holder` should
        be set.
        """
        engine = get_engine(self.engine)
        self._prepare_for_moments_ld()
        engine.data = self.data_holder
        self.data_holder = engine.data_holder
        self.projections = self.data_holder.projections
        self.outgroup = self.data_holder.outgroup
        self.population_labels = self.data_holder.population_labels
        self._inner_data = engine.data
        if self.pts is None:
            max_n = max(self.projections)
            x = (int((max_n - 1) / 10) + 1) * 10
            super(SettingsStorage, self).__setattr__("pts",
                                                     [x, x + 10, x + 20])
        self.number_of_populations = len(self.projections)
        return engine.data

    def read_bootstrap_data(self, return_filenames=False):
        """
        Reads all data in the directory `self.bootstrap_data`.

        :param return_filenames: If True then each data is a tuple of
                                 corresponding filename and data.
        """
        if self.directory_with_bootstrap is None:
            return
        if (not hasattr(self, '_inner_data') and
                hasattr(self, 'data_holder') and
                self.data_holder.filename is not None):
            self.read_data()
        dirname = self.directory_with_bootstrap
        engine = get_engine(self.engine)

        all_boot = []
        set_of_seen_files = set()
        filenames = []
        for filename in os.listdir(dirname):
            filename_without_ext = '.'.join(filename.split('.')[:-1])
            if filename_without_ext in set_of_seen_files:
                continue
            if not hasattr(self, 'data_holder'):
                data_holder = SFSDataHolder(None)
            else:
                data_holder = copy.deepcopy(self.data_holder)
            data_holder.filename = os.path.join(dirname, filename)
            try:
                data = engine.read_data(data_holder)
                all_boot.append(data)
                if return_filenames:
                    filenames.append(filename)
                set_of_seen_files.add(filename_without_ext)
            except ValueError:
                pass
        self._bootstrap_data = all_boot
        if return_filenames:
            return zip(filenames, all_boot)
        return all_boot

    @property
    def inner_data(self):
        if not hasattr(self, '_inner_data'):
            self._inner_data = self.read_data()
        return self._inner_data

    @property
    def bootstrap_data(self):
        if not hasattr(self, '_bootstrap_data'):
            self._bootstrap_data = self.read_bootstrap_data(self)
        return self._bootstrap_data

    def update_from_file(self, param_file, extra_param_file=None):
        """
        Updates settings by reading new from files.

        :param param_file: File with base parameters.
        :param extra_param_file: File with extra parameters.
        """
        # Load all values
        if param_file is None and extra_param_file is None:
            return self
        loaded_dict = {}
        if param_file is not None:
            with open(param_file, encoding="utf-8") as fl:
                loaded_dict = ruamel.yaml.load(fl,
                                               ruamel.yaml.RoundTripLoader)
        if extra_param_file is not None:
            with open(extra_param_file, encoding="utf-8") as fl:
                extra_dict = ruamel.yaml.load(fl,
                                              ruamel.yaml.RoundTripLoader)
            loaded_dict = {**loaded_dict, **extra_dict}

        for key in loaded_dict:
            attr_name = key.lower().strip()
            attr_name = attr_name.replace(" ", "_")
            attr_name = attr_name.replace("'", "_")
            attr_name = attr_name.replace("__", "_")
#            print(attr_name)
            if attr_name in CHANGED_IDENTIFIERS:
                attr_name = CHANGED_IDENTIFIERS[attr_name]
                setting_name = attr_name.replace("_", " ").capitalize()
                msg = ""
                if attr_name == "ancestral_size_as_parameter":
                    msg = f" (`{setting_name}` = not `key`)"
                    assert isinstance(loaded_dict[key], bool)
                    loaded_dict[key] = not loaded_dict[key]
                warnings.warn(f"Setting `{key}` is renamed in 2 version of "
                              f"GADMA to `{setting_name}`. It is successfully"
                              f" read.{msg}")

            if not hasattr(self, attr_name):
                if attr_name in DEPRECATED_IDENTIFIERS:
                    warnings.warn(f"Setting `{key}` was deprecated in 2 "
                                  "version of GADMA. If you have not set it "
                                  "in purpose, ignore this warning.")
                    continue
                else:
                    raise AttributeError(f"Unknown identifier: `{key}`.")
            self.__setattr__(attr_name, loaded_dict[key])
        return self

    @staticmethod
    def from_file(param_file, extra_param_file=None):
        """
        Creates new object with settings from files.

        :param param_file: File with base parameters.
        :param extra_param_file: File with extra parameters.
        """
        obj = SettingsStorage()
        return obj.update_from_file(param_file, extra_param_file)

    def to_files(self, params_file, extra_params_file):
        """
        Saves current options to files.

        :param params_file: File with base parameters.
        :param extra_param_file: File with extra parameters.
        """
        # TODO comments with default values
        known_missed_attrs = ['data_holder', 'model_func',
                              'const_of_time_in_drawing', 'bootstrap_data',
                              'inner_data', 'number_of_populations',
                              'X_init', 'Y_init', 'initial_structure_unit',
                              'test', 'resume_from_settings',
                              'generate_x_transform']
        saved_attrs = []
        for filename, template in zip([params_file, extra_params_file],
                                      [PARAM_TEMPLATE, EXTRA_PARAM_TEMPLATE]):
            with open(template, encoding="utf-8") as fl:
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
                if (isinstance(value, numbers.Integral) and
                        not isinstance(value, bool)):
                    value = int(value)
                elif (isinstance(value, numbers.Real) and
                        not isinstance(value, bool)):
                    value = float(value)
                if isinstance(value, list):
                    for i in range(len(value)):
                        if (isinstance(value[i], numbers.Integral) and
                                not isinstance(value, bool)):
                            value[i] = int(value[i])
                        elif (isinstance(value[i], numbers.Real) and
                                not isinstance(value, bool)):
                            value[i] = float(value[i])

                loaded_template[key_name] = value
                saved_attrs.append(attr_name)

            # For representation of None's and bools
            def my_represent_none(self, data):
                r = self.represent_scalar(u'tag:yaml.org,2002:null', u'Null')
                return r

            def my_represent_bool(self, data):
                if data:
                    value = u'True'
                else:
                    value = u'False'
                return self.represent_scalar(u'tag:yaml.org,2002:bool', value)
            ruamel.yaml.RoundTripRepresenter.add_representer(type(None),
                                                             my_represent_none)
            ruamel.yaml.RoundTripRepresenter.add_representer(bool,
                                                             my_represent_bool)
            with open(filename, 'w', encoding="utf-8") as fl:
                ruamel.yaml.dump(loaded_template, fl,
                                 default_flow_style=True,
                                 Dumper=ruamel.yaml.RoundTripDumper)
        # save missed attributes at the end of extra file
        final_dict = dict()
        with open(extra_params_file, 'a', encoding="utf-8") as fl:
            fl.write("\n#\tOther parameters of run without description:\n")
            for attr_name in set(dir(self) + dir(settings)):
                if (attr_name.startswith("_") or
                        attr_name.isupper() or
                        attr_name in known_missed_attrs or
                        attr_name in saved_attrs):
                    continue
                value = getattr(self, attr_name)
                if callable(value):
                    continue
                # get rid of numpy as yaml could not serialize it
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                final_dict[attr_name] = value
            fl.write(ruamel.yaml.dump(final_dict, default_flow_style=False))

    def get_global_optimizer(self):
        """
        Return object of global optimizer for optimization according to current
        settings.
        """
        opt = get_global_optimizer(self.global_optimizer)
        if self.global_optimizer.lower() == "genetic_algorithm":
            opt.gen_size = self.size_of_generation
            opt.n_elitism = self.n_elitism
            opt.p_mutation = self.p_mutation
            opt.p_crossover = self.p_crossover
            opt.p_random = self.p_random
            opt.mut_rate = self.mean_mutation_rate
            opt.mut_strength = self.mean_mutation_strength
            opt.const_mut_rate = self.const_for_mutation_rate
            opt.const_mut_strength = self.const_for_mutation_strength
            opt.eps = self.eps
            opt.n_stuck_gen = self.stuck_generation_number
        if (self.global_optimizer.lower() == "gpyopt_bayesian_optimization" or
                self.global_optimizer.lower() == "smac_bo_optimization"):
            opt.kernel_name = self.kernel
            opt.acquisition_type = self.acquisition_function

        opt.log_transform = self.global_log_transform
        opt.maximize = True
        if self.random_n_a:
            opt.random_type = 'custom'
            opt.custom_rand_gen = custom_generator
        return opt

    def get_local_optimizer(self):
        """
        Return object of local optimizer for optimization according to current
        settings.
        """
        ls = get_local_optimizer(self.local_optimizer)
        ls.maximize = True
        ls.log_transform = self.local_log_transform
        return ls

#    def get_linear_constrain(self, engine):
#        if (self.upper_bound_of_first_split is None and
#                self.upper_bound_of_second_split is None):
#            return None
#        inv_theta0 = engine.get_N_ancestral_from_theta(1.0)
#
#        A = list()
#        ub = list()
#        if (self.upper_bound_of_first_split is not None):
#            A1, b1 = engine.model.get_involved_for_split_time_vars(1)
#            A.append(A1)
#            ub.append(self.upper_bound_of_first_split /
#                      (2 * inv_theta0) - b1)
#        if (self.upper_bound_of_second_split is not None):
#            A2, b2 = engine.model.get_involved_for_split_time_vars(2)
#            A.append(A2)
#            ub.append(self.upper_bound_of_second_split /
#                      (2 * inv_theta0) - b2)
#        lb = - np.inf * np.ones(len(ub))
#        return LinearConstrainDemographics(np.array(A),
#                                           np.array(lb), np.array(ub),
#                                           engine, self.get_engine_args())

    def get_optimizers_kwargs(self):
        """
        Returns kwargs for optimizations. (`args` and `verbose`).
        """
        kwargs = {}
        kwargs['args'] = self.get_engine_args()
        kwargs['verbose'] = self.verbose
        kwargs['global_num_init_const'] = self.num_init_const
#        kwargs['linear_constrain'] = self.get_linear_constrain()
        return kwargs

    def get_optimizers_init_kwargs(self, variables=None):
        """
        Returns kwargs for first run of optimization. (`X_init` and `Y_init`).
        """
        return {'X_init': self.X_init, 'Y_init': self.Y_init}

    def get_engine_args(self, engine_id=None):
        """
        Returns `args` of :func:`engine.evaluate` function.
        """
        if engine_id is None:
            engine_id = self.engine
        if engine_id == 'dadi':
            args = (self.pts,)
        elif engine_id == "moments":
            args = (MomentsEngine.default_dt_fac,)
        else:
            args = ()
        return args

    def get_linear_constrain_for_model(self, model):
        """
        Returns linear constrain for model based of setted upper bound of
        splits. NOT WORKING.
        """
        if (self.upper_bound_of_first_split is None and
                self.upper_bound_of_second_split is None):
            return None
        A = list()
        ub = list()
        if (self.upper_bound_of_first_split is not None):
            A1, b1 = model.get_involved_for_split_time_vars(1)
            A.append(A1)
            ub.append(self.upper_bound_of_first_split / 2 - b1)
        if (self.upper_bound_of_second_split is not None):
            A2, b2 = model.get_involved_for_split_time_vars(2)
            A.append(A2)
            ub.append(self.upper_bound_of_second_split / 2 - b2)
        lb = - np.inf * np.ones(len(ub))
        return LinearConstrain(np.array(A), np.array(lb), np.array(ub))

    def get_model(self):
        """
        Returns demographic model to use according to current settings.
        """
        gen_time = self.time_for_generation
        theta0 = self.theta0
        mut_rate = self.mutation_rate
        rec_rate = self.recombination_rate
        if (self.initial_structure is not None and
                self.final_structure is not None):
            create_migs = not self.no_migrations
            create_sels = self.selection
            create_dyns = not self.only_sudden
            sym_migs = self.symmetric_migrations
            split_f = self.split_fractions
            migs_mask = self.migration_masks
            create_inbr = self.inbreeding
            model = StructureDemographicModel(
                self.initial_structure,
                self.final_structure,
                has_anc_size=self.ancestral_size_as_parameter,
                has_migs=create_migs,
                has_sels=create_sels,
                has_dyns=create_dyns,
                sym_migs=sym_migs,
                frac_split=split_f,
                migs_mask=migs_mask,
                gen_time=gen_time,
                theta0=theta0,
                mutation_rate=mut_rate,
                recombination_rate=rec_rate,
                has_inbr=create_inbr
            )
            constrain = self.get_linear_constrain_for_model(model)
            model.linear_constrain = constrain
            return model
        elif ((self.custom_filename is not None or
                self.model_func is not None) and
                self.lower_bound is not None and
                self.upper_bound is not None):
            # get_variables take bounds equal to None into account
            lower_bound = self.lower_bound
            upper_bound = self.upper_bound
            if not hasattr(super(SettingsStorage, self), "lower_bound"):
                lower_bound = None

            if not hasattr(super(SettingsStorage, self), "upper_bound"):
                upper_bound = None

            variables = get_variables(self.parameter_identifiers,
                                      self.lower_bound, self.upper_bound,
                                      engine=self.engine)
            if all([
                self.ancestral_size_as_parameter,
                self.fixed_ancestral_size is None,
                self.engine == "momentsLD"
            ]):
                variables.insert(0, PopulationSizeVariable("Nanc",
                                                           units="physical"))

            if self.model_func is not None:
                return CustomDemographicModel(
                    function=self.model_func,
                    variables=variables,
                    gen_time=gen_time,
                    theta0=theta0,
                    mutation_rate=mut_rate,
                    recombination_rate=rec_rate,
                    fixed_anc_size=self.fixed_ancestral_size,
                    has_anc_size=self.ancestral_size_as_parameter
                )
            module_name = module_name_from_path(self.custom_filename)
            spec = importlib.util.spec_from_file_location(module_name,
                                                          self.custom_filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return CustomDemographicModel(
                function=module.model_func,
                variables=variables,
                gen_time=gen_time,
                theta0=theta0,
                mutation_rate=mut_rate,
                recombination_rate=rec_rate,
                fixed_anc_size=self.fixed_ancestral_size,
                has_anc_size=self.ancestral_size_as_parameter
            )

        elif self.custom_filename is None and self.model_func is not None:
            return CustomDemographicModel(
                function=self.model_func,
                variables=None,
                gen_time=gen_time,
                theta0=theta0,
                mutation_rate=mut_rate,
                recombination_rate=rec_rate,
                fixed_anc_size=self.fixed_ancestral_size,
                has_anc_size=self.ancestral_size_as_parameter
            )
        else:
            raise ValueError("Some settings are missed so no model is "
                             "generated")

    def Nanc_will_be_available(self):
        if self.ancestral_size_as_parameter:
            return True
        mu_is_set = self.mutation_rate is not None
        L_is_set = self.sequence_length is not None
        if mu_is_set and L_is_set:
            return True
        if self.theta0 is not None:
            return True
        return False

    def is_valid(self):
        """
        Checks that settings are fine. Rises errors if something is not right.
        """
        if (self.input_data is None and
                self.resume_from is None):
            raise AttributeError("Input file is required. It could be set by "
                                 "-i/--input option or via parameters file.")
        if (self.output_directory is None and
                self.resume_from is None):
            raise AttributeError(
                "Output directory is required. It could be set by -o/--output "
                "option or via parameters file."
            )
        assert self.output_directory is not None

        ensure_dir_existence(self.output_directory,
                             check_emptiness=True)
        if not self.Nanc_will_be_available() and not self.relative_parameters:
            self.relative_parameters = True
            warnings.warn(
                "Parameters will be in genetic units (Relative parameters). "
                "Engines dadi and moments require mutation rate and sequence "
                "length for unit translation"
            )

        if self.inbreeding:
            if self.projections is not None:
                warnings.warn(
                    "For correct inference of the inbreeding input data should"
                    " not be projected. Projections were taken as"
                    f" {self.projections}, please check that data is not "
                    "downsized."
                )
            if self.engine != "dadi":
                raise ValueError(
                    "Please check your engine. If you want to infer "
                    "inbreeding please change engine to `dadi`"
                )
        if (self.recombination_rate is not None and
                self.recombination_rate != 0):
            if self.engine in ['moments', 'dadi']:
                warnings.warn(f"Engine {self.engine} will ignore "
                              "not-zero recombination rate.")
        # check for custom model and engine to draw
        if self.custom_filename is not None:
            if self.model_plot_engine != self.engine:
                if get_engine(self.engine).can_draw_model:
                    warnings.warn(
                        "Custom model was set, engine to draw models will be "
                        f"the same as for evaluations: {self.engine}"
                    )
                    self.model_plot_engine = self.engine
                else:
                    warnings.warn(
                        "Custom model could be drawn with the same engine that"
                        " will be used for evaluation only. However that "
                        f"engine {self.engine} cannot draw models. No picture"
                        "of model will be generated."
                    )

        if self.engine == "momi":
            if "Lin" in self.dynamics:
                warnings.warn("Momi engine does not support Linear size "
                              "function. It is removed from possible dynamics")
                self.dynamics = [dyn for dyn in self.dynamics if dyn != "Lin"]
            if not self.ancestral_size_as_parameter:
                warnings.warn(
                    "Momi engine need ancestral size as parameter. The option "
                    "`Ancestral size as parameter` is set to `True`"
                )
                self.ancestral_size_as_parameter = True
            if not self.no_migrations:
                warnings.warn(
                    "Momi engine does not support continous migrations. The "
                    "option `No migrations` is set to `True`"
                )
                self.no_migrations = True

            if self.selection:
                warnings.warn(
                    "Momi engine does not support inference of selection "
                    "coefficients. The option `Selection` is set to `False`"
                )
                self.selection = False

        if self.model_plot_engine == "momi":
            if not self.no_migrations or "Lin" in self.dynamics:
                warnings.warn(
                    "Momi was chosen to draw models. Mind the fact that it "
                    "does not support linear size change and does not draw "
                    "migrations"
                )
            if self.units_of_time_in_drawing.lower() not in ["generations",
                                                             "years"]:
                warnings.warn(
                    "Model plot engine momi does not draw time in "
                    f"{self.units_of_time_in_drawing}, units are switched "
                    "to `years`"
                )
                self.units_of_time_in_drawing = "years"

        if self.sequence_length is None and momi_available:
            warnings.warn("Code for momi will not be generated as `Sequence "
                          "length` is missed.")

        # Check for sequence length if we have several chrom lengths
        if isinstance(self.sequence_length, dict):
            if isinstance(self.data_holder, VCFDataHolder):
                chromosomes = extract_chromosomes_from_vcf(
                    self.data_holder.filename
                )
                if set(chromosomes) != set(self.sequence_length):
                    raise ValueError(
                        "Sequence length was set as a dict. However it "
                        "contains different or not all chromosomes than "
                        "VCF file.\n"
                        f"Sequence length:\t{set(self.sequence_length)}\n"
                        f"VCF file:\t{set(chromosomes)}")

        # Checks for momentsLD
        moments_ld_ok = True
        if not isinstance(self.data_holder, VCFDataHolder):
            moments_ld_ok = False
            reason = "VCF input data is required."
        else:
            chromosomes = extract_chromosomes_from_vcf(
                self.data_holder.filename
            )
            if (len(chromosomes) > 1 and
                    not isinstance(self.sequence_length, dict)):
                moments_ld_ok = False
                reason = "There are more than one chromosome in VCF file and"\
                         " only total length in `Sequence length`, please, "\
                         "specify length for chromosomes separately via dict"
        if not moments_ld_ok:
            if self.engine != "momentsLD":
                warnings.warn("Code for momentsLD will not be generated as: "
                              f"{reason}")
            else:
                raise ValueError("MomentsLD requirements are not satisfied: "
                                 f"{reason}")

        for engine, is_available in zip(["demes", "momi"],
                                        [demes_available, momi_available]):
            if is_available:
                if not self.Nanc_will_be_available():
                    warnings.warn(
                        f"Code for {engine} engine will not be generated as "
                        "ancestral size will be missed in the dem. model. "
                        "The following options should be set to enable it:\n"
                        f"`Ancestral size as parameter`: True "
                        f"(got `self.ancestral_size_as_parameter`)\nor\n"
                        f"`Mutation rate` (got {self.mutation_rate})\n"
                        f"`Sequence length` (got {self.sequence_length})\nor\n"
                        f"`Theta0` (got {self.theta0})")

        if self.ld_kwargs is not None:
            if self.engine != "momentsLD":
                warnings.warn(
                    "You cannot set ld_kwargs argument if you don't use "
                    f"momentsLD engine. It will be ignored. "
                    f"Your current engine is {self.engine}."
                )
            else:
                import moments.LD
                Full_arg_spec = inspect.getfullargspec(
                    moments.LD.Parsing.compute_ld_statistics)
                args_parsing_ld = Full_arg_spec[0]
                for key in self.ld_kwargs:
                    if key not in args_parsing_ld:
                        raise KeyError(
                            "Computing_ld_stats function hasn't "
                            f"argument {key}! "
                            f"Check your param_file "
                            f"and remove unexpected args."
                        )

        if self.engine == "momentsLD":
            if all([
                not self.ancestral_size_as_parameter,
                not (self.fixed_ancestral_size and self.custom_filename)
            ]):
                warnings.warn(
                    "Moments.LD engine need ancestral size as parameter. "
                    "The option "
                    "`Ancestral size as parameter` is set to `True`"
                )
                self.ancestral_size_as_parameter = True

            if self.mutation_rate is None:
                raise ValueError(
                    "momentsLD engine needs mutation rate "
                    "parameter for correct work"
                )

            if self.data_holder.projections is None:
                if self.data_holder.population_labels is None:
                    (self.data_holder.projections,
                     self.data_holder.population_labels
                     ) = check_and_return_projections_and_labels(
                        self.data_holder)
                else:
                    (self.data_holder.projections,
                     labels) = check_and_return_projections_and_labels(
                        self.data_holder)
