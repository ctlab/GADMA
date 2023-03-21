from ..utils import PopulationSizeVariable, TimeVariable, MigrationVariable,\
                    DynamicVariable, FractionVariable, GrowthRateVariable,\
                    SelectionVariable
from .. import moments_available, dadi_available
from .. import demes_available, demesdraw_available

# Main options. Output and input.
output_directory = None
input_data = None
bed_file = None
# input_data = None
# number_of_populations = None
population_labels = None
projections = None
outgroup = None
sequence_length = None
linked_snp_s = True
directory_with_bootstrap = None
bed_files_dir = None
non_ascertained_pops = None  # special for momi
# boots = None

# Pipeline
theta0 = None
time_for_generation = None
# multinom = None
only_sudden = False
pts = None
dadi_extrapolation = "make_extrap_log_func"
if moments_available:
    engine = 'moments'
elif dadi_available:
    engine = 'dadi'

model_plot_engine = "moments"
if demes_available and demesdraw_available:
    model_plot_engine = "demes"

relative_parameters = False
ancestral_size_as_parameter = False
no_migrations = False
symmetric_migrations = False
split_fractions = True
migration_masks = None
inbreeding = False
selection = False
dominance = False

# LD params
ld_kwargs = None
region_len = 6400000
preprocessed_data = None

# Custom model
custom_filename = None
model_func = None
lower_bound = None
upper_bound = None
parameter_identifiers = None
fixed_ancestral_size = None

# Structure of models
initial_structure_unit = 1
initial_structure = None
final_structure = None

# Time bounds
upper_bound_of_first_split = None
upper_bound_of_second_split = None

# Glocal optimizer
global_optimizer = "Genetic_algorithm"

# GA options
num_init_const = 10
size_of_generation = 10

fractions = [0.3, 0.2, 0.3]
n_elitism = 3
p_mutation = 0.2
p_crossover = 0.3
p_random = 0.2

mean_mutation_strength = 0.775539
const_for_mutation_strength = 1.302280

mean_mutation_rate = 0.273263
const_for_mutation_rate = 1.475288

stuck_generation_number = 100
eps = 1e-2

# BO options
kernel = "Matern52"
acquisition_function = "EI"

# just for logging evaluations
# output_log_file = None
# max_num_of_eval = None # maximum number of logll eval.
# num_init_pts = None # can get value from Inference.optimize_ga

# Local search
local_optimizer = 'BFGS_log'

# Printing and drawing
print_models_code_every_n_iteration = 0
silence = False
draw_models_every_n_iteration = 0
units_of_time_in_drawing = 'years'
const_of_time_in_drawing = 1.0
vmin = 1

number_of_repeats = 1
number_of_processes = 1
test = False
resume_from = None
only_models = False
generate_x_transform = False

# Extra parameters

# Bounds on models parameters. They are relative to N_A (!)
min_n = PopulationSizeVariable.default_domain[0]
max_n = PopulationSizeVariable.default_domain[1]
min_t = TimeVariable.default_domain[0]
max_t = TimeVariable.default_domain[1]
min_m = MigrationVariable.default_domain[0]
max_m = MigrationVariable.default_domain[1]
dynamics = list(DynamicVariable.default_domain)

# Parameters for local search alg
# ls_verbose = None
# ls_flush_delay = 0.5
# ls_epsilon = 1e-3
# ls_gtol = 1e-05
# ls_maxiter = None
# for hill climbing
# hc_mutation_rate = None
# hc_const_for_mutation_rate = None
# hc_stop_iter = None

# Options of mutation, crossing and random generating
random_n_a = False
# multinom_cross = False
# multinom_mutate = False

# Options of printing summary information about repeats
time_to_print_summary = 1  # min

verbose = 1

# Options of distributions
# distribution = 'normal'  # can be 'uniform'
# std = None  # std for normal dist

# Some options about drawing plots:
# matplotlib_available = False
# pil_available = False
# moments_available = False

X_init = None
Y_init = None
mutation_rate = None
recombination_rate = None
recombination_maps = None

global_maxiter = None
global_maxeval = None
global_log_transform = True
local_maxiter = None
local_maxeval = None
local_log_transform = True

# Additional constants
P_IDS = {'n': PopulationSizeVariable, 't': TimeVariable,
         'm': MigrationVariable, 'd': DynamicVariable, 's': FractionVariable,
         'f': FractionVariable, 'p': FractionVariable,
         'r': GrowthRateVariable, 'g': [GrowthRateVariable, SelectionVariable]}
LONG_NAME_2_SHORT = {"log-likelihood": "logLL",
                     "aic score": "aic",
                     "claic score": "claic"}
BASE_OUTPUT_DIR_PREFIX = "best_"
BASE_OUTPUT_DIR_PREFIX_FINAL = "best_"
LOCAL_OUTPUT_DIR_PREFIX = "current_best_"
LOCAL_OUTPUT_DIR_PREFIX_FINAL = "final_best_"
