import logging
import warnings

import moments
import gadma
import os
import sys
import importlib

import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter,\
                                        UniformFloatHyperparameter

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bohb_facade import BOHB4HPO
from smac.facade.smac_ac_facade import SMAC4AC
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

import multiprocessing
from multiprocessing import Pool

INSTANCES = [['2_DivMig_5_Sim'], ['2_ExpDivNoMig_5_Sim'],
             ['2_BotDivMig_8_Sim'], ['2_YRI_CEU_6_Gut']]

ITER_PER_PAR = 1000

def load_module(dirname, filename):
    save_dir = os.path.abspath(".")
    os.chdir(os.path.abspath(dirname))
    sys.path.append(".")#dirname)
    if "demographic_model" in sys.modules:
        del sys.modules["demographic_model"]
    module = importlib.import_module(
        os.path.join(dirname, filename).replace('/', '.').rstrip('.py'))
    sys.path = sys.path[:-1]
    os.chdir(save_dir)
    return module

# Target Algorithm
def gadma_from_cfg(cfg, seed, instance):
    """
    Creates gadma run from the proposed configuration.

    :param outputdir: directory with output where gadma could save its output.
    :param cfg: configuration of hyperparameters.
    :param seed: random seed.
    :param instance: instance to run gadma on.
    """

    # Get info from instance
    module = load_module(os.path.join("demographic_inference_data", instance),
                         "main_script.py")
    data = module.data        
    model_func = module.model_func
    lower_bound = module.lower_bound
    upper_bound = module.upper_bound
    p_ids = module.par_labels
    n_par = len(lower_bound)

    engine = 'moments'

    # Remember seed
    np.random.seed(seed)

    # Run gadma
    res = gadma.Inference.optimize_ga(data, model_func, engine, args=(),
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound,
                                      p_ids=p_ids,
                                      X_init=None, Y_init=None,
                                      gen_size=cfg['gen_size'],
                                      mut_strength=cfg['mut_strength'],
                                      const_mut_strength=cfg['const_mut_str'],
                                      mut_rate=cfg['mut_rate'],
                                      const_mut_rate=cfg['const_mut_rate'],
                                      eps=cfg['eps'],
                                      n_stuck_gen=cfg['n_stuck_gen'],
                                      n_elitism=int(cfg['gen_size']*cfg['p_elitism']),
                                      p_mutation=cfg['p_mutation'],
                                      p_crossover=cfg['p_crossover'],
                                      p_random=cfg['p_random'],
                                      ga_maxiter=None,
                                      ga_maxeval=n_par*ITER_PER_PAR,
                                      local_optimizer=None,
                                      ls_maxiter=None,
                                      ls_maxeval=None,
                                      verbose=0,
                                      callback=None,
                                      save_file=None,
                                      eval_file=None,
                                      report_file=None)
    
    return - res.y
logging.getLogger("Inference").setLevel(logging.DEBUG)
logger = logging.getLogger("SMAC")
logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output
logging.getLogger().setLevel(logging.INFO)

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible parameters
gen_size = CategoricalHyperparameter('gen_size', choices=[10, 50, 100, 200],
                                     default_value=10)
mut_strength = UniformFloatHyperparameter('mut_strength', 1e-15, 1,
                                          default_value=0.2)
const_mut_strength = UniformFloatHyperparameter('const_mut_str', 1, 2,
                                                default_value=1.1)
mut_rate = UniformFloatHyperparameter('mut_rate', 1e-15, 1,
                                      default_value=0.2)
const_mut_rate = UniformFloatHyperparameter('const_mut_rate', 1, 2,
                                            default_value=1.2)
n_stuck_gen = CategoricalHyperparameter('n_stuck_gen', choices=[20, 50, 100],
                                        default_value=100)
eps = UniformFloatHyperparameter('eps', 1e-6, 1,
                                 default_value=1e-2)
p_elitism = UniformFloatHyperparameter('p_elitism', 0, 1,
                                       default_value=0.2)
p_mutation = UniformFloatHyperparameter('p_mutation', 0, 1,
                                        default_value=0.3)
p_crossover = UniformFloatHyperparameter('p_crossover', 0, 1,
                                         default_value=0.3)
p_random = UniformFloatHyperparameter('p_random', 0, 1,
                                       default_value=0.2)

# Add the parameters to configuration space
cs.add_hyperparameters([gen_size, mut_strength, const_mut_strength,
                        mut_rate, const_mut_rate, n_stuck_gen, eps,
                        p_elitism, p_mutation, p_crossover, p_random])

intensifier_kwargs = {'n_seeds': 50}

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": 10000,  # max. number of function evaluations; for this example set to a low number
                     "cs": cs,  # configuration space
                     "deterministic": False,
                     "shared_model": True,
                     "output_dir": "smac3-output/",
                     "input_psmac_dirs": "smac3-output/run_*",
                     "instances": INSTANCES
                     })

i = int(sys.argv[1])
# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = BOHB4HPO(scenario=scenario, rng=np.random.RandomState(i),
                tae_runner=gadma_from_cfg, run_id = i,
                intensifier_kwargs=intensifier_kwargs)

incumbent = smac.optimize()

def_costs = []
for i in INSTANCES:
    cost = smac.get_tae_runner().run(cs.get_default_configuration(), i[0])[1]
    def_costs.append(cost)
print("Value for default configuration: %.4f" % (np.mean(def_costs)))

inc_costs = []
for i in INSTANCES:
    cost = smac.get_tae_runner().run(incumbent, i[0])[1]
    inc_costs.append(cost)
print("Optimized Value: %.4f" % (np.mean(inc_costs)))
