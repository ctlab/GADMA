from .cli import SettingsStorage
from .optimizers import GlobalOptimizerAndLocalOptimizer
from .data import SFSDataHolder
from .engines import get_engine
from .utils import ContinuousVariable, timeout
from . import utils
import warnings
import numpy as np


def load_data_from_dir(dirname, engine, projections=None,
                       population_labels=None, outgroup=None):
    """
    Load data of SFS type from the directory. Data is considered to be
    very consistent: for example, it could be bootstrap of one dataset. All
    data should have the same projections, pop labels and so on.

    :param dirname: Path to the directory with data.
    :param engine: Engine id for data loading. Could be one of the following:
                   - dadi
                   - moments
    :param projections: Sample size of data. If None it will be chosen
                        automatically.
    :param population_labels: Labels of populations in the data.
    :param outgroup: If True then there is outgroup represented in files. Then
                     unfolded SFS will be loaded if SFS is needed.
    """
    settings = SettingsStorage()
    settings.directory_with_bootstrap = dirname
    settings.engine = engine
    settings.data_holder = SFSDataHolder(None,
                                         projections=projections,
                                         outgroup=outgroup,
                                         population_labels=population_labels)
    return settings.read_bootstrap_data()


def get_claic_score(func_ex, all_boot, p0, data, engine=None, args=(),
                    eps=1e-2, pts=None):
    r"""
    Returns CLAIC score for demographic model with specified value of `eps`.

    :param func_ex: Custom function to evaluate demographic model.
                    Usually it is model_func function from generated code of
                    GADMA. It is run by calling func_ex(p, ns, \*args), where
                    p is values of parameters and ns - sample sizes.
    :param all_boot: List of bootstrapped data for CLAIC evaluation.
    :param p0: Values of parameters for ``func_ex`` demographic model.
    :param data: Original data for CLAIC evaluation. It is data that was used
                 for demographic inference.
    :param engine: Engine id for likelihood evaluations. Could be one of the
                   following:
                   - dadi
                   - moments
    :param args: Arguments of ``func_ex`` function.
    :param eps: Step size for Hessian and gradient calculations. Usually is
                between 1e-5 and 1e-2. The smaller eps is the more accurate
                CLAIC value is.
    :param pts: Deprecated parameter from GADMA version 1. If is set then
                warning is printed.

    returns: None if failed to get CLAIC due to singular matrix of Hessian.\
             Could be solved by increasing value of ``eps``.

    note: There differencies between GADMA v1 and GADMA v2, there is some\
          backward compatibility, but sometimes errors could be raised.
    """
    # The following code exists because of backward compatibility with first
    # version of GADMA, where parameters were:
    warning_msg = ("It looks like get_claic_score function is used from GADMA "
                   " of version 1.")
    if pts is not None and engine is None:
        engine = 'dadi'
        warnings.warn(warning_msg + " Deprecated argument pts is used - dadi "
                      "engine is chosen.")
    if (engine is None or
            (isinstance(engine, (list, np.ndarray)) and len(engine) == 3)):
        if engine is not None:
            pts = engine
        engine = 'moments'
        if pts is not None:
            engine = 'dadi'
        warnings.warn(warning_msg + " The 5th argument (engine in GADMA v2 vs"
                      " pts in GADMA v1) argument is not specified (None). If"
                      " some other error will happen next then please specify"
                      f" engine. Engine: {engine}, pts: {pts}, eps: {eps}")

    if not isinstance(args, tuple) and isinstance(args, float):
        eps = args
        args = ()
    # End of backward compatibility of versions.

    settings = SettingsStorage()
    settings.engine = engine
    if len(args) > 0:
        func_ex = utils.fix_args(func_ex, args)
    settings.model_func = func_ex
    engine_obj = get_engine(engine)
    engine_obj.set_data(data)
    engine_obj.set_model(settings.get_model())
    variables = list()
    for i, x in enumerate(p0):
        variables.append(ContinuousVariable(f"var_{i}", domain=[x - 1, x + 1]))
    engine_obj.model.variables = variables
    if pts is not None:
        settings.pts = pts
    return utils.get_claic_score(engine_obj, p0, all_boot,
                                 settings.get_engine_args())


def optimize_ga(data, model_func, engine, args=(),
                lower_bound=None, upper_bound=None, p_ids=None,
                X_init=None, Y_init=None, maxtime_per_eval=None, num_init=50,
                gen_size=10, mut_strength=0.2, const_mut_strength=1.1,
                mut_rate=0.2, const_mut_rate=1.2, eps=1e-2, n_stuck_gen=100,
                n_elitism=2, p_mutation=0.3, p_crossover=0.3, p_random=0.2,
                ga_maxiter=None, ga_maxeval=None,
                local_optimizer='BFGS_log', ls_maxiter=None, ls_maxeval=None,
                verbose=1, callback=None,
                save_file=None, eval_file=None, report_file=None):
    r"""
    Runs genetic algorithm optimizer in order to find best values of
    parameters for ``model_func`` demographic model from ``data``.

    :param data: Data for demographic inference.
    :param model_func: Function to use for demographic inference that
                       simulates SFS to compare it with ``data`` with
                       log-likelihood. Is called by model_func(p, ns, \*args),
                       where p is values of parameters, ns - sample size and
                       args - other arguments.
    :param engine: Engine id for demographic inference. Could be one of the
                   following:
                   - 'dadi'
                   - 'moments'
    :param args: Arguments for ``model_func`` function. It is `pts` for
                 `dadi` engine and could be `dt_fac` (or nothing) for
                 `moments` engine.
    :param lower_bound: Lower bound for each demographic parameter.
    :type lower_bound: list
    :param upper_bound: Upper bound for each demographic parameter.
    :type upper_bound: list
    :param p_ids: Parameter identifiers for demographic parameters. Each
                  identifier should start with one of the following letters:
                  - n or N for size of populations;
                  - t or T for time;
                  - m or M for migration rates;
                  - s or S for fractional parameters (between 0 and 1).

                  For example valid identifiers are:
                  ['nu1F', 'nu2B', 'nu2F', 'm', 'Tp', 'T']
    :type p_ids: list
    :param X_init: list of initial example parameters. GA will be initialized
                   by those values. It could be used for combinations of
                   optimizations or for restart.
    :type X_init: list of lists
    :param Y_init: value of log-likelihood for values in X_init.
    :type Y_init: list
    :param maxtime_per_eval: Maximum time per log-likelihood evaluation.
    :type maxtime_per_eval: int
    :param num_init: Number of initial points to start Genetic algorithm.
    :type num_init: int
    :param gen_size: Size of generation of genetic algorithm. That is the
                     number of individuals/solutions on each step of GA.
    :type gen_size: int
    :param mut_strength: Mean fraction of parameters for mutation in GA.
    :type mut_strength: float
    :param const_mut_strength: Const to change ``mut_strength`` during
                               GA according to one-fifth rule.
    :type const_mut_strength: float
    :param mut_rate: Mean rate of mutation in GA.
    :type mut_rate: float
    :param const_mut_rate: Const to change ``mut_rate`` during GA.
    :type const_mut_rate: float
    :param eps: const for model's log likelihood compare.
                Model is better if its log-likelihood is greater than the
                log-likelihood of another model by epsilon.
    :type eps: float
    :param n_stuck_gen: Number of iterations for GA stopping: GA stops when
                        it can't improve model during n_stuck_gen generations.
    :type n_stuck_gen: int
    :param n_elitism: Number of best models from the previous generation in GA
                      that will be taken to the new generation.
    :type n_elitism: int
    :param p_mutation: probability of mutation in one generation of GA.
    :type p_mutation: float
    :param p_crossover: probability of crossover in one generation of GA.
    :type p_crossover: float
    :param p_random: Probability to generate an individual randomly in one
                     generation of GA.
    :type p_random: float
    :param ga_maxiter: Maximum number of generations in GA.
    :type ga_maxiter: int
    :param ga_maxeval: Maximum number of function evaluations in GA.
    :type ga_maxeval: int
    :param local_optimizer: Local optimizer name to run for best solution of
                            GA. Could be None or one of:
                            * 'BFGS'
                            * 'BFGS_log'
                            * 'L-BFGS-B'
                            * 'L-BFGS-B_log'
                            * 'Powell'
                            * 'Powell_log'
                            * 'Nelder-Mead'
                            * 'Nelder-Mead_log'
    :type local_optimizer: str
    :param ls_maxiter: Maximum number of iterations in local optimization.
    :type ls_maxiter: int
    :param ls_maxeval: Maximum number of function evaluations in local
                       optimization.
    :type ls_maxeval: int
    :param verbose: Verbose of output.
    :type verbose: int
    :param callback: callback to call during optimizations. (callback(x, y))
    :param save_file: File for save GA state on current generation.
    :type save_file: str
    :param eval_file: File to save all evaluations during GA and local
                      optimization.
    :param report_file: File to write reports of GA and local optimization.
    """
    settings = SettingsStorage()
    settings._inner_data = data
    settings.model_func = model_func
    settings.engine = engine
    settings.get_engine_args = args
    settings.lower_bound = lower_bound
    settings.upper_bound = upper_bound
    settings.parameter_identifiers = p_ids
    settings.size_of_generation = gen_size
    settings.mean_mutation_rate = mut_rate
    settings.const_for_mutation_rate = const_mut_rate
    settings.mean_mutation_strength = mut_strength
    settings.const_for_mutation_strength = const_mut_strength
    settings.eps = eps
    settings.stuck_generation_number = n_stuck_gen
    settings.n_elitism = n_elitism
    settings.p_mutation = p_mutation
    settings.p_crossover = p_crossover
    settings.p_random = p_random
    settings.local_optimizer = local_optimizer

    # We could use CoreRun here but instead we will launch everything manually
    engine = get_engine(engine)
    model = settings.get_model()
    engine.set_model(model)
    engine.set_data(data)

    f = engine.evaluate
    if maxtime_per_eval is not None:
        f = timeout(f, maxtime_per_eval)
    variables = model.variables

    global_optimizer = settings.get_global_optimizer()
    local_optimizer = settings.get_local_optimizer()

    opt = GlobalOptimizerAndLocalOptimizer(global_optimizer,
                                           local_optimizer)
    result = opt.optimize(f, variables, args=args, global_num_init=num_init,
                          X_init=X_init, Y_init=Y_init, local_options={},
                          global_maxiter=ga_maxiter, local_maxiter=ls_maxiter,
                          global_maxeval=ga_maxeval, local_maxeval=ls_maxeval,
                          verbose=verbose, callback=callback,
                          eval_file=eval_file, report_file=report_file,
                          save_file=save_file)
    return result
