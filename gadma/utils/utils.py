from functools import wraps
import time
import numpy as np
from collections import namedtuple
import sys
import os
import copy


def logarithm_transform(x):
    if isinstance(x, (int, float, np.integer, np.float)):
        x = [x]
    if not isinstance(x, (list, np.ndarray)):
        return x
    if isinstance(x, np.ndarray):
        x = x.astype(float)
    return np.log(x)


def exponent_transform(x):
    if isinstance(x, (int, float, np.integer, np.float)):
        x = [x]
    if not isinstance(x, (list, np.ndarray)):
        return x
    if isinstance(x, np.ndarray):
        x = x.astype(float)
    return np.exp(x)


def ident_transform(x):
    return x


def choose_by_weight(X, weights, nsample):
    """
    Choose `nsample` samples from `X` according to `weights`.
    The greater weight is the greater the probability to choose sample is.

    Note: if weights is None then choice will be uniform
    """
    if weights is None:
        weights = np.ones(len(X))
    p = np.array(weights)
    p /= np.sum(p)
    return np.random.choice(X, size=nsample, replace=False)


def sort_by_other_list(x, y, reverse=False):
    """
    Sort x and y according to values in y.
    """
    sort_zip = sorted(zip(x, y), key=lambda p: p[1])
    return [p[0] for p in sort_zip], [p[1] for p in sort_zip]


def fix_args(f, *args):
    """
    Fixes argumets of function.
    :param f: function such that f(x, *args)
    :param args: tuple of function arguments.
    :returns: function that will take only x as argument.
    """
    @wraps(f)
    def fix_args_wrapper(x):
        return f(x, *args)
    return fix_args_wrapper


class CacheInfo(object):
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.cache = {}
        self.all_calls = []


def lru_cache(func):
    """
    Our lru cache. We want to get cache itself while functools.lru_cache
    could not do it.
    Please, be carefull as it could be some attributes named the same way
    and it will be ruined. We use it for our decorator :func:`cache_func`.
    """
    func.cache_info = CacheInfo()

    @wraps(func)
    def lru_cache_wrapper(*args):
        try:
            ret = func.cache_info.cache[args]
            func.cache_info.hits += 1
        except KeyError:
            ret = func(*args)
            func.cache_info.cache[args] = ret
            func.cache_info.misses += 1
        func.cache_info.all_calls.append([args, ret])
        return ret
    return lru_cache_wrapper


def cache_func(f):
    """
    Cashes function with one argument.
    :param f: function such that f(x).
    :returns: function that is cashed.
    """
    @lru_cache
    @wraps(f)
    def tuple_wrapper(x_tuple):
        return f(np.array(x_tuple, dtype=object))

    @wraps(tuple_wrapper)
    def cache_wrapper(x):
        return tuple_wrapper(tuple(x))

    cache_wrapper.cache_info = tuple_wrapper.cache_info
    return cache_wrapper


# def cache_func_2d(f):
#     """
#     Cashes function with two arguments.
#     :param f: function such that f(x, y).
#     :returns: function that is cashed.
#     """
#     @lru_cache
#     @wraps(f)
#     def tuple_wrapper(x_y_tuple):
#         x, y = x_y_tuple
#         if isinstance(y, tuple) and isinstance(y[0], tuple):
#             y = np.array(y)
#         return f(x, y)
#
#     @wraps(tuple_wrapper)
#     def cache_wrapper(x, y):
#         y_tuple = tuple(y)
#         if (isinstance(y, (list, np.ndarray)) and
#                 isinstance(y[0], (list, np.ndarray))):
#             for i in range(len(y_tuple)):
#                 y_tuple[i] = tuple(y_tuple[i])
#         return tuple_wrapper(tuple(x), y_tuple)
#
#     cache_wrapper.cache_info = tuple_wrapper.cache_info
#     return cache_wrapper


def nan_fval_to_inf(f):
    """
    Wrappes function to return infinity instead nan.
    """
    @wraps(f)
    def nan_fval_to_inf_wrapper(x):
        y = f(x)
        if y is None or np.isnan(y):
            return np.inf
        return y
    return nan_fval_to_inf_wrapper


def eval_wrapper(f, eval_file=None):
    """
    Returns good function for optimization. Each evaluation of function will
    be written in file. If needed function will be cached.
    :param f: function. Is called as f(x, *args).
    :param args: tuple of arguments.
    :param eval_file: file to write evaluations.
    :param cache: if True then function will be cached.
    """
    time_init = time.time()
    first_line = '\t'.join(['Time of evaluation start', 'Function value',
                            'Parameters values', 'Evaluation time'])
    if eval_file is not None:
        if not check_file_existence(eval_file):
            open(eval_file, 'w').close()
        if not os.stat(eval_file).st_size == 0:
            with open(eval_file, 'r') as fl:
                line = next(fl)
        if os.stat(eval_file).st_size == 0 or line.strip() != first_line:
            with open(eval_file, 'a') as fl:
                print(first_line, file=fl, sep='\t')

    @wraps(f)
    def eval_wrapper_f(x):
        time_start = time.time()
        y = f(x)
        time_end = time.time()
        if eval_file is not None:
            with open(eval_file, 'a') as fl:
                print(time_start - time_init, y, list(x),
                      time_end - time_start, file=fl, sep='\t')
        return y
    return eval_wrapper_f


class WeightedMetaArray(np.ndarray):
    """Array with metadata."""
    def __new__(cls, array, dtype=None, order=None):
        obj = np.asarray(np.array(array, dtype=object),
                         dtype=dtype, order=order).view(cls)
        obj.metadata = ''
        obj.weights = np.ones(obj.shape)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, 'metadata', [{}]*(obj.ndim+1))
        self.weights = getattr(obj, 'weights', [{}]*(obj.ndim+1))

    def __str__(self):
        super_str = super(WeightedMetaArray, self).__str__()
        if hasattr(self, 'metadata'):
            return super_str + '\t' + self.metadata
        return super_str

    def __repr__(self):
        super_str = super(WeightedMetaArray, self).__repr__()
        if hasattr(self, 'metadata'):
            return super_str + '\t' + self.metadata
        return super_str


def update_by_one_fifth_rule(value, const, was_improved):
    if was_improved:
        return value * const
    return value / (const) ** (0.25)


def abspath(path):
    return os.path.abspath(os.path.expanduser(path))


def check_file_existence(path_to_file):
    return os.path.exists(path_to_file) and os.path.isfile(path_to_file)


def check_dir_existence(path_to_dir):
    return os.path.exists(path_to_dir) and os.path.isdir(path_to_dir)


def ensure_file_existence(path_to_file):
    filename = abspath(path_to_file)
    if not check_file_existence(filename):
        open(filename, 'w').close()
    return filename


def ensure_dir_existence(path_to_dir, check_emptiness=False):
    dirname = abspath(path_to_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if os.listdir(dirname) != [] and check_emptiness:
        raise RuntimeError(f"Directory {path_to_dir} is not empty\nYou can "
                           f"write:  rm -rf {dirname}\t to remove directory.")
    return dirname


class StdAndFileLogger(object):
    def __init__(self, log_filename, silent=False):
        self.terminal = sys.stdout
        self.log_filename = log_filename
        self.silent = silent
        if not os.path.exists(self.log_filename):
            open(self.log_filename, 'w'). close()

    def write(self, message):
        if not self.silent:
            self.terminal.write(message)
            self.terminal.flush()
        with open(self.log_filename, 'a') as fl:
            fl.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_aic_score(n_params, log_likelihood):
    return 2 * n_params - 2 * log_likelihood


def get_claic_score(engine, x0, boots,
                    args=(), log_likelihood=None, return_eps=False):
    """Calculate CLAIC score for the model."""
    if log_likelihood is None:
        log_likelihood = engine.evaluate(x0, *args)
    eps = 1e-5
    claic_score = None
    while eps <= 1e-2:
        try:
            claic_component = engine.get_claic_component(x0, boots,
                                                         *args, eps)
            claic_score = 2 * claic_component - 2 * log_likelihood
            break
        except np.linalg.linalg.LinAlgError as e:
            if str(e) == 'Singular matrix':
                eps *= 10
            else:
                print(e)
                raise e
    if return_eps:
        return claic_score, eps
    else:
        return claic_score


# Printing functions
def float_repr(value, precision=5):
    if value < 10**(-precision):
        return f"{value:.2e}"
    return f"{round(value, precision)}"


def variables_values_repr(variables, values):
    val_repr = [float_repr(val) if isinstance(val, float) else val
                for val in values]
    var_val = zip(variables, val_repr)
    x_repr = ",\t".join([f"{var.name}={val}" for var, val in var_val])
    x_repr = f"({x_repr})"
    return x_repr


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{bcolors.WARNING}{category.__name__}: {message}"\
           f"{bcolors.ENDC} ({filename}:{lineno})\n"
