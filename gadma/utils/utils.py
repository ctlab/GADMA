from functools import wraps
import time
import numpy as np
from collections import namedtuple
import sys
import os


def log_transform(x):
    if isinstance(x, (float, np.float)):
        return x
    return np.log(x)


def exp_transform(x):
    if isinstance(x, (float, np.float)):
        return x
    return np.exp(x)


def ident_transform(x):
    return x


def extract_args(func):
    @wraps(func)
    def wrapper(args):
        return func(*args)
    return wrapper


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


def fix_args(f, args):
    """
    Fixes argumets of function.
    :param f: function such that f(x, *args)
    :param args: tuple of function arguments.
    :returns: function that will take only x as argument.
    """
    @wraps(f)
    def wrapper(x):
        return f(x, *args)
    return wrapper


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
    def wrapper(*args):
        try:
            ret = func.cache_info.cache[args]
            func.cache_info.hits += 1
        except KeyError:
            ret = func(*args)
            func.cache_info.cache[args] = ret
            func.cache_info.misses += 1
        func.cache_info.all_calls.append([args, ret])
        return ret
    return wrapper


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
    def wrapper(x):
        return tuple_wrapper(tuple(x))

    wrapper.cache_info = tuple_wrapper.cache_info
    return wrapper


def nan_fval_to_inf(f):
    """
    Wrappes function to return infinity instead nan.
    """
    @wraps(f)
    def wrapper(x):
        y = f(x)
        if y is None or np.isnan(y):
            return np.inf
        return y
    return wrapper


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
    if eval_file:
        with open(eval_file, 'w') as fl:
            print('Time of evaluation start', 'Function value',
                  'Parameters values', 'Evaluation time', file=fl, sep='\t')
    @wraps(f)
    def wrapper(x):
        time_start = time.time()
        y = f(x)
        time_end = time.time()
        if eval_file is not None:
            with open(eval_file, 'a') as fl:
                print(time_start - time_init, y, x, time_end - time_start,
                      file=fl, sep='\t')
        return y
    return wrapper


def parallel_wrap(f, x):
    """
    Partial for first argument and result function could be used
    in multiprocessing.
    """
    return f(*x)


class WeightedMetaArray(np.ndarray):
    """Array with metadata."""
    def __new__(cls, array, dtype=None, order=None):
        obj = np.asarray(np.array(array, dtype=object), dtype=dtype, order=order).view(cls)                                 
        obj.metadata = ''
        obj.weights = np.ones(obj.shape)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', [{}]*(obj.ndim+1))
        self.weights = getattr(obj, 'weights', [{}]*(obj.ndim+1))

    def __str__(self):
        return super(WeightedMetaArray, self).__str__() + '\t' + self.metadata

    def __repr__(self):
        return super(WeightedMetaArray, self).__str__() + '\t' + self.metadata


def update_by_one_fifth_rule(value, const, was_improved):
    if was_improved:
        return value * const
    return value / (const) ** (0.25)


def check_file_existence(path_to_file):
    return os.path.exsists(path_to_file) and os.path.isfile(path_to_file)


def ensure_file_existence(path_to_file):
    if check_file_existence(path_to_file):
        return True
    path_to_file.open('w').close()
    return True

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
        with open(self.log_filename, 'a') as fl:
            fl.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    
