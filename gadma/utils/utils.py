from functools import lru_cache
import time
import numpy as np

def extract_args(func):
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
    def wrapper(x):
        return f(x, *args)
    return wrapper


def cache_func(f):
    """
    Cashes function with one argument.
    :param f: function such that f(x).
    :returns: function that is cashed.
    """
    @lru_cache(maxsize=None)
    def tuple_wrapper(x_tuple):
        return f(np.array(x_tuple))
    def wrapper(x):
        return tuple_wrapper(tuple(x))
    wrapper.cache_info = tuple_wrapper.cache_info
    return wrapper


def eval_wrapper(f, args=(), eval_file=None, cache=True):
    """
    Returns good function for optimization. Each evaluation of function will
    be written in file. If needed function will be cached.
    :param f: function. Is called as f(x, *args).
    :param args: tuple of arguments.
    :param eval_file: file to write evaluations.
    :param cache: if True then function will be cached.
    """
    time_init = time.time()
    if eval_file is not None:
        with open(eval_file, 'w') as fl:
            print('Time of evaluation start', 'Function value',
                  'Parameters values', 'Evaluation time', file=fl, sep='\t')
    if len(args) > 0:
        func = fix_args(f, args)
    else:
        func = f
    if cache:
        func = cache_func(func)
    def wrapper(x):
        time_start = time.time()
        y = func(x)
        time_end = time.time()
        if eval_file is not None:
            with open(eval_file, 'a') as fl:
                print(time_start - time_init, y, x, time_end - time_start,
                      file=fl, sep='\t')
        return y
    if cache:
        wrapper.cache_info = func.cache_info
    return wrapper
