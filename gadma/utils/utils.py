import functools


def extract_args(func):
    def wrapper(args):
        return func(*args)
    return wrapper
