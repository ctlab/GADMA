from ..utils import Variable, ContinuousVariable
from ..utils import fix_args, cache_func, nan_fval_to_inf
from ..utils import ensure_file_existence, variables_values_repr
from ..utils import logarithm_transform, exponent_transform, ident_transform
import copy
import numpy as np
from functools import wraps
import sys

class Optimizer(object):
    """
    Base class for optimizer.

    :param f: function for minimization.
    :param variables: variables
    """
    def __init__(self, log_transform=False, maximize=False):
        self.log_transform = log_transform
        if self.log_transform:
            self.transform = logarithm_transform
            self.inv_transform = exponent_transform
        else:
            self.transform = ident_transform
            self.inv_transform = ident_transform

        self.maximize = maximize

    @property
    def sign(self):
        return -1 if self.maximize else 1

    def prepare_callback(self, callback):
        @wraps(callback)
        def wrapper(x, y):
            return callback(self.inv_transform(x), self.sign * y)
        return wrapper

    def write_report(self, n_iter, variables, x, y, report_file):
        if report_file:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        x_repr = variables_values_repr(variables, x)
        metainfo = ''
        if hasattr(x, 'metadata'):
            metainfo = x.metadata
        print(n_iter, y, x_repr, metainfo, sep='\t', file=stream)
        if report_file:
            stream.close()

    def wrap_for_report(self, f, variables, verbose, report_file):
        @wraps(f)
        def wrapper(x):
            wrapper._counter += 1
            y = f(x)
            if (verbose > 0) and (wrapper._counter % verbose == 0):
                self.write_report(wrapper._counter, variables, x, y,
                                  report_file)
            return y
        if report_file:
            ensure_file_existence(report_file)
        wrapper._counter = 0
        if verbose == 0:
            return f
        return wrapper

    def evaluate(self, f, x, args=()):
        y = f(self.inv_transform(x), *args)
        return self.sign * y

    def prepare_f_for_opt(self, f, args=(), cache=True):
        assert isinstance(cache, bool)
        # Fix args
        f_wrapped = fix_args(f, *args)
        f_wrapped = nan_fval_to_inf(f_wrapped)
        if cache:
            f_wrapped = cache_func(f_wrapped)
        return f_wrapped

    def check_variables(self, variables):
        for var in variables:
            assert isinstance(var, Variable)

    def optimize(f, variables, args=(), options={}, maxiter=None):
        raise NotImplementedError


class ContinuousOptimizer(Optimizer):
    def check_variables(self, variables):
        for var in variables:
            assert isinstance(var, ContinuousVariable)
        super(ContinuousOptimizer, self).check_variables(variables)


class UnconstrainedOptimizer(ContinuousOptimizer):
    def check_variables(self, variables):
        super(UnconstrainedOptimizer, self).check_variables(variables)
        for var in variables:
            assert np.allclose(var.domain, np.array([-np.inf, np.inf]))


class ConstrainedOptimizer(ContinuousOptimizer):
    def check_variables(self, variables):
        super(ConstrainedOptimizer, self).check_variables(variables)
        for var in variables:
            assert np.all(var.domain != np.array([-np.inf, np.inf]))
