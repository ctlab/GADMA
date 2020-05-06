from ..utils import Variable, ContinuousVariable
import copy
import numpy as np
from functools import lru_cache


class Optimizer(object):
    """
    Base class for optimizer.

    :param f: function for minimization.
    :param variables: variables
    """
    def __init__(self, log_transform=False, maximize=False):
        self.log_transform = log_transform
        if self.log_transform:
            self.transform = np.log
            self.inv_transform = np.exp
        else:
            self.transform = lambda x: x
            self.inv_transform = self.transform

        self.maximize = maximize

    def evaluate(self, f, x, args=()):
        y = f(self.inv_transform(x), *args)
        if self.maximize:
            y = -y
        return y

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
