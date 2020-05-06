from .optimizer import Optimizer, ConstrainedOptimizer, UnconstrainedOptimizer

import copy
import numpy as np
import scipy
from functools import partial

_registered_local_optimizers = {}

class LocalOptimizer(Optimizer):
    def optimize(self, f, variables, x0, args=(), options={}, maxiter=None):
        raise NotImplementedError

def register_local_optimizer(id, optimizer):
    """
    Registers the specified local optimizer.
    """
    if id in _registered_local_optimizers:
        raise ValueError(f"Optimizer '{id}' is already registered.")
    if not isinstance(optimizer, LocalOptimizer):
        raise ValueError("Optimizer is not local.")
    _registered_local_optimizers[id] = optimizer
    optimizer.id = id


def get_local_optimizer(id):
    """
    Returns the local optimizer with the specified id.
    """
    if id not in _registered_local_optimizers:
        raise ValueError(f"Optimizer '{id}' is not registered")
    return _registered_local_optimizers[id]


def all_local_optimizers():
    """
    Returns an iterator over all registered local optimizers.
    """
    for optim in _registered_local_optimizers.values():
        yield optim


class ScipyUnconstrOptimizer(LocalOptimizer, UnconstrainedOptimizer):
    def __init__(self, method, log_transform=False, maximize=False):
        if method not in ['BFGS', 'Powell']:
            raise ValueError(f"There is no such unconstrained method {method} im scipy.minimize.")
        self.method = method
        super(ScipyUnconstrOptimizer, self).__init__(log_transform, maximize)

    def optimize(self, f, variables, x0, args=(), options={}, maxiter=None):
        x0_in_opt = self.transform(x0)
        self.check_variables(variables)
        if maxiter is not None:
            options['maxiter'] = int(maxiter)
        f_in_opt = partial(self.evaluate, f)
        res_obj = scipy.optimize.minimize(f_in_opt, x0_in_opt, args=(args,),
                                          method=self.method, options=options)
        ret_x = self.inv_transform(res_obj.x)
        return ret_x, f_in_opt(res_obj.x, args)


class ScipyConstrOptimizer(LocalOptimizer, ConstrainedOptimizer):
    def __init__(self, method, log_transform=False, maximize=False):
        if method not in ['L-BFGS-B']:
            raise ValueError(f"There is no such constrained method {method} in scipy.minimize.")
        self.method = method
        super(ScipyConstrOptimizer, self).__init__(log_transform, maximize)

    def optimize(self, f, variables, x0, args=(), options={}, maxiter=None):
        x0_in_opt = self.transform(x0)
        self.check_variables(variables)
        bounds = [self.transform(var.domain) for var in variables]
        if maxiter is not None:
            options['maxfun'] = int(maxiter / 2)
        f_in_opt = partial(self.evaluate, f)
        res_obj = scipy.optimize.minimize(f_in_opt, x0_in_opt, bounds=bounds,
                                          args=(args,), method=self.method,
                                          options=options)
        ret_x = self.inv_transform(res_obj.x)
        return ret_x, f_in_opt(res_obj.x, args)


register_local_optimizer('L-BFGS-B', ScipyConstrOptimizer('L-BFGS-B'))
register_local_optimizer('L-BFGS-B_log',
                         ScipyConstrOptimizer('L-BFGS-B',
                                              log_transform=True))

class ManuallyConstrOptimizer(LocalOptimizer, ConstrainedOptimizer):
    def __init__(self, optimizer, log_transform=False):
        self.optimizer = optimizer
        super(ManuallyConstrOptimizer, self).__init__(log_transform, self.optimizer.maximize)
        self.out_of_bounds = np.inf

    def evaluate_inner(self, f, x, bounds, args=()):
        for val, domain in zip(x, bounds):
            if val < domain[0] or val > domain[1]:
                return self.out_of_bounds
        return f(self.inv_transform(x), *args)

    def optimize(self, f, variables, x0, args=(), options={}, maxiter=None):
        x0_in_opt = self.optimizer.transform(self.transform(x0))
        bounds = self.transform([var.domain for var in variables])
        vars_in_opt = copy.deepcopy(variables)
        if isinstance(self.optimizer, UnconstrainedOptimizer):
            for var in vars_in_opt:
                var.domain = np.array([-np.inf, np.inf])
        args_in_opt = (bounds, args)
        f_in_opt = partial(self.evaluate_inner, f)
        res = self.optimizer.optimize(f_in_opt, vars_in_opt, x0_in_opt,
                                       args=args_in_opt, options=options,
                                       maxiter=maxiter)
        return self.inv_transform(res[0]), res[1]


register_local_optimizer('BFGS',
                         ManuallyConstrOptimizer(ScipyUnconstrOptimizer('BFGS')))
register_local_optimizer('BFGS_log',
                         ManuallyConstrOptimizer(ScipyUnconstrOptimizer('BFGS'),
                         log_transform=True))
register_local_optimizer('Powell',
                         ManuallyConstrOptimizer(ScipyUnconstrOptimizer('Powell')))
register_local_optimizer('Powell_log',
                         ManuallyConstrOptimizer(ScipyUnconstrOptimizer('Powell'),
                         log_transform=True))
