from .optimizer import Optimizer, ConstrainedOptimizer, UnconstrainedOptimizer
from .optimizer_result import OptimizerResult

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


class ScipyOptimizer(LocalOptimizer):
    scipy_methods = []
    opt_type = ''
    def __init__(self, method, log_transform=False, maximize=False):
        if method not in self.scipy_methods:
            raise ValueError(f"There is no such {self.opt_type} method "
                             f"{method} in scipy.minimize. Available methods"
                             f" are: {self.scipy_methods}")
        self.method = method
        super(ScipyOptimizer, self).__init__(log_transform, maximize)

    def prepare_callback(self, f, callback):
        @wraps(callback)
        def wrapper(xk, result_obj=None):
            return callback(xk, f(xk))
        return wrapper

    def get_addit_scipy_kwargs(self, variables):
        raise NotImplementedError

    def optimize(self, f, variables, x0, args=(), options={}, maxiter=None,
                 callback=None, eval_file=None):
        x0_in_opt = self.transform(x0)
        self.check_variables(variables)
        if maxiter is not None:
            options['maxiter'] = int(maxiter)
        # Fix args in function f and cache it.
        # TODO: not intuitive solution, think more about it.
        cached_f = self.prepare_f_for_opt(f, args, eval_file)
        f_in_opt = partial(self.evaluate, cached_f)

        # Create callback for scipy
        if callback is not None:
            if 'callback' in options:
                raise ValueError("You have set two callbacks - first in "
                                 "options and second via callback. Please "
                                 "specify one callback.")
            options['callback'] = self.prepare_callback(f_in_opt, callback)

        # Run optimization of SciPy
        addit_kw = self.get_addit_scipy_kwargs(variables)
        res_obj = scipy.optimize.minimize(f_in_opt, x0_in_opt, args=(),
                                          method=self.method, options=options,
                                          **addit_kw)

        # Construct OptimizerResult object to return
        result = OptimizerResult.from_SciPy_OptimizeResult(res_obj)
        result.x = self.inv_transform(result.x)
        
        result.X = [np.array(_x) for _x, _y in cached_f.cache_info.all_calls]
        result.Y = [_y for _x, _y in cached_f.cache_info.all_calls]
        result.n_eval = cached_f.cache_info.misses

        assert res_obj.nfev == cached_f.cache_info.misses + cached_f.cache_info.hits
        assert result.y == f_in_opt(res_obj.x)
        
        return result

class ScipyUnconstrOptimizer(ScipyOptimizer, UnconstrainedOptimizer):
    scipy_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                     'COBYLA', 'trust-constr', 'dogleg', 'trust-ncg',
                     'trust-exact', 'trust-krylov']
    opt_type = 'unconstrained'

    def get_addit_scipy_kwargs(self, variables):
        return {}

class ScipyConstrOptimizer(ScipyOptimizer, ConstrainedOptimizer):
    scipy_methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    opt_type = 'constrained'

    def get_addit_scipy_kwargs(self, variables):
        return {'bounds': [self.transform(var.domain) for var in variables]}


register_local_optimizer('L-BFGS-B', ScipyConstrOptimizer('L-BFGS-B'))
register_local_optimizer('L-BFGS-B_log',
                         ScipyConstrOptimizer('L-BFGS-B',
                                              log_transform=True))


class ManuallyConstrOptimizer(LocalOptimizer, ConstrainedOptimizer):
    def __init__(self, optimizer, log_transform=False):
        self.optimizer = optimizer
        super(ManuallyConstrOptimizer, self).__init__(log_transform,
                                                      self.optimizer.maximize)
        self.out_of_bounds = np.inf

    def evaluate_inner(self, f, x, bounds, args=()):
        for val, domain in zip(x, bounds):
            if val < domain[0] or val > domain[1]:
                return self.out_of_bounds
        return f(self.inv_transform(x), *args)

    def optimize(self, f, variables, x0, args=(), options={}, maxiter=None,
                 callback=None, eval_file=None):
        x0_in_opt = self.optimizer.transform(self.transform(x0))
        bounds = self.transform([var.domain for var in variables])
        vars_in_opt = copy.deepcopy(variables)
        if isinstance(self.optimizer, UnconstrainedOptimizer):
            for var in vars_in_opt:
                var.domain = np.array([-np.inf, np.inf])
        args_in_opt = (bounds, args)
        # Fix args in function f and cache it.
        # TODO: not intuitive solution, think more about it.
        cached_f = self.prepare_f_for_opt(f, args, eval_file)
        f_in_opt = partial(self.evaluate_inner, cached_f)

        result = self.optimizer.optimize(f_in_opt, vars_in_opt, x0_in_opt,
                                      args=(bounds,), options=options,
                                      maxiter=maxiter)
        return result

register_local_optimizer('BFGS',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('BFGS')))
register_local_optimizer('BFGS_log',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('BFGS'),
                            log_transform=True))
register_local_optimizer('Powell',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Powell')))
register_local_optimizer('Powell_log',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Powell'),
                            log_transform=True))
register_local_optimizer('Nelder-Mead',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Nelder-Mead')))
register_local_optimizer('Nelder-Mead_log',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Nelder-Mead'),
                            log_transform=True))
