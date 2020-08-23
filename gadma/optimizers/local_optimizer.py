from .optimizer import Optimizer, ConstrainedOptimizer, UnconstrainedOptimizer
from .optimizer_result import OptimizerResult
from ..utils import eval_wrapper, rpartial, fix_args

import warnings
import copy
import numpy as np
import scipy
from functools import partial, wraps

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
    return copy.deepcopy(_registered_local_optimizers[id])


def all_local_optimizers():
    """
    Returns an iterator over all registered local optimizers.
    """
    for optim in _registered_local_optimizers.values():
        yield copy.deepcopy(optim)


class NoneOptimizer(LocalOptimizer):
    def optimize(self, f, variables, x0, args=(), options={},
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, eval_file=None, report_file=None,
                 save_file=None):
        y = f(x0, *args)
        result = OptimizerResult(x0, y, True, 1, message="SUCCESS",
                                 X=[x0], Y=[y], n_eval=1, n_iter=0)
        return result


register_local_optimizer("None", NoneOptimizer())
register_local_optimizer(None, NoneOptimizer())

class ScipyOptimizer(LocalOptimizer):
    scipy_methods = []
    maxeval_kwarg = {}
    opt_type = ''
    def __init__(self, method, log_transform=False, maximize=False):
        if method not in self.scipy_methods:
            raise ValueError(f"There is no such {self.opt_type} method "
                             f"{method} in scipy.minimize. Available methods"
                             f" are: {self.scipy_methods}")
        self.method = method
        super(ScipyOptimizer, self).__init__(log_transform, maximize)            

    def save(self, x_best, y_best, is_finished, save_file):
        if save_file is None:
            return
        with open(save_file, 'wb') as fl:
            pickle.dump((x_best, y_best, is_finished), fl)

    def load(self, save_file):
        with open(save_file, 'rb') as fl:
            x_best, y_best, is_finished = pickle.load(fl)
        return x_best, y_best, is_finished

    def prepare_callback(self, f, callback, save_file=None):
        new_callback = super(ScipyOptimizer, self).prepare_callback(callback)
        @wraps(new_callback)
        def wrapper(xk, result_obj=None):
            yk = f(xk)
            r =  new_callback(xk, yk)
            if wrapper.x_best is None or wrapper.y_best > yk:
                wrapper.x_best = xk
                wrapper.y_best = yk
            self.save(xk, yk, False, save_file)
            return r
        wrapper.x_best = None
        return wrapper

    def get_addit_scipy_kwargs(self, variables):
        raise NotImplementedError

    def optimize(self, f, variables, x0, args=(), options={},
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, eval_file=None, report_file=None,
                 save_file=None):
        # Create logging files
        if eval_file is not None:
            ensure_file_existence(eval_file)
        if report_file is not None:
            ensure_file_existence(report_file)
        if save_file is not None:
            ensure_file_existence(save_file)

        x0 = np.array(x0, dtype=np.float)
        x0_in_opt = self.transform(x0)
        self.check_variables(variables)
        if maxiter:
            options['maxiter'] = int(maxiter)
        if maxeval:
            if self.method not in self.maxeval_kwarg:
                warnings.warn(f"Local optimization {self.method} do not have"
                               "  an option of max number of evaluations. It "
                               "will be used for maxiter.")
                if maxiter:
                    options['maxiter'] = min(maxeval, maxiter)
                else:
                    options['maxiter'] = maxeval
            else:
                kwarg = self.maxeval_kwarg[self.method]
                options[kwarg] = maxeval
                if kwarg == 'maxiter' and maxiter and maxiter != maxeval:
                    warnings.warn(f"Number of iterations is equal to the "
                                  f"number of function evaluations for "
                                  f"{self.method} optimizer.")
                    options[kwarg] = min(maxeval, maxiter)

        # Fix args in function f and cache it.
        # TODO: not intuitive solution, think more about it.
        # Fix args and cache
        prepared_f = self.prepare_f_for_opt(f, args)
        # Wrap for automatic evaluation log
        wrapped_f = eval_wrapper(prepared_f, eval_file)
        # Wrap for writing report
        finally_wrapped_f = self.wrap_for_report(wrapped_f, variables,
                                                 verbose, report_file)

        f_in_opt = partial(self.evaluate, finally_wrapped_f)
        f_in_opt = fix_args(f_in_opt, (), linear_constrain)

        # Create callback for scipy
        if callback is not None:
            if 'callback' in options:
                raise ValueError("You have set two callbacks - first in "
                                 "options and second via callback. Please "
                                 "specify one callback.")
            callback = self.prepare_callback(f_in_opt, callback)

        # IMPORTANT we cannot run scipy optimization for search of 0 params
        if len(variables) == 0:
            x = []
            y = self.sign * f_in_opt(x)
            callback(x, y)
            success = True
            status = 0
            message = "Zero parameters to optimize."
            result = OptimizerResult(x, y, success, status, message, [x], [y], 1, 1)
            return result

        # Run optimization of SciPy
        addit_kw = self.get_addit_scipy_kwargs(variables)
        res_obj = scipy.optimize.minimize(f_in_opt, x0_in_opt, args=(),
                                          method=self.method, options=options,
                                          callback=callback, **addit_kw)
        # Construct OptimizerResult object to return
        result = OptimizerResult.from_SciPy_OptimizeResult(res_obj)
        result.x = self.inv_transform(result.x)
        result.y = self.sign * result.y
        
        result.X = [np.array(_x) for _x, _ in prepared_f.cache_info.all_calls]
        result.Y = [self.sign * _y for _, _y in prepared_f.cache_info.all_calls]

        result.n_eval = prepared_f.cache_info.misses

        self.save(result.x, result.y, True, save_file)
        
        return result

class ScipyUnconstrOptimizer(ScipyOptimizer, UnconstrainedOptimizer):
    scipy_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                     'COBYLA', 'trust-constr', 'dogleg', 'trust-ncg',
                     'trust-exact', 'trust-krylov']
    maxeval_kwarg = {'Nelder-Mead': 'maxfun', 'Powell': 'maxfev',
                     'COBYLA': 'maxiter'}

    opt_type = 'unconstrained'

    def get_addit_scipy_kwargs(self, variables):
        return {}

class ScipyConstrOptimizer(ScipyOptimizer, ConstrainedOptimizer):
    scipy_methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    maxeval_kwarg = {'L-BFGS-B': 'maxfun'}
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

    @property
    def maximize(self):
        return self.optimizer.maximize

    @maximize.setter
    def maximize(self, new_value):
        self.optimizer.maximize = new_value

    def evaluate_inner(self, f, x, bounds, args=()):
        for val, domain in zip(x, bounds):
            if val < domain[0] or val > domain[1]:
                return self.sign * self.out_of_bounds
        y = f(self.inv_transform(x), *args)
        return y

    def prepare_callback(self, callback):
        @wraps(callback)
        def wrapper(x, y):
            return callback(self.inv_transform(x), y)
        return wrapper

    def optimize(self, f, variables, x0, args=(), options={},
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, eval_file=None, report_file=None):
        x0 = np.array(x0, dtype=np.float)
        x0_in_opt = self.optimizer.transform(self.transform(x0))
        bounds = self.transform([var.domain for var in variables])
        vars_in_opt = copy.deepcopy(variables)
        if isinstance(self.optimizer, UnconstrainedOptimizer):
            for var in vars_in_opt:
                var.domain = np.array([-np.inf, np.inf])
        args_in_opt = (bounds, args)
        # Fix args in function f and cache it.
        # TODO: not intuitive solution, think more about it.
        # Fix args
        prepared_f = self.prepare_f_for_opt(f, args, cache=False)
        # Write automatix evaluation log. Incide optimizer it could different
        # x in function as there can be extra logarithm.
        wrapped_f = eval_wrapper(prepared_f, eval_file)
        # The same with report_file
        finally_wrapped_f = self.wrap_for_report(wrapped_f, variables,
                                                 verbose, report_file)
        f_in_opt = partial(self.evaluate_inner, finally_wrapped_f)
#        f_in_opt = fix_args(f_in_opt, (linear_constrain,))

        if callback is not None:
            callback = self.prepare_callback(callback)
        result = self.optimizer.optimize(f_in_opt, vars_in_opt, x0_in_opt,
                                      args=(bounds,), options=options,
                                      verbose=0,
                                      linear_constrain=linear_constrain,
                                      maxiter=maxiter,
                                      maxeval=maxeval, callback=callback)
        # TODO: need to check result.X as they should be transformed somehow.
        result.x = self.inv_transform(result.x)
        result.X = [self.inv_transform(x) for x in result.X]
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
