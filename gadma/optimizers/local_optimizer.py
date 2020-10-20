from . import Optimizer, ConstrainedOptimizer, UnconstrainedOptimizer
from .optimizer_result import OptimizerResult
from ..utils import eval_wrapper, ensure_file_existence, fix_args, cache_func

import warnings
import copy
import numpy as np
import scipy
from functools import partial, wraps
import pickle

_registered_local_optimizers = {}


class LocalOptimizer(Optimizer):
    """
    Base class for local optimization.
    See :class:`gadma.optimizers.Optimizer` for more information.
    """
    def optimize(self, f, variables, x0, args=(), options={}, maxiter=None,
                 maxeval=None, verbose=0, callback=None,
                 report_file=None, eval_file=None, save_file=None,
                 restore_file=None, restore_points_only=False,
                 restore_x_transform=None):
        """
        Run optimization of local search algorithm.

        :param f: Target function to optimize.
        :type f: func
        :param variables: Variables of `f` which values should be optimized.
        :type variables: :class:`gadma.utils.VariablePool`
        :param x0: Initial point to start optimization.
        :type x0: list
        :param args: Additional arguments of target function.
        :type args: tuple
        :param options: Additional options kwargs for optimization.
        :type options: dict
        :param maxiter: Maximum number of iterations to run.
        :type maxiter: int
        :param maxeval: Maximum number of evaluations to run. If None then run
                        until converge.
        :type maxeval: int
        :param verbose: Verbosity of the output. If 0 then no reports.
        :type verbose: int
        :param callback: Callback to run after each iteration of optimization.
                         Should be called as `callback(x, y)`
        :type callback: function
        :param report_file: File to save report. Check option `verbose`.
        :type report_file: str
        :param eval_file: File to save all evaluations of the function `f`.
        :type eval_file: str
        :param save_file: File to save information during optimization for its
                          reconstruction.
        :type save_file: str
        :param restore_file: File to restore previous run.
        :type restore_file: str
        :param restore_points_only: Restore point/points from previous run and
                                    run optimization from them once more. If
                                    False then previous run will be resumed.
        :type restore_points_only: bool
        :param restore_x_transform: Restore points but transform them before
                                    usage in this run.
        :type restore_x_transform: function
        """
        raise NotImplementedError


def register_local_optimizer(id, optimizer):
    """
    Registers the specified local optimizer.

    :param id: ID of local optimizer to register.
    :param optimizer: Optimizer to register.
    :type optimizer: :class:`gadma.optimizers.LocalOptimizer`
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

    :param id: ID of local optimizer.
    """
    if id not in _registered_local_optimizers:
        raise ValueError(f"Optimizer '{id}' is not registered")
    return copy.deepcopy(_registered_local_optimizers[id])


def all_local_optimizers():
    """
    Returns an iterator over all registered local optimizers.
    """
    for optim in set(_registered_local_optimizers.values()):
        yield copy.deepcopy(optim)


class NoneOptimizer(LocalOptimizer):
    """
    Class that inherits :class:`gadma.optimizers.LocalOptimizer` but do not
    run any optimization.
    """
    def save(self, x0, y, save_file):
        info = (x0, y)
        super(NoneOptimizer, self).save(info, save_file)

    def valid_restore_file(self, save_file):
        try:
            info = self.load(save_file)
        except Exception as e:
            return False
        if not isinstance(info, tuple):
            return False
        if not len(info) == 2:
            return False
        return True

    def optimize(self, f, variables, x0, args=(), options={},
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, eval_file=None, report_file=None,
                 save_file=None, restore_file=None, restore_points_only=False,
                 restore_x_transform=None):
        prepared_f = self.prepare_f_for_opt(f, args, cache=True)
        wrapped_f = eval_wrapper(prepared_f, eval_file)
        finally_wrapped_f = self.wrap_for_report(wrapped_f, variables,
                                                 verbose, report_file)
        if restore_file is not None and self.valid_restore_file(restore_file):
            x0, y = self.load(restore_file)
            if restore_x_transform is not None:
                x0 = restore_x_transform(x)
                y = None
        if restore_file is None or restore_points_only or y is None:
            y = finally_wrapped_f(x0)
        self.save(x0, y, save_file)
        result = OptimizerResult(x0, y, True, 1, message="SUCCESS",
                                 X=[x0], Y=[y], n_eval=1, n_iter=0)
        return result


register_local_optimizer(None, NoneOptimizer())
register_local_optimizer("None", _registered_local_optimizers[None])


class ScipyOptimizer(LocalOptimizer):
    """
    Class of Scipy local search algorithms.

    :cvar ScipyOptimizer.scipy_methods: List of methods names that are
                                        available.
    :cvar ScipyOptimizer.maxeval_kwarg: List of methods names that support
                                        `maxeval` argument.

    :param method: name of method from :func:`scipy.optimize.minimize`.
    :type method: str
    :param log_transform: If True then values rae optimized in log scale.
    :type log_transform: bool
    :param maximize: If True then maximize target function.
    :type maximize: bool
    """
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

    def save(self, x_best, y_best, n_iter, n_eval, is_finished, save_file):
        """
        Save current achievement in optimization. Dumps `x_best`, `y_best` and
        `is_finished` to `save_file` using :mod:`pickle`.

        :param x_best: Values of best configuration.
        :param y_best: Value of target function on `x_best`.
        :param is_finished: If True then optimization was finished.
        :param save_file: Filename to save data.
        """
        info = (x_best, y_best, int(n_iter), n_eval, is_finished)
        super(ScipyOptimizer, self).save(info, save_file)

    def valid_restore_file(self, save_file):
        try:
            info = self.load(save_file)
        except Exception as e:
            return False
        if not isinstance(info, tuple):
            return False
        if not len(info) == 5:
            return False
        if not isinstance(info[2], int):
            return False
        if not isinstance(info[4], bool):
            return False
        return True

    def prepare_callback(self, f, callback, save_file=None):
        new_callback = super(ScipyOptimizer, self).prepare_callback(callback)

        @wraps(new_callback)
        def wrapper(xk, result_obj=None):
            yk = f(xk)
            r = new_callback(xk, yk)
            if wrapper.x_best is None or wrapper.y_best > yk:
                wrapper.x_best = xk
                wrapper.y_best = yk
            n_eval = None
            if hasattr(f, 'cache_info'):
                n_eval = f.cache_info.misses
            self.save(self.inv_transform(wrapper.x_best),
                      self.sign * wrapper.y_best, wrapper.n_iter, n_eval,
                      False, save_file)
            wrapper.n_iter += 1
            return r

        wrapper.x_best = None
        wrapper.n_iter = 0
        return wrapper

    def get_addit_scipy_kwargs(self, variables):
        raise NotImplementedError

    def optimize(self, f, variables, x0, args=(), options={},
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, eval_file=None, report_file=None,
                 save_file=None, restore_file=None, restore_points_only=False,
                 restore_x_transform=None):
        """
        Run Scipy optimization.


        :param f: Target function to optimize.
        :type f: func
        :param variables: Variables of `f` which values should be optimized.
        :type variables: :class:`gadma.utils.VariablePool`
        :param x0: Initial point to start optimization.
        :type x0: list
        :param args: Additional arguments of target function.
        :type args: tuple
        :param options: Additional options kwargs for scipy optimization.
        :type options: dict
        :param maxiter: Maximum number of iterations to run.
        :type maxiter: int
        :param maxeval: Maximum number of evaluations to run. If None then run
                        until converge.
        :type maxeval: int
        :param verbose: Verbosity of the output. If 0 then no reports.
        :type verbose: int
        :param callback: Callback to run after each iteration of optimization.
                         Should be called as `callback(x, y)`
        :type callback: function
        :param report_file: File to save report. Check option `verbose`.
        :type report_file: str
        :param eval_file: File to save all evaluations of the function `f`.
        :type eval_file: str
        :param save_file: File to save information during optimization for its
                          reconstruction.
        :type save_file: str
        :param restore_file: File to restore previous run.
        :type restore_file: str
        :param restore_points_only: Restore point/points from previous run and
                                    run optimization from them once more. If
                                    False then previous run will be resumed.
        :type restore_points_only: bool
        :param restore_x_transform: Restore points but transform them before
                                    usage in this run.
        :type restore_x_transform: function
        """
        self.check_variables(variables)
        # Create logging files
        if eval_file is not None:
            ensure_file_existence(eval_file)
        if report_file is not None:
            ensure_file_existence(report_file)
        if save_file is not None:
            ensure_file_existence(save_file)

        is_finished = False
        if restore_file is not None and self.valid_restore_file(restore_file):
            x0, _y, _n_iter, _n_eval, is_finished = self.load(restore_file)
            if restore_x_transform is not None:
                x0 = restore_x_transform(x0)
            if restore_points_only:
                if maxiter is not None:
                    maxiter -= _n_iter
                if maxeval is not None and _n_eval is not None:
                    maxeval -= _n_eval
                is_finished = False

        x0 = np.array(x0, dtype=np.float)

        # Fix args in function f and cache it.
        # TODO: not intuitive solution, think more about it.
        # Fix args (cache we did at the end)
        prepared_f = self.prepare_f_for_opt(f, args, cache=True)
        # Wrap for automatic evaluation log
        wrapped_f = eval_wrapper(prepared_f, eval_file)
        # Wrap for writing report
        finally_wrapped_f = self.wrap_for_report(wrapped_f, variables,
                                                 verbose, report_file)

        f_in_opt = partial(self.evaluate, finally_wrapped_f)
        f_in_opt = fix_args(f_in_opt, (), linear_constrain)

        # Cache our final version of function
#        f_in_opt = cache_func(f_in_opt)
#        print(f_in_opt.cache_info)

        x0_in_opt = self.transform(x0)

        if is_finished or maxiter == 0 or maxeval == 0:
            success = True
            status = 0
            message = "maxiter or maxeval is 0"
            x = x0
            y = self.sign * f_in_opt(x0_in_opt)
            if callback is not None:
                callback(x, y)
            return OptimizerResult(x, y, success, status, message,
                                   [x], [y], 1, 0, X_out=[], Y_out=[])

        self.check_variables(variables)
        options = copy.copy(options)
        if maxiter is not None:
            options['maxiter'] = int(maxiter)
        if maxeval is not None and maxeval > 0:
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
            result = OptimizerResult(x, y, success, status, message,
                                     [x], [y], 1, 1)
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
        result.Y = [self.sign * _y
                    for _, _y in prepared_f.cache_info.all_calls]

        result.n_eval = prepared_f.cache_info.misses
#        print(str(prepared_f.cache_info))

        self.save(result.x, result.y, result.n_iter, result.n_eval,
                  True, save_file)

        return result


class ScipyUnconstrOptimizer(ScipyOptimizer, UnconstrainedOptimizer):
    """
    Base class for Scipy unconstrained optimizations.
    """
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
register_local_optimizer('optimize_lbfgsb',
                         _registered_local_optimizers['L-BFGS-B'])
register_local_optimizer('L-BFGS-B_log',
                         ScipyConstrOptimizer('L-BFGS-B',
                                              log_transform=True))
register_local_optimizer('optimize_log_lbfgsb',
                         _registered_local_optimizers['L-BFGS-B_log'])


class ManuallyConstrOptimizer(LocalOptimizer, ConstrainedOptimizer):
    """
    Class for Constrained optimization that uses unconstrained optimization.
    The value of target function is considered to be inf outside bounds.

    :param optimizer: Unconstrained optimizer to use.
    :type optimizer: class:`gadma.optimizers.UnconstrainedOptimizer`
    :param log_transform: If True then log scale is used for optimization.
                          Be careful there could be `log_transform` already
                          in `optimizer`.
    """
    def __init__(self, optimizer, log_transform=False):
        self.optimizer = optimizer
        super(ManuallyConstrOptimizer, self).__init__(log_transform,
                                                      self.optimizer.maximize)
        self.out_of_bounds = np.inf

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id
        self.optimizer.id = self._id

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

    def valid_restore_file(self, save_file):
        return self.optimizer.valid_restore_file(save_file)

    def optimize(self, f, variables, x0, args=(), options={},
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, eval_file=None, report_file=None,
                 save_file=None, restore_file=None, restore_points_only=False,
                 restore_x_transform=None):
        self.check_variables(variables)
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
        prepared_f = self.prepare_f_for_opt(f, args, cache=True)
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
        points_only = restore_points_only
        x_transf = restore_x_transform
        result = self.optimizer.optimize(f_in_opt, vars_in_opt, x0_in_opt,
                                         args=(bounds,), options=options,
                                         verbose=0,
                                         linear_constrain=linear_constrain,
                                         maxiter=maxiter,
                                         maxeval=maxeval, callback=callback,
                                         eval_file=None,
                                         report_file=None,
                                         save_file=save_file,
                                         restore_file=restore_file,
                                         restore_points_only=points_only,
                                         restore_x_transform=x_transf)
        # TODO: need to check result.X as they should be transformed somehow.
        result.x = self.inv_transform(result.x)
        result.X = [self.inv_transform(x) for x in result.X]
        result.n_eval = prepared_f.cache_info.misses
        return result


register_local_optimizer('BFGS',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('BFGS')))
register_local_optimizer('optimize', _registered_local_optimizers['BFGS'])
register_local_optimizer('BFGS_log',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('BFGS'),
                            log_transform=True))
register_local_optimizer('optimize_log',
                         _registered_local_optimizers['BFGS_log'])
register_local_optimizer('Powell',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Powell')))
register_local_optimizer('optimize_powell',
                         _registered_local_optimizers['Powell'])
register_local_optimizer('Powell_log',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Powell'),
                            log_transform=True))
register_local_optimizer('optimize_log_powell',
                         _registered_local_optimizers['Powell_log'])
register_local_optimizer('Nelder-Mead',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Nelder-Mead')))
register_local_optimizer('optimize_fmin',
                         _registered_local_optimizers['Nelder-Mead'])
register_local_optimizer('Nelder-Mead_log',
                         ManuallyConstrOptimizer(
                            ScipyUnconstrOptimizer('Nelder-Mead'),
                            log_transform=True))
register_local_optimizer('optimize_log_fmin',
                         _registered_local_optimizers['Nelder-Mead_log'])
