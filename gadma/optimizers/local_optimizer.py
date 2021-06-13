from . import Optimizer, ConstrainedOptimizer, UnconstrainedOptimizer
from . import ContinuousOptimizer
from .optimizer_result import OptimizerResult
from ..utils import ContinuousVariable, apply_transform

import warnings
import copy
import numpy as np
import scipy
from functools import wraps

_registered_local_optimizers = {}


class LocalOptimizer(Optimizer):
    """
    Base class for local optimization.
    See :class:`gadma.optimizers.Optimizer` for more information.
    """
    def process_optimize_kwargs(self, f, variables, x0, options):
        """
        Returns kwargs to run :meth:`_optimize` method. Transforms `x0`
        according to `log_transform`.

        :param f: Objective function.
        :param variables: Variables of `f`.
        :param x0: Initial point of local optimization.
        :param options: Some additional options of run.
        """
        # restore
        if self.run_info.result.x is not None:
            x0 = self.run_info.result.x
        return {"x0": apply_transform(variables, self.transform, x0),
                "options": options}

    def _optimize(self, f, variables, x0, options,
                  maxiter, maxeval, iter_callback):
        raise NotImplementedError

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        assert len(run_info.result.X_out) > 0
        y_best_iter = min(run_info.result.Y_out)
        idx = run_info.result.Y_out.index(y_best_iter)
        x_best_iter = run_info.result.X_out[idx]
        string = Optimizer._n_iter_string(n_iter=run_info.result.n_iter,
                                          variables=variables,
                                          x=x_best_iter,
                                          y=y_best_iter)
        print(string, file=stream)

    def optimize(self, f, variables, x0, args=(), options={},
                 linear_constrain=None, maxiter=None,
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
        optimize_kwargs = {"x0": x0, "options": options}
        return super(LocalOptimizer, self).optimize(
            f=f,
            variables=variables,
            args=args,
            linear_constrain=linear_constrain,
            maxiter=maxiter,
            maxeval=maxeval,
            verbose=verbose,
            callback=callback,
            report_file=report_file,
            eval_file=eval_file,
            save_file=save_file,
            restore_file=restore_file,
            restore_points_only=restore_points_only,
            restore_x_transform=restore_x_transform,
            **optimize_kwargs
        )


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
    def _optimize(self, f, variables, x0, options,
                  maxiter, maxeval, iter_callback):
        y = f(x0)
        iter_callback(x0, y, [x0], [y])
        self.run_info.result.success = True
        self.run_info.result.status = 1
        self.run_info.result.message = "SUCCESS"
        return self.run_info.result


register_local_optimizer(None, NoneOptimizer())
register_local_optimizer("None", _registered_local_optimizers[None])


class ScipyOptimizer(LocalOptimizer, ContinuousOptimizer):
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

    def check_variables(self, variables):
        """
        Checks that all `variables` are instances of
        :class:`gadma.utils.ContinuousVariable` class.
        """
        for var in variables:
            assert isinstance(var, ContinuousVariable)

    def _get_scipy_callback(self, f, iter_callback):
        """
        Returns callback function that calls iter_callback but has notation
        for scipy methods.
        """
        @wraps(iter_callback)
        def wrapper_for_scipy(xk, result_obj=None):
            yk = f(xk)
            # Correct X_iter and Y_iter will be updated by each call of f
            iter_callback(xk, yk, X_iter=[xk], Y_iter=[yk])
        return wrapper_for_scipy

    def get_addit_scipy_kwargs(self, variables):
        raise NotImplementedError

    def _optimize(self, f, variables, x0, options,
                  maxiter, maxeval, iter_callback):
        """
        Run Scipy optimization.
        """
        x0 = np.array(x0, dtype=np.float)
        y = f(x0)
        iter_callback(x0, y, [x0], [y])

        if maxiter == 0 or maxeval == 0:
            self.run_info.result.success = True
            self.run_info.result.status = 0
            self.run_info.result.message = "maxiter or maxeval is 0"

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
        if 'callback' in options:
            raise ValueError("You have set callback in options. Please "
                             "specify callback via optimize method.")
        callback = self._get_scipy_callback(f, iter_callback)

        # We want to wrap f in order to update result.X and result.Y
        def f_in_scipy(x):
            y = f(x)
            iter_callback(x, y, [x], [y])
            # we need to fix n_iter as it is not an iteration but evaluation
            self.run_info.result.n_iter -= 1
            return y

        # Good values from dadi
        if (self.method != "Nelder-Mead" and self.method != "Powell" and
                "eps" not in options):
            options["eps"] = 1e-3

        # Run optimization of SciPy
        addit_kw = self.get_addit_scipy_kwargs(variables)
        res_obj = scipy.optimize.minimize(f_in_scipy, x0, args=(),
                                          method=self.method, options=options,
                                          callback=callback, **addit_kw)
        # Call callback after the last iteration
        callback(res_obj.x)
        # Construct OptimizerResult object to return
        result = OptimizerResult.from_SciPy_OptimizeResult(res_obj)
        assert self.sign * self.run_info.result.y <= result.y
        self.run_info.result.success = result.success
        self.run_info.result.message = result.message
        self.run_info.result.status = result.status

        return self.run_info.result


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
        return {'bounds': [var.domain for var in variables]}


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
        self.out_of_bounds = 1e8  # np.inf

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

    def evaluate(self, f, variables, x, args=(), linear_constrain=None):
        if np.any([not var.correct_value(el)
                   for el, var in zip(x, variables)]):
            return self.sign * self.out_of_bounds
        # we multiply by sign to avoid double * by -1 and for correct optim.
        return self.sign * super(ManuallyConstrOptimizer, self).evaluate(
            f=f,
            variables=variables,
            x=x, args=args,
            linear_constrain=linear_constrain
        )

    def _optimize(self, f, variables, x0, options,
                  maxiter, maxeval, iter_callback):
        vars_in_opt = copy.deepcopy(variables)
        if isinstance(self.optimizer, UnconstrainedOptimizer):
            for i in range(len(vars_in_opt)):
                vars_in_opt[i].domain = [-np.inf, np.inf]

        def callback(x, y):
            if np.any([not var.correct_value(el)
                       for el, var in zip(x, variables)]):
                return
            # y is translated back by *(-1) we want correct comparison
            y = self.optimizer.sign * y
            iter_callback(x, y, [x], [y])
            opt_run_info = self.optimizer.run_info.result.X
            self.run_info.result.X = [apply_transform(variables,
                                                      self.inv_transform,
                                                      x)
                                      for x in opt_run_info]
            self.run_info.result.Y = self.optimizer.run_info.result.Y
            self.run_info.result.n_iter = self.optimizer.run_info.result.n_iter

        # Dadi and moments use eps equal to 1e-3. It turned out to be good
        # value. So we want to use it in our optimizers.
        if "eps" not in options:
            options["eps"] = 1e-3

        self.optimizer.optimize(f=f,
                                variables=vars_in_opt,
                                x0=x0,
                                args=(),
                                options=options,
                                maxiter=maxiter,
                                maxeval=maxeval,
                                callback=callback)
        return self.run_info.result


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
