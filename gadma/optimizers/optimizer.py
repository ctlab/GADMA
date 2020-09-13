import copy
import numpy as np
from functools import wraps
import sys

from ..utils import Variable, ContinuousVariable
from ..utils import fix_args, cache_func, nan_fval_to_inf, WeightedMetaArray
from ..utils import ensure_file_existence, variables_values_repr
from ..utils import logarithm_transform, exponent_transform, ident_transform


class Optimizer(object):
    """
    Base class for optimizer. The most important methods:
    :meth:`gadma.optimizers.Optimizer.evaluate` and
    :meth:`gadma.optimizers.Optimizer.optimize`

    :param log_transform: If True then all parameters are optimized in log
                          scale.
    :type log_transform: bool
    :param maximize: If True then maximization of target function is
                     performed.
    :type maximize: bool
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
        """
        Returns -1 if maximization and 1 if minimization of target function.
        """
        return -1 if self.maximize else 1

    def prepare_callback(self, callback):
        """
        Wraps callback function for usage inside optimizer. It transforms x
        according to `log_transform` and multiply y to -1 if `maximize` is
        True.

        :param callback: function of callback that has the following notation:
                         `callback(x, y)`, where x is value of parameters and
                         y is the value of target function on `x`.
        """
        @wraps(callback)
        def wrapper(x, y):
            x_tr = self.inv_transform(x)
            if isinstance(x, WeightedMetaArray):
                x_tr = WeightedMetaArray(x_tr)
                x_tr.metadata = x.metadata
            return callback(x_tr, self.sign * y)
        return wrapper

    def write_report(self, n_iter, variables, x, y, report_file):
        """
        Writes report of optimizer to file or stdout.

        :param n_iter: Number of iteration of optimization.
        :param variables: list of variables which values are optimized.
        :param x: Values of variables.
        :param y: Value of target function on `x`.
        :param report_file: filename to write report. If None then report is
                            printed to stdout.
        """
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
        """
        Wraps function `f` for automatically report output. When function is
        called report is saved to `report_file` every `verbose` call.

        :param f: Function to wrap.
        :param variables: Variables of function `f`.
        :param verbose: Verbosity level.
        :param report_file: Filename to save report.
        """
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

    def evaluate(self, f, x, args=(), linear_constrain=None):
        """
        Evaluates function `f` on values `x`.

        :param f: Target function.
        :param x: Value of parameters of `f`.
        :param args: Other arguments of f. `f(x, args)`.
        :param linear_constrain: Linear constrain on `x`.
        """
        x_tr = self.inv_transform(x)
        if linear_constrain is not None:
            if not linear_constrain.fits(x_tr):
                x_tr, success = linear_constrain.try_to_transform(x_tr)
                if not success:
                    warnings.warn(f"HERE IS A LITTLE PROBLEM. PLEASE CHECK "
                                  f"IT: {x}, {x_tr}")
                    return self.sign * np.inf
        y = f(x_tr, *args)
        return self.sign * y

    def prepare_f_for_opt(self, f, args=(), cache=True):
        """
        Prepares `f` for usage in optimizer. It should be transformed
        according to `log_transform` and `maximize`. Arguments are fixed and
        it could be cached or not.

        :param f: Target function to work with.
        :param args: Arguments of the function. `f(x, args)`.
        :type args: tuple
        :param cache: If True then function is cached.
        :type cache: bool.
        """
        assert isinstance(cache, bool)
        # Fix args
        f_wrapped = fix_args(f, *args)
        f_wrapped = nan_fval_to_inf(f_wrapped)
        if cache:
            f_wrapped = cache_func(f_wrapped)
        return f_wrapped

    def check_variables(self, variables):
        """
        Checks that all `variables` are instances of
        :class:`gadma.utils.Variable` class.
        """
        for var in variables:
            assert isinstance(var, Variable)

    def optimize(f, variables, args=(), options={}, linear_constrain=None,
                 maxiter=None):
        """
        Run optimization for target function.

        :param f: Target function to optimize.
        :type f: func
        :param variables: List of variables which values are optimized.
        :type variables: :class:`gadma.utils.VariablePool`
        :param args: Additional arguments of target function.
        :type args: tuple
        :param options: Additional options kwargs for optimization.
        :type options: dict
        :param linear_constrain: Linear constrain on optimized variables.
        :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
        :param maxiter: Maximum number of iterations to run. If None then run
                        until converge.
        :type maxiter: int
        """
        raise NotImplementedError


class ContinuousOptimizer(Optimizer):
    """
    Base class for optimization of continous variables.
    """
    def check_variables(self, variables):
        """
        Returns True if all variables are instances of
        :class:`gadma.utils.ContinousVariable` class.
        """
        for var in variables:
            assert isinstance(var, ContinuousVariable)
        super(ContinuousOptimizer, self).check_variables(variables)


class UnconstrainedOptimizer(ContinuousOptimizer):
    """
    Base class for unconstrained optimization, i.e. when values of variables
    have no bounds.
    """
    def check_variables(self, variables):
        """
        Returns True if all variables have domain of [-inf, inf].
        """
        super(UnconstrainedOptimizer, self).check_variables(variables)
        for var in variables:
            assert np.allclose(var.domain, np.array([-np.inf, np.inf]))


class ConstrainedOptimizer(ContinuousOptimizer):
    """
    Base class for constrained optimization, i.e. when values of variables
    have some bounds.
    """
    def check_variables(self, variables):
        """
        Returns True if all variables have constrained domain.
        """
        super(ConstrainedOptimizer, self).check_variables(variables)
        for var in variables:
            assert np.all(var.domain != np.array([-np.inf, np.inf]))
