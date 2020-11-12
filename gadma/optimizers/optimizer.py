import copy
import numpy as np
from functools import wraps
import sys
import os

from ..utils import Variable, ContinuousVariable
from ..utils import fix_args, cache_func, WeightedMetaArray
from ..utils import ensure_file_existence, check_file_existence,\
                    variables_values_repr
from ..utils import logarithm_transform, exponent_transform, ident_transform
import pickle
import warnings


class Optimizer(object):
    """
    Base class for optimizer. The most important methods:
    :meth:`gadma.optimizers.Optimizer.evaluate` and
    :meth:`gadma.optimizers.Optimizer.optimize`.

    To create new class for optimizer one should at least implement
    :meth:`gadma.optimizers.Optimizer.valid_restore_file` and
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
        self.maximize = maximize

    @property
    def log_transform(self):
        return self._log_trasform

    @log_transform.setter
    def log_transform(self, log_transform):
        self._log_trasform = log_transform
        if log_transform:
            self.transform = logarithm_transform
            self.inv_transform = exponent_transform
        else:
            self.transform = ident_transform
            self.inv_transform = ident_transform

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
        if y is None or np.isnan(y):
            return self.sign * np.inf
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
#        f_wrapped = nan_fval_to_inf(f_wrapped)
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

    def save(self, info, save_file):
        """
        Save some information into file. Is supposed to save info during
        optimization in order to restore it.

        :param info: Information to dump.
        :param save_file: File to save information.

        :note: if save_file is None then nothing will be done. In base class\
               method just dumps `info` to `save_file` with `pickle`.
        """
        if save_file is None:
            return
        if hasattr(self, 'id'):
            if (check_file_existence(save_file) and
                    os.path.getsize(save_file) > 0):
                with open(save_file, 'rb') as fl:
                    d = pickle.load(fl)
                if not isinstance(d, dict):
                    d = {}
            else:
                d = {}
            d[self.id] = copy.copy(info)
            info = d
        with open(save_file, 'wb') as fl:
            pickle.dump(info, fl)

    def valid_restore_file(self, save_file):
        """
        Checks that `save_file` contains valid information and it could be
        restored from it.

        :param save_file: File to check.
        """
        raise NotImplementedError

    def load(self, save_file):
        """
        Loads information that was saved by :meth:`save` method.

        :param save_file: File to restore information from.

        :note: In base class method just loads from `save_file` with pickle.
        """
        with open(save_file, 'rb') as fl:
            info = pickle.load(fl)
        if hasattr(self, 'id') and isinstance(info, dict):
            return info[self.id]
        return info

    def optimize(f, variables, args=(), options={}, linear_constrain=None,
                 maxiter=None, maxeval=None, verbose=0, callback=None,
                 report_file=None, eval_file=None, save_file=None,
                 restore_file=None, restore_points_only=False,
                 restore_x_transform=None):
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
        :param linear_constrain: Linear constrain on optimized variables. It is
                                 optional argument. Could be missed in
                                 unconstrained optimizers.
        :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
        :param maxiter: Maximum number of iterations to run. If None then run
                        until converge.
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


class ContinuousOptimizer(Optimizer):
    """
    Base class for optimization of continuous variables.
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
