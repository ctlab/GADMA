import copy
import numpy as np
import sys
import os

from ..utils import Variable, ContinuousVariable
from ..utils import fix_args, cache_func
from ..utils import ensure_file_existence, check_file_existence,\
                    variables_values_repr, eval_wrapper
from ..utils import logarithm_transform, exponent_transform, ident_transform
from ..utils import apply_transform
from ..utils import serialize_meta_array, deserialize_meta_array
from ..utils import DiscreteVariable
from .optimizer_result import OptimizerResult
import pickle
from functools import partial
import types


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

    @staticmethod
    def _n_iter_string(n_iter, variables, x, y):
        """
        Returns line with iter information of optimizer.

        :param n_iter: Number of iteration of optimization.
        :param variables: list of variables which values are optimized.
        :param x: Values of variables.
        :param y: Value of target function on `x`.
        """
        x_repr = variables_values_repr(variables, x)
        metainfo = ''
        if hasattr(x, 'metadata'):
            metainfo = x.metadata
        string = f"{n_iter}\t{y}\t{x_repr}\t{metainfo}"
        return string

    def evaluate(self, f, variables, x, args=(), linear_constrain=None):
        """
        Evaluates function `f` on values `x` multiplied by sign
        (-1 if maximize).

        :param f: Target function.
        :param x: Value of parameters of `f`.
        :param args: Other arguments of f. `f(x, args)`.
        :param linear_constrain: Linear constrain on `x`.
        """
        x_tr = apply_transform(variables, self.inv_transform, x)
        if linear_constrain is not None:
            if not linear_constrain.fits(x_tr):
                x_tr, success = linear_constrain.try_to_transform(x_tr)
                if not success:
                    # warnings.warn(f"HERE IS A LITTLE PROBLEM. PLEASE CHECK "
                    #               f"IT: {x}, {x_tr}")
                    return np.inf
        y = f(x_tr, *args)
        if y is None or np.isnan(y):
            return np.inf
        return self.sign * y

    def _prepare_f_for_opt(self, f, args, eval_file, cache=True):
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
        # Wrap for automatic evaluation logging
        f_wrapped = eval_wrapper(f_wrapped, eval_file)
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

    def _create_run_info(self):
        """
        Returns the initial run_info.
        """
        result = OptimizerResult(
            x=None,
            y=self.sign * np.inf,
            success=False,
            status=0,
            message="",
            X=[],
            Y=[],
            n_eval=0,
            n_iter=-1,
        )
        run_info = types.SimpleNamespace(result=result)
        return run_info

    @property
    def run_info(self):
        if self._run_info is None:
            self._run_info = self._create_run_info()
        return self._run_info

    @run_info.setter
    def run_info(self, new_run_info):
        self._run_info = new_run_info
        if new_run_info is not None:
            assert hasattr(self._run_info, "result")

    def _apply_transform_to_run_info(self, run_info, x_transform, y_transform):
        """
        Returns copy of run_info with transformed `result` field.
        """
        run_info_tr = copy.deepcopy(run_info)
        # transform result
        run_info_tr.result.apply_transforms(x_transform, y_transform)
        return run_info_tr

    def _update_run_info(self, run_info, x_best, y_best,
                         X, Y, n_eval, **update_kwargs):
        """
        Updates run_info after one iteration in-place.
        """
        run_info.result.n_iter += 1
        run_info.result.n_eval += n_eval
        run_info.result.x = x_best
        run_info.result.y = y_best
        run_info.result.X_out = copy.copy(X)
        run_info.result.Y_out = copy.copy(Y)
        run_info.result.X.extend(run_info.result.X_out)
        run_info.result.Y.extend(run_info.result.Y_out)
        return run_info

    def save(self, run_info, save_file):
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
        info = self._apply_transform_to_run_info(
            run_info,
            x_transform=serialize_meta_array,
            y_transform=ident_transform
        )
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
        try:
            info = self.load(save_file)
        except Exception:
            return False
        if not hasattr(info, "result"):
            return False
        return True

    def load(self, save_file):
        """
        Loads information that was saved by :meth:`save` method.

        :param save_file: File to restore information from.

        :note: In base class method just loads from `save_file` with pickle.
        """
        with open(save_file, 'rb') as fl:
            info = pickle.load(fl)
        if hasattr(self, 'id') and isinstance(info, dict):
            info = info[self.id]
        run_info = self._apply_transform_to_run_info(
            info,
            x_transform=deserialize_meta_array,
            y_transform=ident_transform
        )
        return run_info

    def process_optimize_kwargs(self, f, variables, **optimize_kwargs):
        raise NotImplementedError

    def write_report(self, variables, run_info, report_file):
        """
        Write report about one iteration of global optimization in report file.

        :param variables: Variables of run.
        :param run_info: Instance of class that contains run info, i.e. current
                         result.
        :param report_file: File to write the report. If None then report
                            should be printed to stdout.

        :note: All values are reported as is, i.e. `X_out`, `x_best` in \
               `run_info` should be already translated from log scale if \
               optimization did so; `Y_out` and `y_best` must be already \
               multiplied by -1 if we have maximization instead of \
               minimization.
        """
        if report_file is not None:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        self._write_report_to_stream(variables, run_info, stream)
        if report_file is not None:
            stream.close()

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        raise NotImplementedError

    def _optimize(self, f, variables,
                  maxiter, maxeval, iter_callback, **optimize_kwargs):
        """
        Main part of optimization. Assumes that `X_init` and `Y_init` are
        correct initial design points. Should run iterations of global
        optimization with callback calling after each iteration.

        :param f: Cached and wrapped (e.g. eval logging) objective function.
        :param variables: Variables for `f`. They are supposed to have
                          transformed domain according to optimizer transform.
        :param X_init: Correct initial points for `f`.
        :param Y_init: Value of `f` on `X_init`.
        :param maxiter: Maximum number of iterations to run.
        :param maxeval: Maximum number of evaluations to run.
        :param iter_callback: Callback to run after each iteration. It has the
            following notation: `iter_callback(x_best,
            y_best, X_iter, Y_iter, **update_kwargs)`, where
            **update_kwargs are arguments of
            :meth:`GlobalOptimizer._update_run_info_except_result`
        """
        raise NotImplementedError

    def optimize(self, f, variables, args=(),
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, report_file=None, eval_file=None,
                 save_file=None, restore_file=None, restore_points_only=False,
                 restore_x_transform=None, **optimize_kwargs):
        r"""
        Return best values of `variables` that minimizes/maximizes
        the function `f`.

        :param f: function to minimize/maximize. The usage must be the
                  following: f(x, \*args), where x is list of values.
        :type f: funstion
        :param variables: list of variables of the function.
        :type variables: list of :class:`gadma.utils.Variable`
        :param args: Additional arguments of function `f`.
        :type args: tuple
        :param linear_constrain: Linear constrain on variables.
        :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
        :param maxiter: maximum number of genetic algorithm's generations.
        :type maxiter: int
        :param maxeval: maximum number of function evaluations.
        :type maxeval: int
        :param verbose: Verbosity of the output. If 0 then no output.
        :type verbose: int
        :param callback: callback to call after each generation.
                         It will be called as callback(x, y), where x, y -
                         best_solution of generation and its fitness.
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
        # Create run_info that will be saved during the optimization
        self.run_info = None

        # Check variables
        self.check_variables(variables)

        # Create logging files
        if eval_file is not None:
            eval_file = ensure_file_existence(eval_file)
        if verbose > 0 and report_file is not None:
            report_file = ensure_file_existence(report_file)
        if save_file is not None:
            save_file = ensure_file_existence(save_file)

        # prepare variables and transform their domain
        vars_in_opt = copy.deepcopy(variables)
        for var in vars_in_opt:
            if not isinstance(var, DiscreteVariable):
                var.domain = self.transform(var.domain)

        # Prepare function to use it.
        # Fix args and cache
        prepared_f = self._prepare_f_for_opt(f, args, eval_file)
        # wrap for automatic transform of x's and multiplication by sign of y's
        f_in_opt = partial(self.evaluate, prepared_f, vars_in_opt)
        # Fix linear constrain as extra args
        f_in_opt = fix_args(f_in_opt, (), linear_constrain)

        # Restore run_info
        if restore_file is not None and self.valid_restore_file(restore_file):
            restored_run_info = self.load(restore_file)
            if restore_x_transform is not None:
                def y_transform(y):
                    return self.sign * y
                restored_run_info = self._apply_transform_to_run_info(
                    run_info=restored_run_info,
                    x_transform=restore_x_transform,
                    y_transform=ident_transform
                )
            if not restore_points_only:
                self.run_info = restored_run_info
            else:
                self.run_info.result.x = restored_run_info.result.x
                self.run_info.result.y = restored_run_info.result.y
                self.run_info.result.X_out = restored_run_info.result.X_out
                self.run_info.result.Y_out = restored_run_info.result.Y_out
            self.run_info.result.message = str(self.run_info.result.message)
            self.run_info.result.message += "(RESTORED)"

        if self.run_info.result.success:
            return self.run_info.result

        optimize_kwargs = self.process_optimize_kwargs(f=prepared_f,
                                                       variables=variables,
                                                       **optimize_kwargs)

        def iter_callback(x, y, X_iter, Y_iter, **update_kwargs):
            x = apply_transform(vars_in_opt, self.inv_transform, x)
            y = self.sign * y
            X = [apply_transform(vars_in_opt, self.inv_transform, _x)
                 for _x in X_iter]
            Y = [self.sign * _y for _y in Y_iter]
            x_best = self.run_info.result.x
            y_best = self.run_info.result.y
            if x_best is None or self.sign * y < self.sign * y_best:
                x_best = x
                y_best = y
            n_eval = prepared_f.cache_info.misses
            self.run_info = self._update_run_info(
                run_info=self.run_info,
                x_best=x_best,
                y_best=y_best,
                X=X,
                Y=Y,
                n_eval=n_eval-self.run_info.result.n_eval,
                **update_kwargs)
            # Write report
            if verbose > 0 and self.run_info.result.n_iter % verbose == 0:
                self.write_report(variables, self.run_info, report_file)
            # Save run_info
            self.save(self.run_info, save_file)
            # Call callback
            if callback is not None:
                callback(self.run_info.result.x, self.run_info.result.y)

        self._optimize(f=f_in_opt,
                       variables=vars_in_opt,
                       maxiter=maxiter,
                       maxeval=maxeval,
                       iter_callback=iter_callback,
                       **optimize_kwargs)

        self.run_info.result.success = True
        # Save last run_info
        self.save(self.run_info, save_file)

        return self.run_info.result


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


class UnconstrainedOptimizer(Optimizer):
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


class ConstrainedOptimizer(Optimizer):
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
