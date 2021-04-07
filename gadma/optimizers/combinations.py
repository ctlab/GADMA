import sys
from functools import wraps
import numpy as np

from .global_optimizer import GlobalOptimizer
from .optimizer import ConstrainedOptimizer
from .optimizer_result import OptimizerResult
from ..utils import DiscreteVariable, sort_by_other_list, get_correct_dtype


class GlobalOptimizerAndLocalOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for run of global optimizer followed by local optimizer. It is
    classified as global optimizer with constrains. Function :meth:`optimize`
    got function and variables, runs global optimizer then filter discrete
    variables out and optimize the rest with local optimizer.

    :param global_optimizer: Global optimizer to use inside.
    :type global_optimizer: :class:`gadma.optimizers.GlobalOptimizer`
    :param local_optimizer: Local optimizer to use inside.
    :type local_optimizer: :class:`gadma.optimizers.LocalOptimizer`
    """
    def __init__(self, global_optimizer, local_optimizer):
        self.global_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        if self.global_optimizer.maximize != self.local_optimizer.maximize:
            raise ValueError("Global and local optimizers must both maximize "
                             "(or minimize) the function.")
        # For restore and save files
        if not hasattr(self.global_optimizer, "id"):
            self.global_optimizer.id = "global_optimizer"
        if not hasattr(self.local_optimizer, "id"):
            self.local_optimizer.id = "local_optimizer"

        super(GlobalOptimizerAndLocalOptimizer, self).__init__(
            log_transform=False, maximize=self.global_optimizer.maximize)

    @staticmethod
    def _get_filter(variables):
        """
        Returns filter to filter out discrete variables for local optimizer.
        Returns numpy array of bools.

        :param variables: List of Variables to get filter for.
        """
        is_not_discrete = [not isinstance(var, DiscreteVariable)
                           for var in variables]
        return np.array(is_not_discrete, dtype=bool)

    def _transform_x(self, x, base_x, is_filtered):
        """
        Transforms vector `x` back from local optimizer according to filter.

        :param x: Vector from local optimizer.
        :param base_x: Base vector from global optimizer that have fixed values
                       that were filtered out.
        :param is_filtered: Filter that was used on `base_x`.
        """
        full_x = np.array(base_x, dtype=get_correct_dtype(base_x))
        full_x[is_filtered] = x
        return full_x

    def _f_for_local_opt(self, f, variables, x_best):
        """
        Transforms objective function to take vectors from local optimization.
        """
        is_filtered = self._get_filter(variables)

        @wraps(f)
        def wrapper(x, *args):
            full_x = self._transform_x(x, x_best, is_filtered)
            return f(full_x, *args)
        return wrapper

    def _callback_for_local_opt(self, callback, variables, x_best):
        """
        Transforms callback function to take vectors from local optimization.
        """
        if callback is None:
            return None
        is_filtered = self._get_filter(variables)

        @wraps(callback)
        def wrapper(x, y):
            full_x = self._transform_x(x, x_best, is_filtered)
            return callback(full_x, y)
        return wrapper

    def optimize(self, f, variables, args=(), global_num_init=50,
                 global_num_init_const=None, X_init=None, Y_init=None,
                 local_options={}, linear_constrain=None,
                 global_maxiter=None, local_maxiter=None,
                 global_maxeval=None, local_maxeval=None,
                 verbose=0, callback=None, eval_file=None,
                 report_file=None, save_file=None,
                 restore_file=None, restore_points_only=False,
                 global_x_transform=None, local_x_transform=None):
        """
        :param f: Objective function.
        :type f: func
        :param variables: List of objective function variables.
        :type variables: list of class:`gadma.utils.VariablesPool`
        :param args: Arguments of `f`.
        :type args: tuple
        :param global_num_init: Number of initial points for global optimizer.
        :type global_num_init: int
        :param X_init: List of initial vectors.
        :type X_init: list
        :param Y_init: List of values of target function on points of `X_init`.
        :type Y_init: list
        :param local_options: Options for local optimizer.
        :type local_options: dict
        :param linear_constrain: Linear constrain on variables.
        :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
        :param global_maxiter: Maximum number of global optimizer iterations
                               to run.
        :type global_maxiter: int
        :param global_maxeval: Maximum number of function evaluation during
                               global optimization.
        :type global_maxeval: int
        :param local_maxiter: Maximum number of local optimizer iterations
                              to run.
        :type local_maxiter: int
        :param local_maxeval: Maximum number of function evaluation during
                              local optimization.
        :type local_maxeval: int
        :param verbose: Varbosity of reports. If 0 then no output.
        :type verbose: int
        :param callback: callback to run after each iteration of both
                         optimizers.
        :type callback: func
        :param eval_file: File to save of objective function evaluations.
        :type eval_file: str
        :param report_file: File to save report each `verbose` iteration. If
                            None and `verbose` > 0 then report will be printed
                            to stdout.
        :type report_file: str
        :param save_file: File to save information during the run.
        :type save_file: str
        :param restore_file: File to restore previous run that was saved by
                             :meth:`save` method.
        :type restore_file: str
        :param restore_points_only: Restore run last results and run again from
                                    it.
        :type restore_points_only: bool
        :param global_x_transform: Transformation of vectors after restore
                                   before run of global optimizer.
        :type global_x_transform: func
        :param local_x_transform: Transformation of vectors after restore
                                  before run of local optimizer.
        :type local_x_transform: bool
        """
        if report_file is not None:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        if verbose != 0:
            print(f"--Start global optimization {self.global_optimizer.id}--",
                  file=stream)
        if report_file:
            stream.close()

        # Run global optimizer
        global_result = self.global_optimizer.optimize(
            f=f,
            variables=variables,
            args=args,
            num_init=global_num_init,
            num_init_const=global_num_init_const,
            X_init=X_init,
            Y_init=Y_init,
            linear_constrain=linear_constrain,
            maxiter=global_maxiter,
            maxeval=global_maxeval,
            verbose=verbose,
            callback=callback,
            report_file=report_file,
            eval_file=eval_file,
            save_file=save_file,
            restore_file=restore_file,
            restore_points_only=restore_points_only,
            restore_x_transform=global_x_transform
        )
        if report_file is not None:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        if verbose != 0:
            print(f"--Finish global optimization {self.global_optimizer.id}--",
                  file=stream)
            print("Result:\n", global_result, file=stream)

        # Transform best x to local optimizer as x0 and functions for local
        x_best = np.array(global_result.x,
                          dtype=get_correct_dtype(global_result.x))
        y_best = global_result.y
        is_not_discrete = self._get_filter(variables)
        x0 = x_best[is_not_discrete]

        variables_local = np.array(variables)[is_not_discrete]

        f_local = self._f_for_local_opt(f, variables, x_best)
        callback_local = self._callback_for_local_opt(callback, variables,
                                                      x_best)
        if verbose != 0:
            print(f"--Start local optimization {self.local_optimizer.id}--",
                  file=stream)

        if report_file is not None:
            stream.close()

        # Run local optimizer
        local_result = self.local_optimizer.optimize(
            f=f_local,
            variables=variables_local,
            x0=x0,
            args=args,
            options=local_options,
            linear_constrain=linear_constrain,
            maxiter=local_maxiter,
            maxeval=local_maxeval,
            verbose=verbose,
            callback=callback_local,
            eval_file=eval_file,
            report_file=report_file,
            save_file=save_file,
            restore_file=restore_file,
            restore_points_only=restore_points_only,
            restore_x_transform=local_x_transform
        )
        # Create result
        success = local_result.success
        message = f"GLOBAL OPTIMIZATION: {global_result.message}; "\
                  f"LOCAL OPTIMIZATION: {local_result.message}"
        status = local_result.status

        if not np.isinf(local_result.y):
            y_best = local_result.y
            x_best[is_not_discrete] = local_result.x
        ga_maximize = self.global_optimizer.maximize
        X_out, Y_out = sort_by_other_list(global_result.X_out,
                                          global_result.Y_out,
                                          reverse=ga_maximize)
        if np.allclose(Y_out[0], global_result.y):
            X_out[0] = x_best
        else:
            X_out.insert(0, x_best)
            Y_out.insert(0, y_best)

        X_total = list(global_result.X) + list(local_result.X)
        Y_total = list(global_result.Y) + list(local_result.Y)
        n_eval = global_result.n_eval + local_result.n_eval
        n_iter = global_result.n_iter + local_result.n_iter
        result = OptimizerResult(x_best, y_best, success, status, message,
                                 X_total, Y_total, n_eval, n_iter,
                                 X_out, Y_out)

        if report_file:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        if verbose != 0:
            print(f"--Finish local optimization {self.local_optimizer.id}--",
                  file=stream)
            print("Result:\n", result, file=stream)
        if report_file:
            stream.close()
        return result
