from .global_optimizer import GlobalOptimizer
from .optimizer import ConstrainedOptimizer
from.optimizer_result import OptimizerResult
from ..utils import DiscreteVariable, sort_by_other_list

from functools import wraps
import numpy as np


class GlobalOptimizerAndLocalOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    def __init__(self, global_optimizer, local_optimizer):
        self.global_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        if self.global_optimizer.maximize != self.local_optimizer.maximize:
            raise ValueError("Global and local optimizers must both maximize "
                             "(or minimize) the function.")
        super(GlobalOptimizerAndLocalOptimizer, self).__init__(
            log_transform=False, maximize=self.global_optimizer.maximize)

    def _get_filter(self, variables):
        is_not_discrete = [not isinstance(var, DiscreteVariable)
                           for var in variables]
        return np.array(is_not_discrete, dtype=bool)

    def _transform_x(self, x, base_x, is_filtered):
        full_x = np.array(base_x)
        full_x[is_filtered] = x
        return full_x

    def _f_for_local_opt(self, f, variables, x_best):
        is_filtered = self._get_filter(variables)
        @wraps(f)
        def wrapper(x, *args):
            full_x = self._transform_x(x, x_best, is_filtered)
            return f(full_x, *args)
        return wrapper

    def _callback_for_local_opt(self, callback, variables, x_best):
        if callback is None:
            return None
        is_filtered = self._get_filter(variables)
        @wraps(callback)
        def wrapper(x, y):
            full_x = self._transform_x(x, x_best, is_filtered)
            return callback(full_x, y)
        return wrapper 

    def optimize(self, f, variables, args=(), global_num_init=50,
                 X_init=None, Y_init=None, local_options={},
                 global_maxiter=None, local_maxiter=None,
                 global_maxeval=None, local_maxeval=None,
                 verbose=0, callback=None, eval_file=None,
                 report_file=None, save_file=None):
        if report_file:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        print(f"--Start global optimization {self.global_optimizer.id}--",
              file=stream)
        if report_file:
            stream.close()

        # Run global optimizer
        global_result = self.global_optimizer.optimize(f, variables,
                                                       args, global_num_init,
                                                       X_init, Y_init,
                                                       global_maxiter,
                                                       global_maxeval,
                                                       verbose, callback,
                                                       report_file, eval_file,
                                                       save_file)
        if report_file:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        print(f"--Finish global optimization {self.global_optimizer.id}--",
              file=stream)
        print("Result:\n", global_result)

        # Transform best x to local optimizer as x0 and functions for local
        x_best = np.array(global_result.x)
        is_filtered = self._get_filter(variables)
        x0 = x_best[is_filtered]
        variables_local = np.array(variables)[is_filtered]

        f_local = self._f_for_local_opt(f, variables, x_best)
        callback_local = self._callback_for_local_opt(callback, variables,
                                                      x_best)

        print(f"--Start local optimization {self.local_optimizer.id}--",
              file=stream)

        if report_file:
            stream.close()

        # Run local optimizer
        local_result = self.local_optimizer.optimize(f_local, variables_local,
                                                     x0, args, local_options,
                                                     local_maxiter,
                                                     local_maxeval,
                                                     verbose, callback_local,
                                                     eval_file, report_file)
        # Create result
        success = local_result.success
        message = f"GLOBAL OPTIMIZATION: {global_result.message}; "\
                  f"LOCAL OPTIMIZATION: {local_result.message}"
        status = local_result.status

        y_best = local_result.y
        x_best = global_result.x
        x_best[is_filtered] = local_result.x
        X_out, Y_out = sort_by_other_list(global_result.X_out,
                                          global_result.Y_out,
                                          reverse=self.global_optimizer.maximize)
        if np.allclose(Y_out[0], global_result.y):
            X_out[0] = x_best
        else:
            X_out.insert(0, x_best)
            Y_out.insert(0, y_best)

        X_total = global_result.X + local_result.X
        Y_total = global_result.Y + local_result.Y
        n_eval = global_result.n_eval + local_result.n_eval
        n_iter = global_result.n_iter + local_result.n_iter
        result = OptimizerResult(x_best, y_best, success, status, message,
                                 X_total, Y_total, n_eval, n_iter, X_out, Y_out)

        if report_file:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        print(f"--Finish local optimization {self.local_optimizer.id}--",
              file=stream)
        print("Result:\n", result, file=stream)
        if report_file:
            stream.close()
        return result
