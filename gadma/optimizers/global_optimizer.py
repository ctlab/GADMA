from .optimizer import Optimizer
from ..utils import ContinuousVariable, DiscreteVariable, get_correct_dtype
from ..utils import serialize_meta_array, deserialize_meta_array
from ..utils import ensure_file_existence, fix_args, ident_transform
from ..utils import sort_by_other_list, eval_wrapper

import numpy as np
from functools import partial
import copy


_registered_global_optimizers = {}


class GlobalOptimizer(Optimizer):
    """
    Base class for global optimization.
    See :class:`gadma.optimizers.Optimizer` for more information.
    """
    def __init__(self, log_transform=False, maximize=False):
        super(GlobalOptimizer, self).__init__(log_transform, maximize)
        self.X = list()
        self.Y = list()
        self.run_info = None

    def randomize(self, variables, random_type='resample',
                  custom_rand_gen=None):
        """
        Generate random solution. The type of generation could be set
        to one of three operators:

        * 'uniform' - uniform over domain.

        * 'resample' - call :meth:`resample` method for all variables.

        * 'custom' - sample values of parameters from `custom_rand_gen`.
        """
        if random_type == 'uniform':
            arr = []
            for var in variables:
                if isinstance(var, ContinuousVariable):
                    arr.append(np.random.uniform(*var.domain))
                else:
                    arr.append(np.random.choice(var.domain))
            return np.array(arr, dtype=get_correct_dtype(arr))
        elif random_type == 'resample':
            arr = [var.resample() for var in variables]
            return np.array(arr, dtype=get_correct_dtype(arr))
        elif random_type == 'custom':
            return custom_rand_gen(variables)
        else:
            raise ValueError(f"Unknown type of generation of random "
                             f"solution: {random_type}.")

    def initial_design(self, f, variables, num_init,
                       X_init=None, Y_init=None,
                       random_type='resample', custom_rand_gen=None):
        """
        Performs initial design for optimization. All x's are transformed and
        all y's will be multiplied by sign.

        :param f: function to use for evaluations. Note that it should be
                  without arguments. Use :meth:`self.fix_f_and_args` to
                  get such function from another one with arguments.
        :param variables: variables of function. They are used for random
                          generation of their values.
        :param num_init: number of initial solutions.
        :param X_init: list of some initial solutions. (Not transformed!)
        :param Y_init: list of function values on the initial solutions.
                       (not multiplied by sign!)

        :returns: pair of lists X and Y. Initial points and value of fitness\
                  function on them.
        """
        X = list()
        Y = list()
        if X_init is not None:
            X = [self.transform(x) for x in X_init]
            if Y_init is not None:
                Y = [self.sign * y for y in Y_init]
            else:
                Y = list()
            for x in X_init[len(Y):]:
                X.append(x)
                Y.append(f(x))
        for _ in range(num_init - len(X)):
            x = self.randomize(variables, random_type, custom_rand_gen)
            X.append(self.transform(x))
            Y.append(self.sign * f(x))
        return X, Y

    def _create_run_info(self):
        raise NotImplementedError

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
        return self._apply_transform_to_run_info_except_result(
            run_info_tr, x_transform, y_transform
        )

    def _apply_transform_to_run_info_except_result(self, run_info,
                                                   x_transform, y_transform):
        """
        Transforms all fields of run_info except `result` in-place.
        """
        raise NotImplementedError

    def _update_run_info(self, run_info, x_best, y_best,
                         X, Y, n_eval, **update_kwargs):
        """
        Updates run_info after one iteration.
        """
        run_info.result.n_iter += 1
        run_info.result.n_eval += n_eval
        run_info.result.x = x_best
        run_info.result.y = y_best
        run_info.result.X_out = copy.copy(X)
        run_info.result.Y_out = copy.copy(Y)
        run_info.result.X.extend(run_info.result.X_out)
        run_info.result.Y.extend(run_info.result.Y_out)
        run_info = self._update_run_info_except_result(run_info,
                                                       **update_kwargs)
        return run_info

    def _update_run_info_except_result(self, run_info, **update_kwargs):
        """
        Updates not standard fields of run_info.
        """
        raise NotImplementedError

    def save(self, run_info, save_file):
        """
        Save some values of genetic algorithm to file.
        """
        if save_file is None:
            return
        info = self._apply_transform_to_run_info(
            run_info,
            x_transform=serialize_meta_array,
            y_transform=ident_transform
        )
        super(GlobalOptimizer, self).save(info, save_file)

    def load(self, save_file):
        """
        Load some values of genetic algorithm from file.
        """
        info = super(GlobalOptimizer, self).load(save_file)
        run_info = self._apply_transform_to_run_info(
            info,
            x_transform=deserialize_meta_array,
            y_transform=ident_transform
        )

        return run_info

    def write_report(self, variables, run_info, report_file):
        """
        Write report about one generation in report file.

        :note: All values are reported as is, i.e. `X_gen`, `x_best` should be\
               already translated from log scale if optimization did so;\
               `Y_gen` and `y_best` must be already multiplied by -1 if we\
               have maximization instead of minimization.
        """
        raise NotImplementedError

    def parent_write_report(self, n_iter, variables, x, y, report_file):
        """
        Wrapper for parent method.
        """
        return super(GlobalOptimizer, self).write_report(n_iter, variables,
                                                         x, y, report_file)

    def _update_X_init_Y_init(self, X_init, Y_init, X_out, Y_out):
        # 1. No X_init and no Y_init
        if X_init is None:
            new_X_init = copy.copy(X_out)
            new_Y_init = copy.copy(Y_out)
        # 2. X_init but no Y_init
        elif Y_init is None:
            # Then we put X_gen and Y_gen first and extend them with X_init
            new_X_init = copy.copy(X_out)
            new_X_init.extend(X_init)
            new_Y_init = copy.copy(Y_out)
        # 3. X_init and Y_inits exists and are of equal length
        elif len(X_init) == len(Y_init):
            # Extend X_init and Y_init with X_out and Y_out
            new_X_init = copy.copy(X_init)
            new_Y_init = copy.copy(Y_init)
            if X_out is not None:
                X_init.extend(X_out)
                if Y_out is not None:
                    Y_init.extend(Y_out)
        # 4. X_init and Y_inits exists but have different length
        else:
            assert len(X_init) > len(Y_init)
            new_X_init = copy.copy(X_init[:len(Y_init)])
            new_X_init.extend(X_out)
            new_X_init.extend(X_init[len(Y_init):])
            new_Y_init = copy.copy(Y_init)
            new_Y_init.extend(Y_out)
        return new_X_init, new_Y_init

    def _optimize(f, variables, X_init, Y_init, maxiter, maxeval, callback):
        raise NotImplementedError

    def optimize(self, f, variables, args=(), num_init=50, num_init_const=None,
                 X_init=None, Y_init=None,
                 linear_constrain=None, maxiter=None, maxeval=None,
                 verbose=0, callback=None, report_file=None, eval_file=None,
                 save_file=None, restore_file=None, restore_points_only=False,
                 restore_x_transform=None):
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
        :param num_init: Number of points in initial design.
        :type num_init: int
        :param num_init_const: If None then `num_init` is used. Otherwise
                               number of points in initial design is equal to
                               `num_init_const` \* len(`variables`).
        :param X_init: list of initial values.
        :type X_init: list of vectors.
        :param Y_init: value of function `f` on initial values from `X_init`.
        :type Y_init: list of floats
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

        # Check for num_init
        if num_init_const is not None:
            num_init = num_init_const * len(variables)

        # Create logging files
        if eval_file is not None:
            eval_file = ensure_file_existence(eval_file)
        if verbose > 0 and report_file is not None:
            report_file = ensure_file_existence(report_file)
        if save_file is not None:
            save_file = ensure_file_existence(save_file)

        # Prepare function to use it.
        # Fix args and cache
        prepared_f = self.prepare_f_for_opt(f, args)
        # Wrap for automatic evaluation logging
        finally_wrapped_f = eval_wrapper(prepared_f, eval_file)
        # wrap for automatic transform of x's and multiplication by sign of y's
        f_in_opt = partial(self.evaluate, finally_wrapped_f)
        # Fix linear constrain as extra args
        f_in_opt = fix_args(f_in_opt, (), linear_constrain)

        # prepare variables and transform their domain
        vars_in_opt = copy.deepcopy(variables)
        for var in vars_in_opt:
            if not isinstance(var, DiscreteVariable):
                var.domain = self.transform(var.domain)

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
                num_init = len(restored_run_info.result.X_out)
            X_init, Y_init = self._update_X_init_Y_init(
                X_init,
                Y_init,
                restored_run_info.result.X_out,
                restored_run_info.result.Y_out
            )

        # Perform initial design. X_init and Y_init have transformed values now
        X_init, Y_init = self.initial_design(finally_wrapped_f, variables,
                                             num_init,
                                             X_init, Y_init, self.random_type,
                                             self.custom_rand_gen)
        X_init, Y_init = sort_by_other_list(X_init, Y_init, reverse=False)

        def iter_callback(x_best, y_best, X_iter, Y_iter, **update_kwargs):
            # x's and y's are transformed lists
            x = self.inv_transform(x_best)
            y = self.sign * y_best
            X = [self.inv_transform(x) for x in X_iter]
            Y = [self.sign * y for y in Y_iter]
            n_eval = prepared_f.cache_info.misses
            self.run_info = self._update_run_info(
                run_info=self.run_info,
                x_best=x,
                y_best=y,
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

        # Run callback for initial design
        iter_callback(x_best=X_init[0],
                      y_best=Y_init[0],
                      X_iter=X_init,
                      Y_iter=Y_init)

        self._optimize(f=f_in_opt,
                       variables=vars_in_opt,
                       X_init=X_init,
                       Y_init=Y_init,
                       maxiter=maxiter,
                       maxeval=maxeval,
                       iter_callback=iter_callback)

        return self.run_info.result


def register_global_optimizer(id, optimizer):
    """
    Registers the specified global optimizer.
    """
    if id in _registered_global_optimizers:
        raise ValueError(f"Optimizer '{id}' is already registered.")
    if not issubclass(optimizer, GlobalOptimizer):
        raise ValueError("Optimizer is not global.")
    _registered_global_optimizers[id] = optimizer
    optimizer.id = id


def get_global_optimizer(id):
    """
    Returns the global optimizer with the specified id.
    """
    if id not in _registered_global_optimizers:
        raise ValueError(f"Optimizer '{id}' is not registered")
    return _registered_global_optimizers[id]()


def all_global_optimizers():
    """
    Returns an iterator over all registered global optimizers.
    """
    for optim in _registered_global_optimizers.values():
        yield optim()
