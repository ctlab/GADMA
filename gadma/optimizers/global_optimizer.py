from .optimizer import Optimizer
from ..utils import ContinuousVariable, get_correct_dtype
from ..utils import sort_by_other_list, apply_transform

import numpy as np
import copy


_registered_global_optimizers = {}


class GlobalOptimizer(Optimizer):
    """
    Base class for global optimization.
    See :class:`gadma.optimizers.Optimizer` for more information.

    This class provides methods for initial design and has additional kwargs
    for it, e.g. `X_init` and `Y_init` in comparison with parent class.

    :param random_type: Type of random generation during initial design.
                        Could be:

                        * 'uniform'
                        * 'resample'
                        * 'custom'

                        See help(:meth:`GlobalOptimizer.randomize`) for more
                        information.
    :type random_type: str
    :param custom_rand_gen: Random generator for 'custom' random_type.
                            Provide generator from variables:
                            custom_rand_gen(variables) = values
    :type custom_rand_gen: func
    :param log_transform: If True then logarithm will be applied for the
                          parameter space.
    :type log_transform: bool
    :param maximize: If True then optimization will maximize function.
    :type maximize: bool
   """
    def __init__(self, random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        super(GlobalOptimizer, self).__init__(log_transform, maximize)
        self.random_type = random_type
        self.custom_rand_gen = custom_rand_gen
        if self.random_type == 'custom' and self.custom_rand_gen is None:
            raise ValueError("Please specify custom random generator "
                             "(custom_rand_gen) for 'custom' type of random "
                             "sampling.")
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
            arr = np.array(arr, dtype=get_correct_dtype(arr))
        elif random_type == 'resample':
            arr = [var.resample() for var in variables]
            arr = np.array(arr, dtype=get_correct_dtype(arr))
        elif random_type == 'custom':
            arr = custom_rand_gen(variables)
        else:
            raise ValueError(f"Unknown type of generation of random "
                             f"solution: {random_type}.")
        return arr

    def initial_design(self, f, variables, num_init,
                       X_init=None, Y_init=None,
                       random_type='resample', custom_rand_gen=None):
        """
        Performs initial design for optimization. All x's will be transformed
        according to `log_transform` and all y's will be multiplied by sign.

        :param f: function to use for evaluations. Note that it should be
                  without arguments. Use :meth:`gadma.utils.fix_args` to
                  get such function from another one with arguments.
        :param variables: variables of function. They are used for random
                          generation of their values if random_type is
                          set to `reasmple`.
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
            X = [apply_transform(variables, self.transform, x)
                 for x in X_init]
            if Y_init is not None:
                Y = [self.sign * y for y in Y_init]
            else:
                Y = list()
            for x in X_init[len(Y):]:
                X.append(x)
                Y.append(f(x))
        for _ in range(num_init - len(X)):
            x = self.randomize(variables, random_type, custom_rand_gen)
            X.append(x)
            Y.append(f(x))
        return X, Y

    def _update_X_init_Y_init(self, X_init, Y_init, X_out, Y_out):
        """
        Updates X_init and Y_init according to restored X_out and Y_out.
        It is just union of two arrays with initial points that takes into
        account if some of this arrays are None.

        :param X_init: Initial points.
        :param Y_init: Value of objective function on initial points `X_init`.
        :param X_out: Restored points to start with.
        :param Y_out: Value of objective function on `X_out`.
        """
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

    def process_optimize_kwargs(self, f, variables,
                                X_init, Y_init, num_init, num_init_const):
        r"""
        Returns kwargs with `X_init` and `Y_init` to run :meth:`_optimize`.

        :param f: Objective function. If should take no arguments except `x`
                  but `x` should not be transformed.
        :param variables: Variables for `f`.
        :param X_init: Initial points for optimization. They are updated with
                       points from `self.run_info` if it was restored from
                       previous run.
        :param Y_init: Value of objective function `f` on `X_init`.
        :param num_init: Number of initial points. If `X_init` does not have
                         enough points they are sampled by :meth:`randomize`.
        :param num_init_const: If None then `num_init` is used. Otherwise
                               number of points in initial design is equal to
                               `num_init_const` \* len(`variables`).
        """
        # Our X_out and Y_out are restored at that point, we want update
        X_init, Y_init = self._update_X_init_Y_init(
            X_init,
            Y_init,
            self.run_info.result.X_out,
            self.run_info.result.Y_out
        )
        # Check for number of initial points
        if num_init_const is not None:
            num_init = num_init_const * len(variables)
        # If we restored our run then we do not need to evaluate so many points
        if self.run_info.result.n_eval > 0:
            num_init = len(self.run_info.result.X_out)
        # Just to be sure
        assert isinstance(num_init, int)

        # Perform initial design. X_init and Y_init have transformed values now
        X_init, Y_init = self.initial_design(f, variables,
                                             num_init,
                                             X_init, Y_init, self.random_type,
                                             self.custom_rand_gen)
        X_init, Y_init = sort_by_other_list(X_init, Y_init, reverse=False)

        # Return our kwargs for _optimize
        return {"X_init": X_init,
                "Y_init": Y_init}

    def _optimize(self, f, variables, X_init, Y_init,
                  maxiter, maxeval, iter_callback):
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
                  following: `f(x, \*args)`, where `x` is a list of values.
        :type f: func
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
        :param maxiter: maximum number of algorithm iterations.
        :type maxiter: int
        :param maxeval: maximum number of function evaluations.
        :type maxeval: int
        :param verbose: Verbosity of the report output. If 0 then no output.
        :type verbose: int
        :param callback: callback to call after each iteration.
                         It will be called as callback(x, y), where x, y -
                         best solution of the iteration and its fitness.
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
        :param restore_x_transform: Restore points but transform them with
                                    given transform before usage in this run.
        :type restore_x_transform: func
        """
        optimize_kwargs = {"X_init": X_init,
                           "Y_init": Y_init,
                           "num_init": num_init,
                           "num_init_const": num_init_const}
        return super(GlobalOptimizer, self).optimize(
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
