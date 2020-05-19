from .optimizer import Optimizer, ConstrainedOptimizer

import copy
import scipy
import numpy as np


_registered_global_optimizers = {}


class GlobalOptimizer(Optimizer):

    def __init__(self, log_transform=False, maximize=False):
        super(GlobalOptimizer, self).__init__(log_transform, maximize)
        self.X = list()
        self.Y = list()

    def randomize(self, variables, random_type='resample'):
        """
        Generate random solution. The type of generation could be set
        to one of three operators:

        * 'uniform' - uniform over domain.

        * 'resample' - resample variables.
        """
        if random_type == 'uniform':
            return np.array([np.random.uniform(*var.domain)
                             for var in variables], dtype=object)
        elif random_type == 'resample':
            arr = [var.resample() for var in variables]
            return np.array(arr, dtype=object)
        else:
            raise ValueError(f"Unknown type of generation of random "
                             f"solution: {random_type}.")

    def initial_design(self, f, variables, num_init,
                       X_init=None, Y_init=None, random_type='resample'):
        """
        Performs initial design for optimization.

        :param f: function to use for evaluations. Note that it should be
                  without arguments. Use :method:`self.fix_f_and_args` to
                  get such function from another one with arguments.
        :param variables: variables of function. They are used for random
                          generation of their values.
        :param num_init: number of initial solutions.
        :param X_init: list of some initial solutions.
        :param Y_init: list of function values on the initial solutions.

        :returns: pair of lists X and Y. Initial points and value of fitness\
                  function on them.
        """
        X = list()
        Y = list()
        if X_init is not None:
            if Y_init is not None:
                X = X_init
                Y = Y_init
                assert len(X) == len(Y)
            else:
                for x in X_init:
                    X.append(x)
                    Y.append(f(x))
        for _ in range(num_init - len(X)):
            x = self.randomize(variables, random_type)
            X.append(x)
            Y.append(f(x))
        return X, Y

    def optimize(self, f, variables, num_init, X_init=None, Y_init=None,
                 args=(), options={}, maxiter=None):
        raise NotImplementedError


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
