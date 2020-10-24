import operator as op
from functools import partial, wraps
import numpy as np
import copy
import pickle
import sys
import time

import logging
from .optimizer import ConstrainedOptimizer
from .global_optimizer import GlobalOptimizer, register_global_optimizer
from .optimizer_result import OptimizerResult
from ..utils import sort_by_other_list, choose_by_weight, eval_wrapper
from ..utils import ensure_file_existence, fix_args

from .. import GPyOpt
from .. import GPy

logger = logging.getLogger(__name__)


class BayesianOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization
    """
    def __init__(self, kernel="Matern52", ARD=True, acquisition_type='MPI',
                 log_transform=False, maximize=False):
        self.kernel_name = kernel
        self.ARD = ARD
        self.acquisition_type = acquisition_type
        super(BayesianOptimizer, self).__init__(log_transform, maximize)
        self.original_transform = copy.deepcopy(self.transform)
        self.original_inv_transform = copy.deepcopy(self.inv_transform)

        def new_transform(x):
            return self.gpyopt_transform(self.original_transform(x))

        def new_inv_transform(x):
            return self.gpyopt_inv_transform(self.original_inv_transform(x))
        self.transform = new_transform
        self.inv_transform = new_inv_transform

    def gpyopt_transform(self, x):
        return [x]

    def gpyopt_inv_transform(self, x):
        return x[0]

    def _get_kernel_class(self):
        return op.attrgetter(self.kernel_name)(GPy.kern.src.stationary)

    def get_kernel(self, ndim):
        kernel = self._get_kernel_class()(ndim, ARD=self.ARD)
        return kernel

    def get_domain(self, variables):
        gpy_domain = []
        for var in variables:
            gpy_domain.append({'name': var.name,
                               'type': var.var_type,
                               'domain': self.original_transform(var.domain)})
        return gpy_domain

#    def evaluate(self, f, x, args=(), linear_constrain=None):
#        return [super(BayesianOptimizer, self).evaluate(f, x[0], args,
#                                                        linear_constrain)]

    def _concatenate_f_and_callback(self, f, callback):
        @wraps(f)
        def concat_wrapper(x):
            y = f(x)
            if callback is not None:
                callback(self.inv_transform(x), y)
            return [y]
        return concat_wrapper

#    def initial_design(self, f, variables, num_init, X_init=None, Y_init=None,
#                       random_type='resample', custom_rand_gen=None):
#        X = list()
#        if X_init is not None:
#            X = list(X_init)
#        for _ in range(num_init - len(X)):
#            x = [self.randomize(variables, random_type, custom_rand_gen)]
#            X.append(x)
#        Y = None
#        if Y_init is not None:
#            Y = np.array(Y).reshape(len(Y), -1)
#        X, Y = super(BayesianOptimizer, self).initial_design(f, variables,
#                                                             num_init, X, Y)
#        return [x[0] for x in X], Y

    def write_report(self, bo_obj, report_file, x, y):
        """
        Writes report about each iteration in file or stdout.

        :param bo_obj: Object of Bayesian Optimization.
        :type bo_obj: GPyOpt.methods.BayesianOptimization
        :param report_file: file to write report to. If None then to stdout.

        :param x: Current solution
        :param y: Current value of fitness function
        """
        bo_obj._compute_results()
        x_best = bo_obj.x_opt
        y_best = bo_obj.fx_opt
        n_iter = bo_obj.num_acquisitions

        if y < y_best:
            x_best = x
            y_best = y

        if report_file is not None:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout

        print('====================== Iteration %05d ======================' %
              n_iter, file=stream)
        print('Current state of the model:', file=stream)

        print(str(bo_obj.model), file=stream)
        print(bo_obj.model.model.kern.lengthscale, file=stream)
        print('=============================================================',
              file=stream)

        print('*************************************************************',
              file=stream)
        print('Current optimum: %0.3f' % y_best, file=stream)
        print('*************************************************************',
              file=stream)

        if report_file is not None:
            stream.close()

    def optimize(self, f, variables, args=(), num_init=10,
                 X_init=None, Y_init=None,
                 linear_constrain=None, maxiter=100, maxeval=100,
                 verbose=0, callback=None, report_file=None, eval_file=None,
                 save_file=None):
        r"""
        Return best values of `variables` that minimizes/maximizes
        the function `f`.

        :param f: function to minimize/maximize. The usage must be the
                  following: f(x, *args), where x is list of values.
        :param variables: list of variables (instances of
                          :class:`gadma.Variable` class) of the function.
        :param X_init: list of initial values.
        :param Y_init: value of function `f` on initial values from `X_init`.
        :param args: arguments of function `f`.
        :param maxiter: maximum number of genetic algorithm's generations.
        :param maxeval: maximum number of function evaluations.
        :param callback: callback to call after each generation.
                         It will be called as callback(x, y), where x, y -
                         best_solution of generation and its fitness.
        """
        from GPyOpt.methods import BayesianOptimization
        if maxiter is None:
            maxiter = 100
        # Create logging files
        if eval_file is not None:
            ensure_file_existence(eval_file)
        if verbose > 0 and report_file is not None:
            report_file = ensure_file_existence(report_file)
        if save_file is not None:
            ensure_file_existence(save_file)

        # Prepare function to use it.
        # Fix args and cache
        prepared_f = self.prepare_f_for_opt(f, args)
        # Wrap for automatic evaluation logging
        finally_wrapped_f = eval_wrapper(prepared_f, eval_file)

        f_in_opt = partial(self.evaluate, finally_wrapped_f)
        f_in_opt = fix_args(f_in_opt, (), linear_constrain)

        if callback is not None:
            callback = self.prepare_callback(callback)

        # Stuff for BO
        ndim = len(variables)
        if ndim == 0:
            x_best = []
            y_best = f_in_opt([x_best])
            return OptimizerResult(x=x_best, y=self.sign*y_best,
                                   success=True, status="0",
                                   message="Number of variables == 0",
                                   X=[x_best], Y=[y_best],
                                   n_eval=1, n_iter=1,
                                   X_out=[x_best], Y_out=[y_best])

        kernel = self.get_kernel(ndim)
        gpy_domain = self.get_domain(variables)

        # Initial design
        X, Y = self.initial_design(finally_wrapped_f, variables, num_init,
                                   X_init, Y_init)

        Y = np.array(Y).reshape(len(Y), -1)

        bo = BayesianOptimization(f=f_in_opt,
                                  domain=gpy_domain,
                                  model_type='GP',
                                  acquisition_type=self.acquisition_type,
                                  kernel=kernel,
                                  ARD=self.ARD,
                                  X=np.array(X, dtype=object),
                                  Y=np.array(Y),
                                  exact_feval=True,
                                  verbosity=True,
                                  )

        def union_callback(x, y):
            if verbose > 0:
                self.write_report(bo, report_file, x, y)
            if callback is not None:
                callback(x, y)

        f_in_opt = self._concatenate_f_and_callback(f_in_opt, union_callback)

        bo.f = bo._sign(f_in_opt)
        bo.objective = GPyOpt.core.task.objective.SingleObjective(
            bo.f, bo.batch_size, bo.objective_name)

        bo.run_optimization(max_iter=min(maxiter, maxeval)-len(X), eps=0,
                            verbosity=False)

        result = OptimizerResult.from_GPyOpt_OptimizerResult(bo)
        return result


register_global_optimizer('Bayesian_optimization', BayesianOptimizer)
