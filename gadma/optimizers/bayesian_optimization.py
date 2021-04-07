import operator as op
import numpy as np
import copy

from .optimizer import ConstrainedOptimizer
from .global_optimizer import GlobalOptimizer, register_global_optimizer
from .optimizer_result import OptimizerResult
from ..utils import ContinuousVariable, WeightedMetaArray

#from .. import GPyOpt
from .. import GPy
from .. import smac_available


class BayesianOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization
    """
    def __init__(self, kernel="Matern52", ARD=True, acquisition_type='MPI',
                 random_type='resample', custom_rand_gen=None,
                 use_gpyopt=False,
                 log_transform=False, maximize=False):
        self.kernel_name = kernel
        self.ARD = ARD
        self.acquisition_type = acquisition_type
        super(BayesianOptimizer, self).__init__(
            random_type=random_type,
            custom_rand_gen=custom_rand_gen,
            log_transform=log_transform,
            maximize=maximize
        )
        self.use_gpyopt = use_gpyopt

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
                               'domain': var.domain})
        return gpy_domain

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        bo_obj = run_info.bo_obj
        if bo_obj is not None:
            bo_obj._compute_results()
        x_best = run_info.result.x
        y_best = run_info.result.y
        n_iter = run_info.result.n_iter

        if n_iter > 0:
            print("\n", file=stream)
        print('====================== Iteration %05d ======================' %
              n_iter, file=stream)

        if bo_obj is None:
            print("Initial design:", file=stream)
        else:
            print("Got points:", file=stream)

        print("Fitness function\tParameters", file=stream)
        for x, y in zip(run_info.result.X_out, run_info.result.Y_out):
            print(f"{y}\t{x}", file=stream)

        if bo_obj is not None:
            print('\nCurrent state of the model:', file=stream)

            print(str(bo_obj.model), file=stream)
            print(bo_obj.model.model.kern.lengthscale, file=stream)

        print('=============================================================',
              end="\n\n", file=stream)
        print('*************************************************************',
              file=stream)
        print('Current optimum: %0.3f' % y_best, file=stream)
        print(f'On parameters: {x_best}', file=stream)
        print('*************************************************************',
              file=stream)

    def _create_run_info(self):
        """
        Creates the initial run_info. It has the following fields:
        * `result` - empty :class:`gadma.optimizers.OptimizerResult` with\
          `n_iter`==-1.
        * `bo_obj` - Object of BO from GpyOpt.
        """
        run_info = super(BayesianOptimizer, self)._create_run_info()
        run_info.bo_obj = None
        return run_info

    def valid_restore_file(self, save_file):
        try:
            run_info = self.load(save_file)
        except Exception:
            return False
        if (not isinstance(run_info.result.n_eval, int) or
                not isinstance(run_info.result.n_iter, int)):
            return False
        return True

    def save(self, run_info, save_file):
        # run_info.bo_obj could not be deepcopied and pickled so we ignore it
        info = self._create_run_info()
        info.result = copy.deepcopy(run_info.result)
        # also change X_out to be equal to X_total. For good restore
        info.result.X_out = info.result.X
        info.result.Y_out = info.result.Y
        super(BayesianOptimizer, self).save(info, save_file)

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        if self.use_gpyopt:
            import GPyOpt
            from GPyOpt.methods import BayesianOptimization
        else:
            import MyGPyOpt as GPyOpt
            from GPyOpt.methods import BayesianOptimization

        if maxiter is None:
            maxiter = 100
        if maxeval is None:
            maxeval = maxiter

        ndim = len(variables)

        if ndim == 0:
            x_best = []
            y_best = f([x_best])
            iter_callback(x_best, y_best, [x_best], [y_best])
            self.run_info.result.success = True
            self.run_info.result.status = 0
            self.run_info.result.message = "Number of variables == 0"
            return self.run_info.result

        if len(Y_init) > 0:
            x_best = X_init[0]
            y_best = Y_init[0]
            iter_callback(x_best, y_best, X_init, Y_init)

        kernel = self.get_kernel(ndim)
        gpy_domain = self.get_domain(variables)

        Y_init = np.array(Y_init).reshape(len(Y_init), -1)

        bo = BayesianOptimization(f=f,
                                  domain=gpy_domain,
                                  model_type='GP',
                                  acquisition_type=self.acquisition_type,
                                  kernel=kernel,
                                  ARD=self.ARD,
                                  X=np.array(X_init),
                                  Y=np.array(Y_init),
                                  exact_feval=True,
                                  verbosity=True,
                                  )
        bo.num_acquisitions = self.run_info.result.n_eval
        self.run_info.bo_obj = bo

        def f_in_gpyopt(X):
            Y = []
            x_best = self.transform(self.run_info.result.x)
            y_best = self.sign * self.run_info.result.y
            for x in X:
                y = f(x)
                if y_best is None or y < y_best:
                    x_best = x
                    y_best = y
                Y.append(y)
            iter_callback(x=x_best, y=y_best, X_iter=X, Y_iter=Y)
            return np.array(Y).reshape(len(Y), -1)

        bo.f = bo._sign(f_in_gpyopt)
        bo.objective = GPyOpt.core.task.objective.SingleObjective(
            bo.f, bo.batch_size, bo.objective_name)

        bo.run_optimization(max_iter=min(maxiter, maxeval)-len(X_init), eps=0,
                            verbosity=False)

        result = OptimizerResult.from_GPyOpt_OptimizerResult(bo)
        self.run_info.result.success = True
        self.run_info.status = result.status
        self.run_info.message = result.message
        return self.run_info.result


register_global_optimizer('Bayesian_optimization', BayesianOptimizer)


class SMACOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization with SMAC from Black Box challenge.
    """
    def __init__(self, n_suggestions=4,
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        self.n_suggestions = n_suggestions
        super(SMACOptimizer, self).__init__(
            random_type=random_type,
            custom_rand_gen=custom_rand_gen,
            log_transform=log_transform,
            maximize=maximize
        )

    def get_configs(self, variables):
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter
        from ConfigSpace.hyperparameters import CategoricalHyperparameter
        api_config = {}
        cs = ConfigurationSpace()
        hp_list = []
        for var in variables:
            if isinstance(var, ContinuousVariable):
                api_config[var.name] = {'type': 'real',
                                        'space': 'linear',
                                        'range': var.domain}
                hp_list.append(UniformFloatHyperparameter(name=var.name,
                                                          lower=var.domain[0],
                                                          upper=var.domain[1],
                                                          log=False))
            else:
                api_config[var.name] = {'type': 'cat',
                                        'values': var.domain}
                hp_list.append(CategoricalHyperparameter(name=var.name,
                                                         choices=var.domain))
        cs.add_hyperparameters(hp_list)
        return api_config, cs

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        run_info.bo_obj = None
        BayesianOptimizer._write_report_to_stream(variables, run_info, stream)

    def valid_restore_file(self, save_file):
        try:
            run_info = self.load(save_file)
        except Exception:
            return False
        if (not isinstance(run_info.result.n_eval, int) or
                not isinstance(run_info.result.n_iter, int)):
            return False
        return True

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        from .smac_optim import SMAC4EPMOpimizer

        if maxiter is None:
            maxiter = 100
        if maxeval is None:
            maxeval = (maxiter * self.n_suggestions +
                       self.run_info.result.n_eval)
        x_best = X_init[0]
        y_best = Y_init[0]
        iter_callback(x_best, y_best, X_init, Y_init)

        api, cs = self.get_configs(variables)

        opt = SMAC4EPMOpimizer(api_config=api,
                               config_space=cs,
                               parallel_setting='KB')

        def get_x_guess(X):
            x_guess = cs.sample_configuration(len(X))
            for i, x in enumerate(X):
                for var, par in zip(variables, x):
                    x_guess[i][var.name] = par
            return x_guess

        opt.observe(get_x_guess(X_init), np.array(Y_init))

        while (self.run_info.result.n_iter < maxiter and
                self.run_info.result.n_eval < maxeval):
            n_suggestions = min(self.n_suggestions,
                                maxeval - self.run_info.result.n_eval)
            X_returned = opt.suggest(n_suggestions=n_suggestions)
            X_iter = []
            for conf, info in X_returned:
                d = conf
                x = [d[var.name] for var in variables]
                x = WeightedMetaArray(x)
                x.metadata = info
                X_iter.append(x)
            X_guess = [el[0] for el in X_returned]
            Y_iter = [f(x) for x in X_iter]
            y = min(Y_iter, default=np.inf)
            if y < y_best:
                y_best = y
                x_best = X_iter[Y_iter.index(y)]
            if len(Y_iter) > 0:
                opt.observe(X_guess, np.array(Y_iter))
            iter_callback(x=x_best, y=y_best,
                          X_iter=X_iter, Y_iter=Y_iter)

        # report why we stop
        self.run_info.result.success = True
        if self.run_info.result.n_eval == maxeval:
            self.run_info.result.status = 1
            self.run_info.result.message = (f"Maximum number of evaluations "
                                            f"({maxeval}) achieved")
        if self.run_info.result.n_iter == maxeval:
            self.run_info.result.status = 2
            self.run_info.result.message = (f"Maximum number of iterations "
                                            f"({maxiter}) achieved")

        return self.run_info.result


if smac_available:
    register_global_optimizer('SMAC_optimization', SMACOptimizer)
