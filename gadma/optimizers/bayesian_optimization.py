import operator as op
import numpy as np
import copy
import time
import random
from collections import OrderedDict

from .gaussian_process import GPyGaussianProcess, SMACGaussianProcess
from .optimizer import ConstrainedOptimizer
from .global_optimizer import GlobalOptimizer, register_global_optimizer
from .optimizer_result import OptimizerResult
from ..utils import ContinuousVariable, WeightedMetaArray, get_correct_dtype
from ..utils import normalize, get_best_kernel

from .. import GPyOpt_available, GPyOpt
from .. import GPy_available, GPy
from .. import smac_available, smac, ConfigSpace
from .. import bayesmark_available

if smac_available:
    import skopt
    skopt.__version__  # somehow it fixes import errors in smac
    import skopt.learning.gaussian_process.kernels as kernels
    from smac.epm.gp_kernels import ConstantKernel, Matern, RBF
    from smac.epm.gp_kernels import WhiteKernel, HammingKernel
    from smac.optimizer.acquisition import LogEI, EI, PI, LCB
    from smac.scenario.scenario import Scenario
    from smac.epm.gaussian_process_mcmc import GaussianProcess
    from smac.epm.util_funcs import get_types
    from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior
    from ConfigSpace import ConfigurationSpace, Configuration
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter
    from ConfigSpace.hyperparameters import CategoricalHyperparameter
    from smac.facade.smac_bo_facade import SMAC4BO
    from smac.utils.constants import MAXINT
    from smac.tae import StatusType
    from smac.stats.stats import Stats
    from smac.optimizer.random_configuration_chooser import ChooserProb
    from smac.runhistory.runhistory import RunHistory
    from smac.runhistory.runhistory2epm import (
        RunHistory2EPM4Cost,
        RunHistory2EPM4LogScaledCost,
    )
    from smac.optimizer.ei_optimization import LocalAndSortedRandomSearch


def get_maxeval_for_bo(maxeval, maxiter):
    maxit = maxiter
    maxev = maxeval
    if maxit is None:
        if maxeval is not None:
            maxit = maxeval
        else:
            maxit = 100
    if maxev is None:
        maxev = maxit
    return min(maxit, maxev)


def choose_kernel_if_needed(
    optimizer,
    variables,
    X,
    Y,
    kernels=["exponential", "matern32", "matern52", "rbf"]
):
    # If needed we choose our kernel
    if optimizer.kernel_name.lower() == "auto":
        # Safeguard, just in case...
        Y = np.array(Y)
        if np.any(~np.isfinite(Y)):
            Y[~np.isfinite(Y)] = np.max(Y[np.isfinite(Y)])
        optimizer.kernel_name = get_best_kernel(
            optimizer=optimizer,
            variables=variables,
            X=X,
            Y=Y,
            mode="rassmusen",
            kernels=kernels,
        )
        message = f"Kernel was chosen automatikally: {optimizer.kernel_name}\n"
        return optimizer.kernel_name, message
    return optimizer.kernel_name, None


class GPyOptBayesianOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization
    """
    def __init__(self, kernel="Auto", ARD=True, acquisition_type='MPI',
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        if not GPy_available or not GPyOpt_available:
            raise ValueError("Install GPyOpt and GPy to use "
                             "Bayesian optimization.")
        self.kernel_name = kernel
        self.ARD = ARD
        self.acquisition_type = acquisition_type
        super(GPyOptBayesianOptimizer, self).__init__(
            random_type=random_type,
            custom_rand_gen=custom_rand_gen,
            log_transform=log_transform,
            maximize=maximize
        )

    def _get_kernel_class(self):
        if self.kernel_name.lower() == "auto":
            return None
        kernel_name = self.kernel_name.lower().capitalize()
        if kernel_name == "Rbf":
            kernel_name = "RBF"
        return op.attrgetter(kernel_name)(GPy.kern)

    def get_kernel(self, config_space):
        kernel = self._get_kernel_class()(len(config_space), ARD=self.ARD)
        return kernel

    def get_config_space(self, variables):
        gpy_domain = []
        for var in variables:
            gpy_domain.append({'name': var.name,
                               'type': var.var_type,
                               'domain': var.domain})
        return gpy_domain

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):

        def get_x_repr(x):
            if isinstance(x, WeightedMetaArray):
                return x.str_as_list()
            return str(list(x))

        bo_obj = run_info.bo_obj
        if bo_obj is not None:
            bo_obj._compute_results()
        x_best = run_info.result.x
        y_best = run_info.result.y
        n_iter = run_info.result.n_iter

        print('====================== Iteration %05d ======================' %
              n_iter, file=stream)

        if n_iter == 0:
            print("Initial design:", file=stream)
        else:
            print("Got points:", file=stream)

        print("Fitness function\tParameters", file=stream)
        for x, y in zip(run_info.result.X_out, run_info.result.Y_out):
            print(f"{y}\t{get_x_repr(x)}", file=stream)

        if bo_obj is not None:
            print('\nCurrent state of the model:', file=stream)

            print(str(bo_obj.model), file=stream)
            print(bo_obj.model.model.kern.lengthscale, file=stream)

        if hasattr(run_info, "gp_train_times"):
            print("\nTime for GP training:", run_info.gp_train_times[n_iter],
                  file=stream)
        if hasattr(run_info, "gp_predict_times"):
            print("Time for GP prediction:",
                  run_info.gp_predict_times[n_iter],
                  file=stream)
        if hasattr(run_info, "acq_opt_times"):
            print("Time for acq. optim.:",
                  run_info.acq_opt_times[n_iter], file=stream)
        if hasattr(run_info, "eval_times"):
            print("Time of evaluation:",
                  run_info.eval_times[n_iter], file=stream)
        if hasattr(run_info, "iter_times"):
            print("Total time of iteration:",
                  run_info.iter_times[n_iter], file=stream)

        print('=============================================================',
              end="\n\n", file=stream)
        print('*************************************************************',
              file=stream)
        print('Current optimum: %0.3f' % y_best, file=stream)
        print(f'On parameters: {get_x_repr(x_best)}', file=stream)
        print('*************************************************************',
              file=stream)
        print("\n", file=stream)

    def _create_run_info(self):
        """
        Creates the initial run_info. It has the following fields:
        * `result` - empty :class:`gadma.optimizers.OptimizerResult` with\
          `n_iter`==-1.
        * `bo_obj` - Object of BO from GpyOpt.
        """
        run_info = super(GPyOptBayesianOptimizer, self)._create_run_info()
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
        super(GPyOptBayesianOptimizer, self).save(info, save_file)

    def load(self, save_file):
        run_info = super(GPyOptBayesianOptimizer, self).load(save_file)
        run_info.result.X_out = list(run_info.result.X)
        run_info.result.Y_out = list(run_info.result.Y)
        return run_info

    def get_model(self, config_space):
        from GPyOpt.models import GPModel
        kernel = self.get_kernel(config_space)
        gp_model = GPModel(
            kernel=kernel,
            noise_var=None,
            exact_feval=True,
            optimizer="lbfgs",
            max_iters=1000,
            optimize_restarts=5,
            sparse=False,
            num_inducing=10,
            verbose=False,
            ARD=self.ARD
        )
        return GPyGaussianProcess(gp_model)

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        from GPyOpt.methods import BayesianOptimization
        from GPyOpt.core.task.objective import SingleObjective

        maxeval = get_maxeval_for_bo(maxeval, maxiter)

        kernel_name, message = choose_kernel_if_needed(
            self, variables, X_init, Y_init
        )
        self.kernel_name = kernel_name

        ndim = len(variables)

        if len(Y_init) > 0:
            x_best = X_init[0]
            y_best = Y_init[0]
            iter_callback(x_best, y_best, X_init, Y_init, message=message)

        gpy_domain = self.get_config_space(variables)
        kernel = self.get_kernel(config_space=gpy_domain)

        Y_init = np.array(Y_init).reshape(len(Y_init), -1)
        X_init = np.array(X_init, dtype=float)

        gpy_model = self.get_model(config_space=gpy_domain).gp_model

        bo = BayesianOptimization(f=f,
                                  domain=gpy_domain,
                                  model_type='user-specified',
                                  model=gpy_model,
                                  acquisition_type=self.acquisition_type,
                                  kernel=kernel,
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
        bo.objective = SingleObjective(
            bo.f, bo.batch_size, bo.objective_name)

        bo.run_optimization(max_iter=maxeval-len(X_init), eps=0,
                            verbosity=False)

        result = OptimizerResult.from_GPyOpt_OptimizerResult(bo)
        self.run_info.result.success = True
        self.run_info.status = result.status
        self.run_info.message = result.message
        return self.run_info.result


if GPyOpt_available:
    register_global_optimizer(
        'GPyOpt_Bayesian_optimization',
        GPyOptBayesianOptimizer
    )


class SMACSquirrelOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization with SMAC from Black Box challenge.
    """
    def __init__(self, n_suggestions=4,
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        if not smac_available:
            raise ValueError("Install SMAC to use it in SMAC squirrel "
                             "optimization")
        if not bayesmark_available:
            raise ValueError("Install bayesmark to use it in SMAC squirrel "
                             "optimization")
        self.n_suggestions = n_suggestions
        super(SMACSquirrelOptimizer, self).__init__(
            random_type=random_type,
            custom_rand_gen=custom_rand_gen,
            log_transform=log_transform,
            maximize=maximize
        )

    def get_configs(self, variables):
        api_config = OrderedDict()
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
        GPyOptBayesianOptimizer._write_report_to_stream(
            variables,
            run_info,
            stream
        )

    def valid_restore_file(self, save_file):
        try:
            run_info = self.load(save_file)
        except Exception:
            return False
        if (not isinstance(run_info.result.n_eval, int) or
                not isinstance(run_info.result.n_iter, int)):
            return False
        return True

    def load(self, save_file):
        run_info = super(SMACSquirrelOptimizer, self).load(save_file)
        run_info.result.X_out = list(run_info.result.X)
        run_info.result.Y_out = list(run_info.result.Y)
        return run_info

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        from .smac_optim import SMAC4EPMOpimizer

        maxiter = get_maxeval_for_bo(maxeval, maxiter)
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
                for ind, (var, par) in enumerate(zip(variables, x)):
                    if isinstance(variables[ind], ContinuousVariable):
                        par = float(par)
                    x_guess[i][var.name] = par
            return x_guess

        X_init = np.array(X_init, dtype=get_correct_dtype(x_best))
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


if smac_available and bayesmark_available:
    register_global_optimizer(
        'SMAC_squirrel_optimization',
        SMACSquirrelOptimizer
    )


class SMACBayesianOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization with SMAC library.

    :param kernel: Name of Gaussian process kernel.
    :param acquisition_type: Name of acquisition function.

    Other parameters are standard for :class:`gadma.GlobalOptimizer`.
    """
    def __init__(self, kernel="Auto", acquisition_type='MPI',
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        if not smac_available:
            raise ValueError("Install SMAC to use it in Bayesian optimization")
        self.kernel_name = kernel
        self.acquisition_type = acquisition_type
        super(SMACBayesianOptimizer, self).__init__(
            random_type=random_type,
            custom_rand_gen=custom_rand_gen,
            log_transform=log_transform,
            maximize=maximize
        )

    @property
    def kernel_name(self):
        return self._kernel_name

    @kernel_name.setter
    def kernel_name(self, value):
        self._kernel_name = value
        self._get_kernel_class_and_nu()

    @property
    def acquisition_type(self):
        return self._acquisition_type

    @acquisition_type.setter
    def acquisition_type(self, value):
        self._acquisition_type = value
        self.get_acquisition_function_class()

    def _get_kernel_class_and_nu(self):
        kernel_cls = Matern  # The most common case
        if self.kernel_name.lower() == "exponential":
            nu = 0.5
        elif self.kernel_name.lower() == "matern32":
            nu = 1.5
        elif self.kernel_name.lower() == "matern52":
            nu = 2.5
        elif self.kernel_name.lower() == "rbf":
            kernel_cls = RBF
            nu = None
        elif self.kernel_name.lower() == "auto":
            # In that case we should choose kernel from initial design by ll
            kernel_cls = None
            nu = None
        else:
            raise ValueError(f"Unknown name of kernel: {self.kernel_name}.")
        return kernel_cls, nu

    def _get_random_state(self):
        return np.random.RandomState(seed=0)

    def get_kernel(self, config_space):
        """
        Code is very similar to those from :class:`smac.SMAC4BO`.
        Length scale were chosen from SMAC where they were obtained by
        hyperparameter optimization made in https://arxiv.org/abs/1908.06674.

        :params config_space: SMAC configuration space with parameters.
        :type config_space: :class:`smac.configspace.ConfigurationSpace`
        """
        # First of all get type of kernel
        kernel_cls, nu = self._get_kernel_class_and_nu()
        # get types and bounds for config_space
        types, bounds = get_types(config_space, instance_features=None)
        # get random state for priors
        rng = self._get_random_state()

        # create kernel to hold variance
        cov_amp = smac.epm.gp_kernels.ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
        )

        # Understand information about parameters
        cont_dims = np.where(np.array(types) == 0)[0]
        cat_dims = np.where(np.array(types) != 0)[0]

        # bounds for length scale from https://arxiv.org/abs/1908.06674
        lslims = (np.exp(-6.754111155189306), np.exp(0.0858637988771976))

        # create kernel for continuous parameters
        if len(cont_dims) > 0:
            exp_kwargs = {
                "length_scale": np.ones([len(cont_dims)]),
                "length_scale_bounds": [lslims for _ in range(len(cont_dims))],
                "operate_on": cont_dims,
                "prior": None,
                "has_conditions": False,
            }
            if nu is not None:
                exp_kwargs["nu"] = nu
            exp_kernel = kernel_cls(**exp_kwargs)

        # kernel for categorical parameters
        if len(cat_dims) > 0:
            ham_kernel = smac.epm.gp_kernels.HammingKernel(
                length_scale=np.ones([len(cat_dims)]),
                length_scale_bounds=[lslims for _ in range(len(cat_dims))],
                operate_on=cat_dims,
            )

        # create noise kernel
        noise_kernel = smac.epm.gp_kernels.WhiteKernel(
            noise_level=1e-8,
            noise_level_bounds=(np.exp(-25), np.exp(2)),
            prior=HorseshoePrior(scale=0.1, rng=rng),
        )

        # create final kernel as combination
        if len(cont_dims) > 0 and len(cat_dims) > 0:
            # both
            kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
        elif len(cont_dims) > 0 and len(cat_dims) == 0:
            # only continuous parameters
            kernel = cov_amp * exp_kernel + noise_kernel
        elif len(cont_dims) == 0 and len(cat_dims) > 0:
            # only categorical parameters
            kernel = cov_amp * ham_kernel + noise_kernel

        return kernel

    def get_model(self, config_space):
        kernel = self.get_kernel(config_space)
        types, bounds = get_types(config_space, instance_features=None)
        smac_gp = GaussianProcess(
            configspace=config_space,
            types=types,
            bounds=bounds,
            seed=self._get_random_state().randint(MAXINT),
            kernel=kernel,
            normalize_y=False,
        )
        return SMACGaussianProcess(smac_gp)

    def get_acquisition_function_class(self):
        if self.acquisition_type.lower() == "logei":
            return LogEI
        elif self.acquisition_type.lower() == "ei":
            return EI
        elif self.acquisition_type.lower() in ["pi", "mpi"]:
            return PI
        elif self.acquisition_type.lower() == "lcb":
            return LCB
        else:
            raise ValueError("Unknown name of acquisition function: "
                             f"{self.acquisition_type}")

    def get_acquisition_function(self, model):
        assert isinstance(model, SMACGaussianProcess)
        return self.get_acquisition_function_class()(model=model.gp_model)

    def get_acquisition_function_optimizer(self,
                                           config_space,
                                           acquisition_function):
        return LocalAndSortedRandomSearch(
            config_space=config_space,
            rng=self._get_random_state(),
            max_steps=5,
            n_steps_plateau_walk=5,
            n_sls_iterations=5,
            acquisition_function=acquisition_function,
        )

    def get_config_space(self, variables):
        api_config = OrderedDict()
        cs = ConfigSpace.ConfigurationSpace()
        hp_list = []
        for var in variables:
            if isinstance(var, ContinuousVariable):
                hp_list.append(UniformFloatHyperparameter(name=var.name,
                                                          lower=var.domain[0],
                                                          upper=var.domain[1],
                                                          log=False))
            else:
                hp_list.append(CategoricalHyperparameter(name=var.name,
                                                         choices=var.domain))
        cs.add_hyperparameters(hp_list)
        return cs

    def get_scenario(self, maxeval, config_space):
        scenario = Scenario({
            "run_obj": "quality",  # we optimize quality
            "runcount-limit": maxeval,  # max. number of function evaluations;
            "cs": config_space,  # configuration space
            "deterministic": "true",
            "limit_resources": False,
        })
        if self.acquisition_type.lower() == "logei":
            scenario.transform_y = "LOG"
        return scenario

    def get_runhistory2epm(self, scenario):
        if self.acquisition_type.lower() == "logei":
            rh2epm_cls = RunHistory2EPM4LogScaledCost
        else:
            rh2epm_cls = RunHistory2EPM4Cost

        return rh2epm_cls(
            scenario=scenario,
            num_params=len(scenario.cs.get_hyperparameters()),
            success_states=[StatusType.SUCCESS],
            impute_censored_data=False,
            scale_perc=5
        )

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        run_info.bo_obj = None
        GPyOptBayesianOptimizer._write_report_to_stream(
            variables=variables,
            run_info=run_info,
            stream=stream
        )

    def valid_restore_file(self, save_file):
        try:
            run_info = self.load(save_file)
        except Exception:
            return False
        if (not isinstance(run_info.result.n_eval, int) or
                not isinstance(run_info.result.n_iter, int)):
            return False
        return True

    def _create_run_info(self):
        run_info = super(SMACBayesianOptimizer, self)._create_run_info()
        run_info.gp_train_times = []
        run_info.gp_predict_times = []
        run_info.acq_opt_times = []
        run_info.eval_times = []
        run_info.iter_times = []
        return run_info

    def _update_run_info(self, run_info, x_best, y_best, X, Y,
                         n_eval, gp_train_time=None, gp_predict_time=None,
                         acq_opt_time=None, eval_time=None, iter_time=None):
        super(SMACBayesianOptimizer, self)._update_run_info(
            run_info=run_info,
            x_best=x_best,
            y_best=y_best,
            X=X,
            Y=Y,
            n_eval=n_eval
        )
        run_info.gp_train_times.append(gp_train_time)
        run_info.gp_predict_times.append(gp_predict_time)
        run_info.acq_opt_times.append(acq_opt_time)
        run_info.eval_times.append(eval_time)
        run_info.iter_times.append(iter_time)
        return run_info

    def load(self, save_file):
        run_info = super(SMACBayesianOptimizer, self).load(save_file)
        run_info.result.X_out = list(run_info.result.X)
        run_info.result.Y_out = list(run_info.result.Y)
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

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        maxeval = get_maxeval_for_bo(maxeval, maxiter)

        # Safeguard, when Y_init is bad
        Y_init = np.array(Y_init)
        if np.all(~np.isfinite(Y_init)):
            iter_callback(X_init[0], Y_init[0], X_init, Y_init)
            return self.run_info.result
        if np.any(~np.isfinite(Y_init)):
            Y_init[~np.isfinite(Y_init)] = np.max(Y_init[np.isfinite(Y_init)])

        kernel_name, message = choose_kernel_if_needed(
            self, variables, X_init, Y_init
        )
        self.kernel_name = kernel_name

        iter_callback(X_init[0], Y_init[0], X_init, Y_init, message=message)

        # Get config space
        config_space = self.get_config_space(variables)
        # get scenario, runhistory and stats
        scenario = self.get_scenario(maxeval, config_space)
        runhistory = RunHistory()
        stats = Stats(scenario)
        # for acq function optimizer
        rnd_chooser = ChooserProb(rng=self._get_random_state(), prob=0.0)
        # get class to get valid train data from run history
        rh2epm = self.get_runhistory2epm(scenario)

        # we will add configs to run history by using the following function
        def add_to_runhistory(config, cost):
            runhistory.add(
                config=config,
                cost=cost,
                time=0,
                status=StatusType.SUCCESS
            )

        # create gp and other stuff
        model = self.get_model(config_space)
        acq_fun = self.get_acquisition_function(model)
        acq_fun_opt = self.get_acquisition_function_optimizer(
            config_space,
            acq_fun
        )

        # transform our X_init for valid configurations
        # we create random valid configs and then fill them with our values
        X_init_configs = config_space.sample_configuration(len(X_init))
        for x in X_init:
            for i, x in enumerate(X_init):
                for ind, (var, par) in enumerate(zip(variables, x)):
                    if isinstance(variables[ind], ContinuousVariable):
                        par = float(par)
                    X_init_configs[i][var.name] = par

        # add our initial design to run history
        for x, y in zip(X_init_configs, Y_init):
            add_to_runhistory(x, y)

        # begin Bayesian optimization
        while self.run_info.result.n_eval < maxeval and \
                (maxiter is None or
                 self.run_info.result.n_iter < maxiter):
            total_t_start = time.time()

            X, y = rh2epm.transform(runhistory)

            # If all are not finite then we return nothing
            is_bad = np.array([not _x for _x in np.isfinite(y)])
            if np.all(is_bad):
                message = "All fitness function values are infinite."
                self.run_info.result.message = message
                return self.run_info.result

            # Safeguard, just in case...
            if np.any(is_bad):
                y[is_bad] = np.max(y[np.isfinite(y)])

            t_start = time.time()
            model.train(X, y)
            gp_train_time = time.time() - t_start

            t_start = time.time()
            predictions = model.predict(X)[0]
            best_index = np.argmin(predictions)
            best_observation = y[best_index]
            x_best_array = X[best_index]
            gp_predict_time = time.time() - t_start

            t_start = time.time()
            acq_fun.update(
                model=model.gp_model,
                eta=best_observation,
                incumbent_array=x_best_array,
                num_data=len(X),
                X=X,
            )
            new_config_iterator = acq_fun_opt.maximize(
                runhistory=runhistory,
                stats=stats,
                num_points=10000,
                random_configuration_chooser=rnd_chooser,
            )
            accept = False
            for next_config in new_config_iterator:
                if next_config in runhistory.get_all_configs():
                    continue
                else:
                    accept = True
                    break
            assert accept
            acq_opt_time = time.time() - t_start

            t_start = time.time()
            x = [next_config[var.name] for var in variables]
            cost = f(x)
            eval_time = time.time() - t_start
            add_to_runhistory(next_config, cost)

            total_iter_time = time.time() - total_t_start
            update_kwargs = {"gp_train_time": gp_train_time,
                             "gp_predict_time": gp_predict_time,
                             "acq_opt_time": acq_opt_time,
                             "eval_time": eval_time,
                             "iter_time": total_iter_time}
            iter_callback(x, cost, [x], [cost], **update_kwargs)

        return self.run_info.result


if smac_available:
    register_global_optimizer("SMAC_BO_optimization", SMACBayesianOptimizer)


class SMACBOKernelCombination(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization with 4 combinations of Gaussian Processes
    using SMAC library.

    Four different combinations are used on each iteration: Matern52 and RBF
    as kernels of GP; and MPI and logEI as acquisition functions.
    """
    def __init__(self, kernels_to_choose=["matern52", "rbf"],
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        if not smac_available:
            raise ValueError("Install SMAC to use it in Bayesian optimization")
        super(SMACBOKernelCombination, self).__init__(
            random_type=random_type,
            custom_rand_gen=custom_rand_gen,
            log_transform=log_transform,
            maximize=maximize
        )
        self.kernels_to_choose = kernels_to_choose

    def _create_run_info(self):
        run_info = super(SMACBOKernelCombination, self)._create_run_info()
        run_info.gp_train_times = []
        run_info.gp_predict_times = []
        run_info.acq_opt_times = []
        run_info.eval_times = []
        run_info.iter_times = []
        run_info.kernel1 = None
        run_info.kernel2_log = None
        return run_info

    def _update_run_info(self, run_info, x_best, y_best, X, Y,
                         n_eval, gp_train_time=None, gp_predict_time=None,
                         acq_opt_time=None, eval_time=None, iter_time=None):
        super(SMACBOKernelCombination, self)._update_run_info(
            run_info=run_info,
            x_best=x_best,
            y_best=y_best,
            X=X,
            Y=Y,
            n_eval=n_eval
        )
        run_info.gp_train_times.append(gp_train_time)
        run_info.gp_predict_times.append(gp_predict_time)
        run_info.acq_opt_times.append(acq_opt_time)
        run_info.eval_times.append(eval_time)
        run_info.iter_times.append(iter_time)
        return run_info

    def load(self, save_file):
        run_info = super(SMACBOKernelCombination, self).load(save_file)
        run_info.result.X_out = list(run_info.result.X)
        run_info.result.Y_out = list(run_info.result.Y)
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

    def _create_gp_model(self, config_space, kernel_name):
        opt = SMACBayesianOptimizer(kernel=kernel_name)
        return opt.get_model(config_space=config_space)

    def _create_acquisition_function(self, model, acquisition_name):
        assert acquisition_name in ["PI", "logEI"]
        opt = SMACBayesianOptimizer(acquisition_type=acquisition_name)
        return opt.get_acquisition_function(model=model)

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        run_info.bo_obj = None
        GPyOptBayesianOptimizer._write_report_to_stream(
            variables=variables,
            run_info=run_info,
            stream=stream
        )

    def do_gp_optimization(self):
        """
        We optimize gp:
        1) every iteration for first 50 iterations
        2) every 5th iteration for next 50 iterations
        3) every 10th iteration for the last iterations
        """
        n_iter = self.run_info.result.n_iter + 1  # this is correct
        if n_iter <= 100:
            return True
        if n_iter <= 200:
            return (n_iter % 5) == 0
        return (n_iter % 10) == 0

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        maxeval = get_maxeval_for_bo(maxeval, maxiter)

        # Create help optimizer with usual SMAC to call the same functions
        help_opt = SMACBayesianOptimizer(kernel="Auto")
        help_opt_log = SMACBayesianOptimizer(
            kernel="Auto", acquisition_type="logEI"
        )

        kernel1_restored = False
        if self.run_info.kernel1 is None:
            kernel_name1, message1 = choose_kernel_if_needed(
                optimizer=help_opt,
                variables=variables,
                X=X_init,
                Y=Y_init,
                kernels=self.kernels_to_choose
            )
            self.run_info.kernel1 = kernel_name1
        else:
            kernel_name1 = self.run_info.kernel1
            message1 = f"Kernel was restored: {kernel_name1}\n"
            kernel1_restored = True
        help_opt.kernel_name = self.run_info.kernel1

        kernel2_restored = False
        if self.run_info.kernel2_log is None:
            kernel_name2, message2 = choose_kernel_if_needed(
                optimizer=help_opt_log,
                variables=variables,
                X=X_init,
                Y=Y_init,
                kernels=self.kernels_to_choose
            )
            self.run_info.kernel2_log = kernel_name2
        else:
            kernel_name2 = self.run_info.kernel2_log
            message2 = f"Kernel was restored: {kernel_name2}\n"
            kernel2_restored = True
        help_opt_log.kernel_name = self.run_info.kernel2_log

        message = f"For usual Y:\n{message1}For log_transformed Y:\n{message2}"

        x_best = X_init[0]
        y_best = Y_init[0]
        iter_callback(x_best, y_best, X_init, Y_init, message=message)

        # Get config space
        config_space = help_opt.get_config_space(variables)
        # get scenario, runhistory and stats
        scenario = help_opt.get_scenario(maxeval, config_space)
        runhistory = RunHistory()
        stats = Stats(scenario)
        # for acq function optimizer
        rnd_chooser = ChooserProb(rng=help_opt._get_random_state(), prob=0.0)

        # get classes to get valid train data from run history
        rh2epm_no_transform = help_opt.get_runhistory2epm(scenario)
        rh2epm_log = help_opt_log.get_runhistory2epm(scenario)
        acq2rh2epm = {"PI": rh2epm_no_transform, "logEI": rh2epm_log}

        # we will add configs to run history by using the following function
        def add_to_runhistory(config, cost):
            runhistory.add(
                config=config,
                cost=cost,
                time=0,
                status=StatusType.SUCCESS
            )

        combinations = []
        for model_name, acq_name in [(kernel_name1, "PI"),
                                     (kernel_name2, "logEI")]:
            gp = self._create_gp_model(config_space, model_name)
            acq = self._create_acquisition_function(
                model=gp,
                acquisition_name=acq_name
            )
            acq_opt = help_opt.get_acquisition_function_optimizer(
                config_space,
                acq
            )
            mark = f"{model_name}_{acq_name}"
            combinations.append(
                (mark, gp, acq, acq_opt, acq2rh2epm[acq_name])
            )

        # transform our X_init for valid configurations
        # we create random valid configs and then fill them with our values
        X_init_configs = config_space.sample_configuration(len(X_init))
        for x in X_init:
            for i, x in enumerate(X_init):
                for ind, (var, par) in enumerate(zip(variables, x)):
                    if isinstance(variables[ind], ContinuousVariable):
                        par = float(par)
                    X_init_configs[i][var.name] = par

        # add our initial design to run history
        for x, y in zip(X_init_configs, Y_init):
            add_to_runhistory(x, y)

        # begin Bayesian optimization
        while self.run_info.result.n_eval < maxeval or \
                (maxiter is not None and
                 self.run_info.result.n_iter * 4 < maxiter):
            # we miss some optimizations but the case of restore is special
            do_gp_optim = self.do_gp_optimization()

            total_t_start = time.time()

            gp_train_time = 0
            gp_predict_time = 0
            acq_opt_time = 0
            eval_time = 0
            iter_configs = []
            iter_X = []
            iter_y = []

            # shuffle our list in-place
            # random.shuffle(combinations)
            kernel_ind = np.random.choice([0, 1])
            mark, gp, acq, acq_opt, rh2epm = combinations[kernel_ind]
            # for mark, gp, acq, acq_opt, rh2epm in combinations:
            X, y = rh2epm.transform(runhistory)

            # If the kernel was restored we should optimize GP
            # jn the first iteration.
            if kernel1_restored and kernel_ind == 0:
                do_gp_optim = True
                kernel1_restored = False
            if kernel2_restored and kernel_ind == 1:
                do_gp_optim = True
                kernel2_restored = False

            message = f"GP was optimized: {do_gp_optim}\n"
            message += f"Current number of points: {len(X)}"

            # If all are not finite then we return nothing
            is_bad = np.array([not _x for _x in np.isfinite(y)])
            if np.all(is_bad):
                message = "All fitness function values are infinite."
                self.run_info.result.message = message
                return self.run_info.result

            # Safeguard, just in case...
            if np.any(is_bad):
                y[is_bad] = np.max(y[np.isfinite(y)])

            t_start = time.time()
            gp.train(X, y, optimize=do_gp_optim)
            gp_train_time += time.time() - t_start

            t_start = time.time()
            # we do not care what model is used here
            predictions = gp.predict(X)[0]
            best_index = np.argmin(predictions)
            best_observation = y[best_index]
            x_best_array = X[best_index]
            gp_predict_time += time.time() - t_start

            t_start = time.time()
            acq.update(
                model=gp.gp_model,
                eta=best_observation,
                incumbent_array=x_best_array,
                num_data=len(X),
                X=X,
            )
            new_config_iterator = acq_opt.maximize(
                runhistory=runhistory,
                stats=stats,
                num_points=10000,
                random_configuration_chooser=rnd_chooser,
            )
            accept = False
            for next_config in new_config_iterator:
                if next_config in runhistory.get_all_configs():
                    continue
                else:
                    accept = True
                    break
            assert accept
            acq_opt_time += time.time() - t_start

            t_eval = time.time()
            x = [next_config[var.name] for var in variables]
            cost = f(x)
            eval_time += time.time() - t_eval

            iter_configs.append((next_config, cost))

            x = WeightedMetaArray(x)
            x.metadata = mark
            iter_X.append(x)
            iter_y.append(cost)
            if cost < y_best:
                x_best = x
                y_best = cost

            for config, cost in iter_configs:
                if not np.isfinite(cost):
                    cost = np.max(y)
                add_to_runhistory(config, cost)

            total_iter_time = time.time() - total_t_start
            update_kwargs = {"gp_train_time": gp_train_time,
                             "gp_predict_time": gp_predict_time,
                             "acq_opt_time": acq_opt_time,
                             "eval_time": eval_time,
                             "iter_time": total_iter_time,
                             "message": message}
            iter_callback(x_best, y_best, iter_X, iter_y, **update_kwargs)

        return self.run_info.result


if smac_available:
    register_global_optimizer("SMAC_BO_combination", SMACBOKernelCombination)
