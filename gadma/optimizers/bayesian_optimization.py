import operator as op
import numpy as np
import copy

from .optimizer import ConstrainedOptimizer
from .global_optimizer import GlobalOptimizer, register_global_optimizer
from .optimizer_result import OptimizerResult
from ..utils import ContinuousVariable, WeightedMetaArray, get_correct_dtype

from .. import GPyOpt_available, GPyOpt
from .. import GPy_available, GPy
from .. import smac_available, smac, ConfigSpace


def get_maxeval_for_bo(maxeval, maxiter):
    if maxiter is None:
        if maxeval is not None:
            maxiter = maxeval
        else:
            maxiter = 100
    if maxeval is None:
        maxeval = maxiter
    return min(maxiter, maxeval)


class GPyOptBayesianOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization
    """
    def __init__(self, kernel="Matern52", ARD=True, acquisition_type='MPI',
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

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        from GPyOpt.methods import BayesianOptimization
        from GPyOpt.core.task.objective import SingleObjective

        maxeval = get_maxeval_for_bo(maxeval, maxiter)

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
        X_init = np.array(X_init, dtype=float)

        bo = BayesianOptimization(f=f,
                                  domain=gpy_domain,
                                  model_type='GP',
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


register_global_optimizer('GPyOpt_Bayesian_optimization', GPyOptBayesianOptimizer)


class SMACSquirellOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization with SMAC from Black Box challenge.
    """
    def __init__(self, n_suggestions=4,
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        self.n_suggestions = n_suggestions
        super(SMACSquirellOptimizer, self).__init__(
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

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        from .smac_optim import SMAC4EPMOpimizer

        if maxiter is None:
            if maxeval is not None:
                maxiter = maxeval
            else:
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


if smac_available:
    register_global_optimizer('SMAC_squirell_optimization', SMACSquirellOptimizer)


class SMACBayesianOptimizer(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Bayesian optimization with SMAC from Black Box challenge.
    """
    def __init__(self, kernel="Matern52", ARD=True, acquisition_type='MPI',
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        if not smac_available:
            raise ValueError("Install SMAC to use it in Bayesian optimization")
        self.kernel_name = kernel
        self.ARD = ARD
        self.acquisition_type = acquisition_type
        super(SMACSquirellOptimizer, self).__init__(
            random_type=random_type,
            custom_rand_gen=custom_rand_gen,
            log_transform=log_transform,
            maximize=maximize
        )
        # checks
        self._get_kernel_class_and_nu()
        self.get_acquisition_function_class()

    def _get_kernel_class_and_nu(self):
        kernel_cls = smac.epm.gp_kernels.Matern  # The most common case
        if self.kernel_name.lower() == "exponential":
            nu = 0.5
        elif self.kernel_name.lower() == "matern32":
            nu = 1.5
        elif self.kernel_name.lower() == "matern52":
            nu = 2.5
        elif self.kernel_name.lower() == "rbf":
            kernel_cls = smac.epm.gp_kernels.RBF
            nu = None
        else:
            raise ValueError(f"Unknown name of kernel: {self.kernel_name}.")
        return kernel_cls, nu

    def get_kernel_for_configuration_space(self, configuration_space):
        """
        Code is very similar to those from :class:`smac.SMAC4BO`.
        Length scale were chosen from SMAC where they were obtained by
        hyperparameter optimization made in https://arxiv.org/abs/1908.06674.

        :params configuration_space: SMAC configuration space with parameters.
        :type configuration_space: :class:`smac.configspace.ConfigurationSpace`
        """
        # First of all get type of kernel
        kernel_cls, nu = self._get_kernel_class_and_nu()

        types, bounds = smac.epm.util_funcs.get_types(configuration_space,
                                                      instance_features=None)

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
        bounds = (np.exp(-6.754111155189306), np.exp(0.0858637988771976))

        # create kernel for continuous parameters
        if len(cont_dims) > 0:
            kwargs = {
                "length_scale": np.ones([len(cont_dims)]),
                "length_scale_bounds": [bounds for _ in range(len(cont_dims))],
                "operate_on": cont_dims,
                "prior": None,
                "has_conditions": False,
            }
            if nu is not None:
                kwargs["nu"] = nu
            exp_kernel = kernel_cls(**kwargs)

        # kernel for categorical parameters
        if len(cat_dims) > 0:
            ham_kernel = smac.epm.gp_kernels.HammingKernel(
                length_scale=np.ones([len(cat_dims)]),
                length_scale_bounds=[bounds for _ in range(len(cat_dims))],
                operate_on=cat_dims,
            )

        assert (len(cont_dims) + len(cat_dims)) == len(scenario.cs.get_hyperparameters())

        noise_kernel = smac.epm.gp_kernels.WhiteKernel(
            noise_level=1e-8,
            noise_level_bounds=(np.exp(-25), np.exp(2)),
            prior=HorseshoePrior(scale=0.1, rng=rng),
        )

        if len(cont_dims) > 0 and len(cat_dims) > 0:
            # both
            kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
        elif len(cont_dims) > 0 and len(cat_dims) == 0:
            # only cont
            kernel = cov_amp * exp_kernel + noise_kernel
        elif len(cont_dims) == 0 and len(cat_dims) > 0:
            # only cont
            kernel = cov_amp * ham_kernel + noise_kernel

        return kernel

    def get_acquisition_function_class(self):
        if self.acquisition_type.lower() == "logei":
            return smac.optimizer.acquisition.LogEI
        elif self.acquisition_type.lower() == "ei":
            return smac.optimizer.acquisition.EI
        elif self.acquisition_type.lower() in ["pi", "mpi"]:
            return smac.optimizer.acquisition.PI
        elif self.acquisition_type.lower() == "lcb":
            return smac.optimizer.acquisition.LCB
        else:
            raise ValueError("Unknown name of acquisition function: "
                             f"{self.acquisition_type}")


    def get_smac_configuration_space(self, variables):
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter
        from ConfigSpace.hyperparameters import CategoricalHyperparameter
        api_config = {}
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

    def get_smac_scenario(self, maxeval, configuration_space):
        scenario = Scenario({
            "run_obj": "quality",  # we optimize quality
            "runcount-limit": maxeval,  # max. number of function evaluations;
            "cs": configuration_space,  # configuration space
            "deterministic": "true"
        })
        if self.acquisition_type.lower() == "logei":
            scenario.transform_y = "LOG"
        return scenario

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

        maxeval = get_maxeval_for_bo(maxeval, maxxiter)

        configuration_space = self.get_smac_configuration_space(variables)

        scenario = self.get_smac_scenario(maxeval, configuration_space)

        model = smac.epm.gaussian_process_mcmc.GaussianProcess
        kernel = self.get_kernel_for_configuration_space(configuration_space)
        model_kwargs = {
            "kernel": kernel,
            "normalize_y": True,
            "seed": rng.randint(0, 2 ** 20),
        }

        acquisition_function = self.get_acquisition_function_class()

        init_configs = []
        for x in X_init:
            init_configs.append(Configuration(
                configuration_space=configuration_space,
                vector=x
            ))

        def f_in_smac(x):
            y = f([x[var.name] for var in variables])
            iter_callback(x, y, [x], [y])


        smac_opt = smac.facade.smac_bo_facade.SMAC4BO(
            scenario=scenario,
            tae_runner=f_in_smac,
            model=model,
            model_kwargs=model_kwargs,
            acquisition_function=acquisition_function,
            initial_configurations=config_space,
        )

        smac_opt.optimize()

        return self.run_info.result


#register_global_optimizer("SMAC_BO_optimization", SMACBayesianOptimizer)
