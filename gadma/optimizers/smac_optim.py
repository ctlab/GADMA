# code from
# https://github.com/automl/Squirrel-Optimizer-BBO-NeurIPS20-automlorg
# squirrel-optimizer/smac_optim.py
#
# Some lines with print were commented out and some meta info is stored
# Code was formatted with black -l 80

# Check that smac abd configspace are available
from .. import smac_available

import copy
import typing
import numpy as np

if smac_available:
    from ConfigSpace import Configuration

    from smac.scenario.scenario import Scenario

    from smac.runhistory.runhistory import RunHistory
    from smac.runhistory.runhistory2epm import (
        RunHistory2EPM4Cost,
        RunHistory2EPM4LogScaledCost,
    )
    from smac.stats.stats import Stats

    from smac.epm.gaussian_process import GaussianProcess
    from smac.epm.rf_with_instances import RandomForestWithInstances
    from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior
    from smac.epm.gp_kernels import (
        ConstantKernel,
        Matern,
        WhiteKernel,
        HammingKernel,
    )

    from smac.optimizer.acquisition import (
        EI,
        LogEI,
        LCB,
        PI,
    )
    from smac.optimizer.ei_optimization import LocalAndSortedRandomSearch
    from smac.epm.util_funcs import get_types

    from smac.tae.execute_ta_run import StatusType

    from smac.utils.constants import MAXINT

    from smac.optimizer.random_configuration_chooser import ChooserProb

    from bayesmark.abstract_optimizer import AbstractOptimizer
    # from bayesmark.experiment import experiment_main


class RunHistory2EPM4GaussianCopulaCorrect(RunHistory2EPM4Cost):
    """TODO"""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:

        import scipy as sp

        quants = (sp.stats.rankdata(values.flatten()) - 1) / (len(values) - 1)
        cutoff = 1 / (
            4
            * np.power(len(values), 0.25)
            * np.sqrt(np.pi * np.log(len(values)))
        )
        quants = np.clip(quants, a_min=cutoff, a_max=1 - cutoff)
        # Inverse Gaussian CDF
        rval = np.array([sp.stats.norm.ppf(q)
                         for q in quants]).reshape((-1, 1))
        return rval


class SMAC4EPMOpimizer(AbstractOptimizer):
    def __init__(self, api_config, config_space, parallel_setting="LS"):
        super(SMAC4EPMOpimizer, self).__init__(api_config)
        self.cs = config_space
        self.num_hps = len(self.cs.get_hyperparameters())

        if parallel_setting not in ["CL_min", "CL_max", "CL_mean", "KB", "LS"]:
            raise ValueError(
                "parallel_setting can only be one of the following: "
                "CL_min, CL_max, CL_mean, KB, LS"
            )
        self.parallel_setting = parallel_setting

        rng = np.random.RandomState(seed=0)
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alt. to runtime)
                "runcount-limit": 128,
                "cs": self.cs,  # configuration space
                "deterministic": True,
                "limit_resources": False,
            }
        )

        self.stats = Stats(scenario)
        # traj = TrajLogger(output_dir=None, stats=self.stats)

        self.runhistory = RunHistory()

        r2e_def_kwargs = {
            "scenario": scenario,
            "num_params": self.num_hps,
            "success_states": [
                StatusType.SUCCESS,
            ],
            "impute_censored_data": False,
            "scale_perc": 5,
        }

        self.random_chooser = ChooserProb(rng=rng, prob=0.0)

        types, bounds = get_types(self.cs, instance_features=None)
        model_kwargs = {
            "configspace": self.cs,
            "types": types,
            "bounds": bounds,
            "seed": rng.randint(MAXINT),
        }

        models = []

        cov_amp = ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
        )

        cont_dims = np.array(np.where(np.array(types) == 0)[0], dtype=np.int)
        cat_dims = np.where(np.array(types) != 0)[0]

        if len(cont_dims) > 0:
            exp_kernel = Matern(
                np.ones([len(cont_dims)]),
                [
                    (np.exp(-6.754111155189306), np.exp(0.0858637988771976))
                    for _ in range(len(cont_dims))
                ],
                nu=2.5,
                operate_on=cont_dims,
            )

        if len(cat_dims) > 0:
            ham_kernel = HammingKernel(
                np.ones([len(cat_dims)]),
                [
                    (np.exp(-6.754111155189306), np.exp(0.0858637988771976))
                    for _ in range(len(cat_dims))
                ],
                operate_on=cat_dims,
            )
        assert len(cont_dims) + len(cat_dims) == len(
            scenario.cs.get_hyperparameters()
        )

        noise_kernel = WhiteKernel(
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
        else:
            raise ValueError()
        gp_kwargs = {"kernel": kernel}

        rf_kwargs = {}
        rf_kwargs["num_trees"] = model_kwargs.get("num_trees", 10)
        rf_kwargs["do_bootstrapping"] = model_kwargs.get(
            "do_bootstrapping", True
        )
        rf_kwargs["ratio_features"] = model_kwargs.get("ratio_features", 1.0)
        rf_kwargs["min_samples_split"] = model_kwargs.get(
            "min_samples_split", 2
        )
        rf_kwargs["min_samples_leaf"] = model_kwargs.get("min_samples_leaf", 1)
        rf_kwargs["log_y"] = model_kwargs.get("log_y", True)

        rf_log = RandomForestWithInstances(**model_kwargs, **rf_kwargs)

        rf_kwargs = copy.deepcopy(rf_kwargs)
        rf_kwargs["log_y"] = False
        rf_no_log = RandomForestWithInstances(**model_kwargs, **rf_kwargs)

        rh2epm_cost = RunHistory2EPM4Cost(**r2e_def_kwargs)
        rh2epm_log_cost = RunHistory2EPM4LogScaledCost(**r2e_def_kwargs)
        rh2epm_copula = RunHistory2EPM4GaussianCopulaCorrect(**r2e_def_kwargs)

        self.combinations = []

        # 2 models * 4 acquisition functions
        acq_funcs = [EI, PI, LogEI, LCB]
        acq_func_instances = []
        # acq_func_maximizer_instances = []

        n_sls_iterations = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 10,
            6: 10,
            7: 8,
            8: 6,
        }.get(len(self.cs.get_hyperparameters()), 5)

        acq_func_maximizer_kwargs = {
            "config_space": self.cs,
            "rng": rng,
            "max_steps": 5,
            "n_steps_plateau_walk": 5,
            "n_sls_iterations": n_sls_iterations,
        }
        self.idx_ei = 0

        self.num_models = len(models)
        self.num_acq_funcs = len(acq_funcs)

        no_transform_gp = GaussianProcess(
            **copy.deepcopy(model_kwargs), **copy.deepcopy(gp_kwargs)
        )
        ei = EI(model=no_transform_gp)
        acq_func_maximizer_kwargs["acquisition_function"] = ei
        ei_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((no_transform_gp, ei, ei_opt, rh2epm_cost))

        pi = PI(model=no_transform_gp)
        acq_func_maximizer_kwargs["acquisition_function"] = pi
        pi_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((no_transform_gp, pi, pi_opt, rh2epm_cost))

        lcb = LCB(model=no_transform_gp)
        acq_func_maximizer_kwargs["acquisition_function"] = lcb
        lcb_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((no_transform_gp, lcb, lcb_opt, rh2epm_cost))

        gp = GaussianProcess(
            **copy.deepcopy(model_kwargs), **copy.deepcopy(gp_kwargs)
        )
        ei = EI(model=gp)
        acq_func_maximizer_kwargs["acquisition_function"] = ei
        ei_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((gp, ei, ei_opt, rh2epm_copula))

        gp = GaussianProcess(
            **copy.deepcopy(model_kwargs), **copy.deepcopy(gp_kwargs)
        )
        ei = LogEI(model=gp)
        acq_func_maximizer_kwargs["acquisition_function"] = ei
        ei_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((gp, ei, ei_opt, rh2epm_log_cost))

        ei = EI(model=rf_no_log)
        acq_func_maximizer_kwargs["acquisition_function"] = ei
        ei_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((rf_no_log, ei, ei_opt, rh2epm_cost))

        ei = LogEI(model=rf_log)
        acq_func_maximizer_kwargs["acquisition_function"] = ei
        ei_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((rf_log, ei, ei_opt, rh2epm_log_cost))

        ei = EI(model=rf_no_log)
        acq_func_maximizer_kwargs["acquisition_function"] = ei
        ei_opt = LocalAndSortedRandomSearch(**acq_func_maximizer_kwargs)
        self.combinations.append((rf_no_log, ei, ei_opt, rh2epm_copula))

        self.num_acq_instances = len(acq_func_instances)
        self.best_observation = np.inf

        self.next_evaluations = []

    def suggest(self, n_suggestions: int = 1) -> typing.List[typing.Dict]:
        """Get a suggestion from the optimizer.
        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output
        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
            CHANGED: each suggestion is a tuple of suggestion and string info!
        """
        all_previous_configs = self.runhistory.get_all_configs()
        num_points = len(all_previous_configs)

        # we will save our info
        info_list = []
        if len(self.next_evaluations) < n_suggestions:
            n_new = n_suggestions - len(self.next_evaluations)

            # import time

            order = np.random.permutation(list(range(len(self.combinations))))
            optimized_this_iter = set()
            while len(self.next_evaluations) < n_new:
                model, acq, acq_opt, rh2epm = self.combinations[
                    order[len(self.next_evaluations)]
                ]
                # start_time = time.time()

                info = ""
                if model.__class__ == RandomForestWithInstances:
                    info += "RF"
                elif model.__class__ == GaussianProcess:
                    info += "GP"
                else:
                    raise ValueError(model.__class__.__name__)
                info += f" {acq.__class__.__name__}"
                if rh2epm.__class__ == RunHistory2EPM4Cost:
                    info += " cost"
                elif rh2epm.__class__ == RunHistory2EPM4LogScaledCost:
                    info += " log_cost"
                elif rh2epm.__class__ == RunHistory2EPM4GaussianCopulaCorrect:
                    info += " copula"
                else:
                    raise ValueError(rh2epm.__class__.__name__)

                # print(model.__class__.__name__,
                #       acq.__class__.__name__,
                #       rh2epm.__class__.__name__)

                X, y = rh2epm.transform(self.runhistory)

                # If all are not finite then we return nothing
                if np.all(~np.isfinite(y)):
                    self.next_evaluations = []
                    return []

                # Safeguard, just in case...
                if np.any(~np.isfinite(y)):
                    y[~np.isfinite(y)] = np.max(y[np.isfinite(y)])

                if (
                    self.parallel_setting != "LS"
                    and len(self.next_evaluations) != 0
                ):
                    x_inc = np.array(
                        [
                            next_config.get_array()
                            for next_config in self.next_evaluations
                        ]
                    )
                    if self.parallel_setting == "CL_min":
                        y_inc = np.min(y)
                    elif self.parallel_setting == "CL_max":
                        y_inc = np.max(y)
                    elif self.parallel_setting == "CL_mean":
                        y_inc = np.mean(y)
                    elif self.parallel_setting == "KB":
                        if model in optimized_this_iter and isinstance(
                            model, GaussianProcess
                        ):
                            # Safe some time by re-using the optimized
                            # hyperparameters from before
                            model._train(X, y, do_optimize=False)
                        else:
                            model.train(X, y)
                            optimized_this_iter.add(model)
                        y_inc, var = model.predict_marginalized_over_instances(
                            x_inc
                        )
                        y_inc = y_inc.flatten()
                    else:
                        raise ValueError(
                            "parallel_setting can only be one of the "
                            "following: CL_min, CL_max, CL_mean, KB, LS"
                        )
                    if self.parallel_setting in ("CL_min", "CL_max", "CL_mean"):  # NOQA
                        y_inc = np.repeat(
                            y_inc, len(self.next_evaluations)
                        ).reshape((-1, 1))
                    else:
                        y_inc = y_inc.reshape((-1, 1))
                    X = np.concatenate((X, x_inc))
                    y = np.concatenate((y, y_inc))
                    if (
                        isinstance(model, GaussianProcess)
                        and self.parallel_setting == "KB"
                    ):
                        # Safe some time by re-using the optimized
                        # hyperparameters from above
                        model._train(X, y, do_optimize=False)
                    else:
                        model.train(X, y)
                        # As the training data for each subsequent model
                        # changes quite drastically (taking the max of all
                        # observations can create really disconnected error
                        # landscapes in the region of the optimum) we have
                        # to re-optimize the hyperparameters here and cannot
                        # add the model to the set of previously
                        # optimized models.
                        # optimized_this_iter.add(model)
                else:
                    model.train(X, y)
                    optimized_this_iter.add(model)

                predictions = model.predict_marginalized_over_instances(X)[0]
                best_index = np.argmin(predictions)
                best_observation = predictions[best_index]
                x_best_array = X[best_index]

                acq.update(
                    model=model,
                    eta=best_observation,
                    incumbent_array=x_best_array,
                    num_data=num_points,
                    X=X,
                )

                new_config_iterator = acq_opt.maximize(
                    runhistory=self.runhistory,
                    stats=self.stats,
                    num_points=10000,
                    random_configuration_chooser=self.random_chooser,
                )

                accept = False
                for next_config in new_config_iterator:
                    if (
                        next_config in self.next_evaluations
                        or next_config in all_previous_configs
                    ):
                        continue
                    else:
                        accept = True
                        break
                if not accept:
                    # If we don't find anything within 100 random
                    # configurations, we re-run a configuration
                    for next_config in self.cs.sample_configuration(100):
                        if (
                            next_config not in self.next_evaluations
                            or next_config in all_previous_configs
                        ):
                            break
                self.next_evaluations.append(next_config)
                info_list.append(info)
                # print(time.time() - start_time)
        next_guess = [{} for _ in range(n_suggestions)]
        while len(self.next_evaluations) < len(range(n_suggestions)):
            self.next_evaluations.append(self.cs.sample_configuration())
            info_list.append("Random")
        for i in range(n_suggestions):
            eval_next = self.next_evaluations.pop(0)
            next_guess[i] = (eval_next.get_dictionary(), info_list[i])
        return next_guess

#    def init_with_rh(self, rh, iteration):
#        self.runhistory.empty()
#        for rh_value in rh:
#            configuration = Configuration(
#                configuration_space=self.cs, values=rh_value[0]
#            )
#            self.runhistory.add(
#                config=configuration,
#                cost=rh_value[1],
#                time=0,
#                status=StatusType.SUCCESS,
#            )

    def observe(self, X, y):
        """Feed an observation back.
        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary 使用where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        for xx, yy in zip(X, y):
            configuration = Configuration(
                configuration_space=self.cs, values=xx
            )
            self.runhistory.add(
                config=configuration,
                cost=yy,
                time=0,
                status=StatusType.SUCCESS
            )


# if __name__ == "__main__":
#     experiment_main(SMAC4EPMOpimizer)
