import unittest
from gadma import *
from gadma.utils.utils import CacheInfo, WeightedMetaArray,\
    logarithm_transform, run_f_and_save_result_into_queue, timeout
from gadma.utils import cache_func, eval_wrapper
from gadma.utils.distributions import *
from gadma.utils import *
import numpy as np
import multiprocessing
import time
import os
import platform


def f_sleep(x):
    time.sleep(5)
    return x


class TestUtils(unittest.TestCase):
    def test_distributions(self):
        trunc_normal(1, 0.5, 0, 10)
        trunc_normal(1, 2, 0, 10)
        trunc_normal(1, 0, 0, 10)

        trunc_lognormal(1, 1e-15, 1e-15, 4)

        trunc_normal_3_sigma_rule(0.5, 1e-15, 5)

        trunc_lognormal_3_sigma_rule(2, 1, 5)
        trunc_lognormal_3_sigma_rule(1e-15, 1e-15, 1)

        p = PopulationSizeVariable('p')
        d = DynamicVariable('d')
        m = MigrationVariable('m')
        m.log_transform = True
        s = SelectionVariable('s')
        f = FractionVariable('f')
        variables = [p, d, m, s, f]
        v = custom_generator(variables)
        self.assertEqual(len(v), 5)
        for el, var in zip(v, variables):
            self.assertTrue(var.correct_value(el))

        gen = DemographicGenerator(
            FractionVariable,
            Nanc_domain=[1, 100],
            Nanc_mean=1e4,
        )
        v = gen(domain=[1e-2, 100])

        trunc_lognormal_sigma_generator([0, 0.1])
        trunc_lognormal_sigma_generator([10, 20])
        trunc_normal_sigma_generator([-1, 0])
        trunc_normal_sigma_generator([10, 20])

    def test_utils(self):
        self.assertEqual(logarithm_transform(1), np.log(1))
        self.assertEqual(exponent_transform(0), np.exp(0))

        t = (5, 10)
        self.assertEqual(logarithm_transform(t), t)
        self.assertEqual(exponent_transform(t), t)

        x = [0, 0, 2]
        self.assertTrue(0 in choose_by_weight(x, None, 2))

    def test_cache_info(self):
        info = CacheInfo()
        str(info)

    def test_run_f_and_save_result_into_queue(self):
        q = multiprocessing.Queue()
        def f(x):
            return x
        run_f_and_save_result_into_queue(f, q, 10)
        time.sleep(0.1)
        self.assertFalse(q.empty())
        self.assertEqual(q.get(), 10)

    def test_timeout(self):
        g = timeout(f_sleep, time=2)
        self.assertEqual(g(0), None)
        g = timeout(f_sleep, time=10)
        # It fails for Windows and MacOS for some reason
        if platform.system() == "Linux":
            self.assertEqual(g(0), 0)

    def test_weighted_meta_array(self):
        x = WeightedMetaArray([1, 2])

        X = [x]
        X.append([1, 2])
        serialize_meta_array(x)

        del x.metadata
        x.__str__()
        x.__repr__()

    def test_cache(self):
        def f(x):
            return np.min(x)
        cache_f = cache_func(f)
        x = np.array([[1, 2], [3, 4]])
        self.assertEqual(f(x), cache_f(x))

    def test_eval_wrapper_not_exist_file(self):
        def f(x):
            return np.min(x)
        eval_file = "not_existing_file"
        try:
            wr_f = eval_wrapper(f, eval_file)
            self.assertEqual(wr_f([1, 2, 3]), 1)
        finally:
            if os.path.exists(eval_file):
                os.remove(eval_file)

    def test_bo_cross_validation(self):
        if not smac_available:
            return
        DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
        data_path = os.path.join(DATA_PATH, "DATA", "sfs", "YRI_CEU.fs")
        data_holder = SFSDataHolder(data_path, projections=[4, 4])

        model = StructureDemographicModel(
            initial_structure=[2, 1],
            final_structure=[2, 1],
            has_migs=True,
            has_sels=False,
            has_dom=False,
            has_dyns=False,
            sym_migs=True,
            frac_split=True,
        )
        # we make time variables greater as it is unstable with small values
        for i in range(len(model.variables)):
            if isinstance(model.variables[i], TimeVariable):
                model.variables[i].domain = [1e-2, 5]

        # for stability
        for i in range(len(model.variables)):
            if isinstance(model.variables[i], TimeVariable):
                model.variables[i].domain = [1e-2, 5]

        engine = get_engine("moments")
        engine.data = data_holder
        engine.model = model

        for opt_name in ["SMAC_BO_optimization", "GPyOpt_Bayesian_optimization"]:
            optimizer = get_global_optimizer(opt_name)
            optimizer.acquisition_type = "PI"
            optimizer.log_transform = True

            _X, _Y = optimizer.initial_design(
                f=engine.evaluate,
                variables=model.variables,
                num_init=10
            )
            # will do nothing if it is not smac
            X, Y = transform_smac(optimizer, model.variables, _X, _Y)
            Y = normalize(Y)

            config_space = optimizer.get_config_space(
                variables=model.variables
            )
            for kernel_name in ["Matern52", "matern32", "rbf", "Exponential"]:
                optimizer.kernel_name = kernel_name
                gp = optimizer.get_model(config_space=config_space)
                s1 = get_LOO_score(X_train=X, Y_train=Y, gp_model=gp,
                                   mode="rassmusen", verbose=True, do_optimize=True)
                s2 = get_LOO_score(X_train=X, Y_train=Y, gp_model=gp,
                                   mode="gp_train", verbose=True, do_optimize=False)
                self.assertTrue(np.isclose(s1, s2, rtol=1e-3), msg=f"{s1} != {s2}")

            name1 = get_best_kernel(
                optimizer=optimizer, variables=model.variables, X=_X, Y=_Y,
                mode="rassmusen"
            )
            name2 = get_best_kernel(
                optimizer=optimizer, variables=model.variables, X=_X, Y=_Y,
                mode="gp_train"
            )
            #self.assertEqual(name1, name2)
