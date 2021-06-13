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


def f_sleep_10(x):
    time.sleep(10)
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

        def gen_gen(domain):
            return np.random.uniform(domain[0], domain[1])
        gen = DemographicGenerator(
            genetic_generator=gen_gen,
            N_A_domain=[1, 100],
            gen_time=20
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
        g = timeout(f_sleep_10, time=5)
        self.assertEqual(g(0), None)
        g = timeout(f_sleep_10, time=15)
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
