import unittest
from gadma import *
import numpy as np


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
        s = SelectionVariable('s')
        f = FractionVariable('f')
        variables = [p, d, m, s, f]
        v = custom_generator(variables)
        self.assertEqual(len(v), 5)
        for el, var in zip(v, variables):
            self.assertTrue(var.correct_value(el))

        trunc_lognormal_sigma_generator([0, 0.1])
        trunc_lognormal_sigma_generator([10, 20])
        trunc_normal_sigma_generator([-1, 0])
        trunc_normal_sigma_generator([10, 20])

    def test_utils(self):
        t = (5, 10)
        self.assertEqual(logarithm_transform(t), t)
        self.assertEqual(exponent_transform(t), t)

        x = [0, 0, 2]
        self.assertTrue(0 in choose_by_weight(x, None, 2))

    def test_cache_info(self):
        info = CacheInfo()
        str(info)

    def test_weighted_meta_array(self):
        x = WeightedMetaArray([1, 2])

        X = [x]
        X.append([1, 2])
        list_with_weights_for_pickle(X)

        del x.metadata
        x.__str__()
        x.__repr__()
