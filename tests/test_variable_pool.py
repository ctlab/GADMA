import unittest

from gadma import *

import numpy as np
import pickle

class TestVariablePool(unittest.TestCase):
    def test_init(self):
        pool = VariablePool()

        v = Variable("var1", '', '', '')
        m = MigrationVariable("nu")
        n = PopulationSizeVariable("nu")
        t = TimeVariable("t")

        var_list = [v, n, t]
        pool = VariablePool(var_list)
        var_list.append(m)
        self.assertRaises(NameError, VariablePool, var_list)

    def test_append(self):
        v = Variable("var1", '', '', '')
        m = MigrationVariable("m")
        n = PopulationSizeVariable("nu")
        t = TimeVariable("t")
        x = MigrationVariable("nu")

        pool = VariablePool()
        pool.append(v)
        self.assertEqual(pool.names, set(['var1']))
        pool.append(n)
        self.assertEqual(pool.names, set(['var1', 'nu']))
        self.assertRaises(NameError, pool.append, v)
        self.assertEqual(pool.names, set(['var1', 'nu']))
        self.assertRaises(NameError, pool.append, x)
        self.assertEqual(pool.names, set(['var1', 'nu']))
        self.assertRaises(ValueError, pool.append, 0)
        self.assertEqual(pool.names, set(['var1', 'nu']))
        pool[1] = x
        self.assertEqual(pool.names, set(['var1', 'nu']))
        self.assertRaises(NameError, pool.__setitem__, 1, v)
        self.assertEqual(pool.names, set(['var1', 'nu']))
        pool.extend([m, t])  # v,x, m, t
        self.assertEqual(pool.names, set(['var1', 'nu', 'm', 't']))
        pool[2:4] = [t, m]
        self.assertRaises(NameError, pool.__setitem__, slice(2, 4), [t, n])

    def test_delete(self):
        v = Variable("var1", '', '', '')
        m = MigrationVariable("nu")
        n = PopulationSizeVariable("nu")
        t = TimeVariable("t")

        pool = VariablePool([v, n, t])
        del pool[1]
        pool.append(m)
        del pool[:1]

    def test_copy(self):
        v = Variable("var1", '', '', '')
        n = PopulationSizeVariable("nu")
        t = TimeVariable("t")

        pool1 = VariablePool([v, n, t])
        self.assertTrue(pool1[0] is v)
        self.assertTrue(pool1[1] is n)
        self.assertTrue(pool1[2] is t)
        pool2 = pool1
        self.assertTrue(pool1 is pool2)
        self.assertTrue(pool2[0] is v)
        self.assertTrue(pool2[1] is n)
        self.assertTrue(pool2[2] is t)
        pool2 = copy.copy(pool1)
        self.assertTrue(pool1 is not pool2)
        self.assertTrue(pool2[0] is v)
        self.assertTrue(pool2[1] is n)
        self.assertTrue(pool2[2] is t)
        pool2 = copy.deepcopy(pool1)
        self.assertTrue(pool1 is not pool2)
        self.assertEqual(pool2[0].name, v.name)
        self.assertEqual(pool2[1].name, n.name)
        self.assertEqual(pool2[2].name, t.name)
        self.assertTrue(pool2[0] is not v)
        self.assertTrue(pool2[1] is not n)
        self.assertTrue(pool2[2] is not t)

    def test_pickling(self):
        v = Variable("var1", '', '', '')
        n = PopulationSizeVariable("nu")
        t = TimeVariable("t")

        pool = VariablePool([v, n, t])
        out_file = "pickle_file"
        with open(out_file, 'wb') as fl:
            pickle.dump(pool, fl)

        with open(out_file, 'rb') as fl:
            pool = pickle.load(fl)
        pool.names
        os.remove(out_file)

        del pool.names
        self.assertRaises(AttributeError, pool.__getattribute__, "name")
        pool.fix_pickling()
        self.assertEqual(pool.names, set(["var1", "nu", "t"]))
