import unittest

from gadma import *
import gadma


class TestVariables(unittest.TestCase):
    def _test_is_implemented(self, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except NotImplementedError:
            self.fail("Function %s.%s is not implemented"
                      % (function.__module__, function.__name__))
        except:  # NOQA
            pass

    def test_vars_equal_operator(self):
        var1 = Variable('abc', 'continous', [0, 1], None)
        var2 = Variable('cba', 'continous', [0, 1], None)
        self.assertNotEqual(var1, var2)

    def _test_variable(self, variable):
        self.assertIsInstance(variable, Variable)
        self._test_is_implemented(variable.get_bounds)
        self._test_is_implemented(variable.get_possible_values)
        value = variable.resample()
        self.assertIsNotNone(value)
        if isinstance(variable, ContinuousVariable):
            domain = variable.get_bounds()
            self.assertTrue(value >= domain[0])
            self.assertTrue(value <= domain[1])
            self.assertRaises(ValueError, variable.__class__, variable.name,
                              variable.domain[::-1])
        elif isinstance(variable, DiscreteVariable):
            pos_val = variable.get_possible_values()
            self.assertTrue(value in pos_val)
        else:
            self.fail("Variable class should be ContinuousVariable"
                      " or DiscreteVariable instance")

    def test_variable_classes(self):
        test_classes = [PopulationSizeVariable,
                        TimeVariable,
                        MigrationVariable,
                        SelectionVariable,
                        DynamicVariable,
                        FractionVariable]
        for i, cls in enumerate(test_classes):
            with self.subTest(variable_cls=cls):
                self._test_variable(cls('var%d' % i))

        d = DiscreteVariable('d')
        self.assertRaises(ValueError, d.get_bounds)
        d = DiscreteVariable('d', [0, 1, 5, 4])
        self.assertTrue(d.get_bounds() == [0, 5])

        ContinuousVariable('c')

        d = DynamicVariable('d')
        self.assertRaises(Exception, DynamicVariable, 'd', domain=[5, 'Sud'])
        self.assertRaises(Exception, d.get_func_from_value, 100)

        dyn = gadma.variables.Dynamic()
        self.assertRaises(NotImplementedError, dyn._inner_func, 1, 2, 0.2)
        self.assertRaises(NotImplementedError, dyn.__str__)

        var = Variable('var', 'discrete', [0, 1], np.random.choice)
        self.assertRaises(NotImplementedError, var.get_bounds)
        self.assertRaises(NotImplementedError, var.get_possible_values)
        self.assertRaises(NotImplementedError, var.correct_value, 5)

    def test_dynamics(self):
        y1 = 1
        y2 = 5
        t = 3
        for cls in [gadma.variables.Exp, gadma.variables.Lin,
                    gadma.variables.Sud]:
            el = cls()
            print(el)
            func = el._inner_func(y1, y2, t)
            print(func)
            if str(el) != 'Sud':
                self.assertEqual(func(0), y1)
            else:
                self.assertEqual(func(0), y2)
                self.assertEqual(func(t / 2), y2)
            self.assertEqual(func(t), y2)

