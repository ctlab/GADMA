import unittest

from gadma import *

class TestVariables(unittest.TestCase):
    def _test_is_implemented(self, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except NotImplementedError:
            self.fail("Function %s.%s is not implemented" % (function.__module__, function.__name__))
        except:
            pass

    def test_vars_equal_operator(self):
        var1 = Variable('abc', 'continous', [0,1])
        var2 = Variable('cba', 'continous', [0,1])
        self.assertNotEqual(var1, var2)
        with self.assertRaises(AttributeError):
            Variable('abc', 'continous', [0,1])
            PopulationSizeVariable(name='abc', domain=[0,1])

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
        elif isinstance(variable, DiscreteVariable):
            pos_val = variable.get_possible_values()
            self.assertTrue(value in pos_val)
        else:
            self.fail("Variable class should be ContinuousVariable or DiscreteVariable instance")

    def test_variable_classes(self):
        test_classes = [PopulationSizeVariable, 
                        TimeVariable,
                        MigrationVariable,
                        SelectionVariable,
                        DynamicVariable,
                        PercentVariable]
        for i, cls in enumerate(test_classes):
            self._test_variable(cls('var%d' % i))

if __name__ == '__main__':
    unittest.main()
