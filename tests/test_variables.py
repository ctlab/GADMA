import unittest

from gadma import *
import gadma
import numpy as np
import copy


test_classes = [PopulationSizeVariable,
                TimeVariable,
                MigrationVariable,
                SelectionVariable,
                DynamicVariable,
                FractionVariable]


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
            variable.correct_value(variable.domain[1] + 1e-15)
        elif isinstance(variable, DiscreteVariable):
            pos_val = variable.get_possible_values()
            self.assertTrue(value in pos_val)
        else:
            self.fail("Variable class should be ContinuousVariable"
                      " or DiscreteVariable instance")

    def test_variable_classes(self):
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

        dyn = gadma.utils.variables.Dynamic()
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
        for cls in [gadma.utils.variables.Exp, gadma.utils.variables.Lin,
                    gadma.utils.variables.Sud]:
            el = cls()
            func = el._inner_func(y1, y2, t)
            if str(el) != 'Sud':
                self.assertEqual(func(0), y1)
            else:
                self.assertEqual(func(0), y2)
                self.assertEqual(func(t / 2), y2)
            self.assertEqual(func(t), y2)

    def test_log_trasform(self):
        for var_cls in test_classes:
            var = var_cls("name")
            if var_cls == DynamicVariable:
                self.assertRaises(NotImplementedError, var.apply_logarithm)
                continue
            var.log_transformed = False
            self.assertFalse(var.log_transformed)
            old_domain = list(var.domain)
            var.log_transformed = True
            self.assertTrue(var.log_transformed)
            var.resample()
            var.log_transformed = False
            self.assertFalse(var.log_transformed)
            self.assertEqual(list(var.domain), old_domain)

    def test_demographic_variables(self):
        var = DemographicVariable("dem")
        self.assertRaises(NotImplementedError,
                          var._transform_value_from_gen_to_phys,
                          value=0,
                          Nanc=10)
        self.assertRaises(NotImplementedError,
                          var._transform_value_from_phys_to_gen,
                          value=0,
                          Nanc=10)
        var1 = PopulationSizeVariable("var1", units="physical")
        var2 = PopulationSizeVariable("var2")
        self.assertEqual(var1.translate_value_into("physical", 1e4), 1e4)
        self.assertEqual(var2.translate_value_into("physical", 1.2, 1e4), 1.2e4)
        N_A = PopulationSizeVariable("N_A", domain=[1e3, 1e4])
        var2.translate_units_to("physical", N_A.domain)
        self.assertEqual(var2.units, "physical")
        var3 = TimeVariable("var3", units="physical")
        self.assertEqual(var3.translate_value_into("genetic", 1e4, 1e2), 50)
        var3.translate_units_to("genetic", N_A.domain)
        self.assertEqual("genetic", var3.units)

        for var_cls in test_classes:
            if (issubclass(var_cls, ContinuousVariable) and 
                    var_cls is not FractionVariable):
                var = var_cls("name", units="genetic")
                var._transform_value_from_gen_to_phys(value=0, Nanc=10)
                var._transform_value_from_phys_to_gen(value=0, Nanc=10)
                self.assertEqual(list(var.domain),
                                 list(var_cls.default_domain))
                var.translate_units_to(units="physical")
                self.assertEqual(var.units, "physical")
                self.assertNotEqual(list(var.domain),
                                    list(var_cls.default_domain))

                var = var_cls("name", units="physical")
                self.assertNotEqual(list(var.domain),
                                    list(var_cls.default_domain))
                for i in range(1000):
                    self.assertTrue(var.domain[0] <= var.resample() \
                                    <= var.domain[1])
                var.translate_units_to(units="genetic")
                self.assertEqual(var.units, "genetic")
                self.assertEqual(list(var.domain),
                                 list(var_cls.default_domain))
                self.assertRaises(ValueError, var.translate_value_into,
                                  units="physical", value=var.domain[0])
                self.assertRaises(ValueError, var.translate_value_into,
                                  units="some_invalid", value=var.domain[0],
                                  Nanc=1e4)

                # rescaling
                old_domain = np.array(var.domain)
                var.rescale(Nref=2)
                var.resample()
                # as it is genetic units nothing happens
                self.assertEqual(list(old_domain), list(var.domain))

                var.translate_units_to("physical")
                old_domain = np.array(var.domain)
                var.rescale(Nref=None)
                # As it is Nref == None nothing happens
                self.assertEqual(list(old_domain), list(var.domain))
                var.rescale(Nref=2)
                var.resample()
                self.assertNotEqual(list(old_domain), list(var.domain))

                # rescale back
                var.rescale(Nref=2, reverse=True)
                var.resample()
                self.assertEqual(list(old_domain), list(var.domain))
            else:
                var = var_cls("name")
                self.assertEqual(var.units, "universal")
                self.assertRaises(TypeError, var_cls, units="physical")
                var_orig = copy.deepcopy(var)
                for units in ["physical", "genetic"]:
                    var.translate_units_to(units)
                    var._transform_value_from_gen_to_phys(value=0, Nanc=10)
                    var._transform_value_from_phys_to_gen(value=0, Nanc=10)
                    self.assertEqual(var.units, "universal")
                    self.assertEqual(list(var_orig.domain), list(var.domain))
            self.assertRaises(ValueError, var.translate_value_into,
                              units="physical", value=-1)

