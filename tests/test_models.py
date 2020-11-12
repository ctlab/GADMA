import unittest

from gadma import *
from gadma import ContinuousVariable

from .test_data import YRI_CEU_DATA

try:
    import dadi
    DADI_NOT_AVAILABLE = False
except ImportError:
    DADI_NOT_AVAILABLE = True

import numpy as np
import os
import operator as op
import sys

EXAMPLE_FOLDER = os.path.join(os.path.dirname(__file__), "test_data")


class TestModels(unittest.TestCase):
    def test_init(self):
        var = TimeVariable('t')
        m = Model(raise_excep=False)
        m.add_variable(var)
        m.add_variable(1.0)
        m = Model(raise_excep=True)
        m.add_variable(var)
        self.assertRaises(ValueError, m.add_variable, 1.0)
        self.assertRaises(TypeError, m.var2value, set())
        
    def dadi_wrapper(self, func):
        def wrapper(param, ns, pts):
            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = func(param, xx, phi)
            sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
            return sfs
        return wrapper

    def get_variables_for_gut_2009(self):
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')
        return (nu1F, nu2B, nu2F, m, Tp, T, Dyn)

    def test_custom_dm_init(self):
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                filename = f"demographic_model_{engine.id}_YRI_CEU.py"
                location = os.path.join(EXAMPLE_FOLDER, filename)
                spec = importlib.util.spec_from_file_location("module",
                                                              location)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules['module'] = module
                func = getattr(module, 'model_func')
                variables = self.get_variables_for_gut_2009()[:-1] 
                dm = CustomDemographicModel(func, variables)
                dm.as_custom_string([var.resample() for var in dm.variables])
                dm.as_custom_string({var.name: var.resample()
                                     for var in dm.variables})

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_1pop_0(self):
        @self.dadi_wrapper
        def inner(param, xx, phi):
            return phi

        ns = (20,)
        pts = [40, 50, 60]
        func_ex = dadi.Numerics.make_extrap_log_func(inner)
        real = func_ex([], ns, pts)

        dm = EpochDemographicModel()
        pts = [40, 50, 60]
        d = get_engine('dadi')
        d.set_model(dm)
        got = d.simulate([], ns, pts)
        self.assertTrue(np.allclose(got, real))
        self.assertEqual(dm.number_of_populations(), 1)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_1pop_1(self):
        @self.dadi_wrapper
        def inner(param, xx, phi):
            T, nu, sel = param
            phi = dadi.Integration.one_pop(phi, xx, T=T, gamma=sel, nu=nu)
            return phi

        ns = (20,)
        pts = [40, 50, 60]
        param = [1., 0.5, 1.5]
        func_ex = dadi.Numerics.make_extrap_log_func(inner)
        real = func_ex(param, ns, pts)

        T = TimeVariable('T1')
        nu = PopulationSizeVariable('nu2')
        sl = SelectionVariable('sel')
        dm = EpochDemographicModel()
        dm.add_epoch(T, [nu], sel_args=[sl])
        d = get_engine('dadi')
        d.set_model(dm)
        got = d.simulate(param, ns, pts)
        self.assertTrue(np.allclose(got, real))
        self.assertEqual(dm.number_of_populations(), 1)
        dm.as_custom_string(param)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_gut_2pop(self):
        """
        Check loglikelihood of the demographic model from the YRI_CEU
        example of dadi.
        """
        nu1F, nu2B, nu2F, m, Tp, T, Dyn = self.get_variables_for_gut_2009()

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', 'Exp'])

        dic = {'nu1F': 1.880, 'nu2B': 0.0724, 'nu2F': 1.764, 'm': 0.930,
               'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp'}

        data = SFSDataHolder(YRI_CEU_DATA)
        d = DadiEngine(model=dm, data=data)
        values = [dic[var.name] for var in dm.variables]
        ll = d.evaluate(values, pts=[40, 50, 60])
        self.assertEqual(int(ll), -1066)
        self.assertEqual(dm.number_of_populations(), 2)

    def test_fix_vars(self):
        nu1F, nu2B, nu2F, m, Tp, T, Dyn = self.get_variables_for_gut_2009()
        Dyn2 = DynamicVariable('SudDyn')

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F], dyn_args=[Dyn2])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', Dyn])

        dic = {'nu1F': 1.880, nu2B: 0.0724, 'nu2F': 1.764, 'm': 0.930,
               'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp', 'SudDyn': 'Sud'}

        data = SFSDataHolder(YRI_CEU_DATA)
        d = DadiEngine(model=dm, data=data)
        values = dic#[dic[var.name] for var in dm.variables]
        ll1 = d.evaluate(values, pts=[40, 50, 60])
        n_par_before = dm.get_number_of_parameters(dic)
        
        dm.fix_variable(Dyn, 'Exp')
        d.model = dm
        ll2 = d.evaluate(dic, pts=[40, 50, 60])
        n_par_after = dm.get_number_of_parameters(dic)

        dm.unfix_variable(Dyn)
        n_par_after_after = dm.get_number_of_parameters(dic)

        self.assertEqual(ll1, ll2)
        self.assertEqual(n_par_before, 8)
        self.assertEqual(n_par_before, n_par_after + 1)
        self.assertEqual(n_par_before, n_par_after_after)

        dm.fix_dynamics(dic)
        n_par_without_dyns = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_before, n_par_without_dyns + 2)

        dm.unfix_if_fixed(Dyn)
        n_par_without_one = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_without_dyns + 1, n_par_without_one)
        dm.unfix_if_fixed(Dyn)
        n_par_same = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_same, n_par_without_one)

        dm.unfix_dynamics()
        n_par_with_dyns = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_before, n_par_with_dyns)

        dic['Dyn'] = 'Sud'
        n_par_sud_model = dm.get_number_of_parameters(dic)
        self.assertEqual(n_par_before, n_par_sud_model + 1)

        # check fail
        var = PopulationSizeVariable('nu3')
        self.assertRaises(ValueError, dm.fix_variable, var, 3)
        self.assertRaises(ValueError, dm.unfix_variable, var)

#        dm.events[2].set_value(nu2F, 1.0)
#        n_par_after = dm.get_number_of_parameters(dic)
#        self.assertEqual(n_par_sud_model, n_par_after + 1)

        model = Model()
        model.add_variables([nu1F, m, Tp, Dyn,
                             SelectionVariable('sel'), FractionVariable('f')])
        self.assertRaises(ValueError, model.fix_variable, var, 1)
        model.string_repr([1 for _ in model.variables])

    def test_printing_and_translation(self):
        nu1F, nu2B, nu2F, m, Tp, T, Dyn = self.get_variables_for_gut_2009()
        f = FractionVariable('f')
        Dyn2 = DynamicVariable('SudDyn')
        sel = SelectionVariable('s')
        dom = FractionVariable('dom')

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F], dyn_args=[Dyn2],
                     sel_args=[sel], dom_args=[dom])
        dm.add_split(0, [nu1F, Multiplication(f, nu2B)])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', Dyn])

        dic = {'nu1F': 1.880, nu2B: 0.0724, 'f': 1.0, 'nu2F': 1.764,
               'm': 0.930, 'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp',
               'SudDyn': 'Sud', 's': 0.1, 'dom': 0.5}

        data = SFSDataHolder(YRI_CEU_DATA)
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                if engine.id == 'dadi':
                    args = ([5, 10, 15],)
                else:
                    args = ()
                engine.set_model(dm)
                engine.set_data(data)
                Nanc = engine.get_theta(dic, *args)
                tr = dm.translate_units(dic, Nanc)
                dm_copy = copy.deepcopy(dm)
                dm_copy.add_variable(ContinuousVariable("some", [-1, 10]))
                dic['some'] = 0
                self.assertRaises(ValueError, dm_copy.translate_units,
                                  dic, Nanc)
                dm.as_custom_string(dic)

        # test failures
        event = Event()
        x = [var.resample() for var in event.variables]
        self.assertRaises(NotImplementedError, event.as_custom_string, x)

        self.assertRaises(ValueError, Epoch, T, [], [],
                          sel_args=None, dom_args=[])

    def test_var_combinations(self):
        var1 = PopulationSizeVariable('nu1')
        var2 = FractionVariable('f')

        values = [1.0, 2.0]
        const = 5

        comb = VariablesCombination()
        comb.__str__()

        binary_classes = [Addition, Subtraction, Multiplication, Division]
        strings = ['+', '-', '*', '/'] 
        for op_f, cls, op_str in zip([op.add, op.sub, op.mul, op.truediv],
                                     binary_classes,
                                     strings):
            with self.subTest(operator=op_str):
                obj = cls(var1, var2)
                self.assertEqual(obj.get_value(values), op_f(*values))
                self.assertEqual(obj.operation_str(), op_str)
                obj.string_repr(values)
                self.assertEqual(obj.name, f'nu1 {op_str} f')

                obj = cls(var1, const)
                self.assertEqual(obj.get_value(values[:1]),
                                 op_f(values[0], const))
                self.assertEqual(obj.operation_str(), op_str)
                obj.string_repr(values[:1])
                self.assertEqual(obj.name, f'nu1 {op_str} 5')

                for cls2, op_str2 in zip(binary_classes, strings):
                    with self.subTest(operator_2=op_str2):
                        obj2 = cls2(var2, const)
                        obj = cls(var1, obj2)
                        op_f2 = obj2.operation
                        self.assertEqual(obj.get_value(values),
                                         op_f(values[0], op_f2(values[1],
                                              const)))
                        obj.string_repr(values)
                        self.assertEqual(obj.name,
                                         f'nu1 {op_str} (f {op_str2} 5)')

                self.assertRaises(AssertionError, cls, const, const)
        # failures
        bin_op = BinaryOperation(var1, var2)
        self.assertRaises(NotImplementedError, bin_op.operation, 1, 2)
        self.assertRaises(NotImplementedError, bin_op.operation_str)

    def _sfs_datasets(self):
        yield ("usual fs",
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER, '3d_sfs.fs')))
        yield ("folded fs with mixed labels and downsizing",
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER, '3d_sfs.fs'),
                             projections=[8, 8, 8],
                             population_labels=['YRI', 'ASW', 'CEU'],
                             outgroup=False))
        yield ("fs without pop labels",
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER, '3d_sfs_no_name.fs'),
                             population_labels=['1', '2', '3']))
        yield ("dadi snp file",
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER,
                                          'dadi_snp_file.txt')))

    def test_code_generation(self):
        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        t = TimeVariable('t1')
        t2 = TimeVariable('t2')
        m = MigrationVariable('m')
        s = SelectionVariable('s')
        d1 = DynamicVariable('d1')
        d2 = DynamicVariable('d2')
        f = FractionVariable('f')
        h = FractionVariable('h')
        fxnu1 = Multiplication(f, nu1)
        tf = Multiplication(f, t)

        model1 = EpochDemographicModel()
        model1.add_epoch(t, [nu1])
        model1.add_split(0, [nu1, nu2])
        model1.add_epoch(t, [nu2, fxnu1], [[0, m], [0, 0]], [d1, d2])
        model1.add_split(1, [nu2, nu1])
        model1.add_epoch(t, [nu1, nu2, nu1], None, None)

        model2 = EpochDemographicModel()
        model2.add_epoch(t, [nu1])
        model2.add_split(0, [nu1, nu2])
        model2.add_epoch(10, [nu2, fxnu1], [[0, m], [0, 0]], [d1, d2],
                         [0, s])
        model2.add_split(1, [nu2, nu1])
        model2.add_epoch(t, [nu1, nu2, nu1], None, None, [s, 0, s])

        model2.get_involved_for_split_time_vars(1)

        model3 = EpochDemographicModel()
        model3.add_epoch(t, [nu1])
        model3.add_split(0, [nu1, nu2])
        model3.add_epoch(tf, [nu2, fxnu1], [[0, m], [0, 0]], [d1, d2],
                         [0, s],  [0.1, 0.8])
        model3.add_split(1, [nu2, nu1])
        model3.add_epoch(t, [nu1, nu2, nu1], None, [d1, 'Sud', 'Sud'],
                         [s, 0, s], [h, 0.5, 0])

        values = {nu1: 2, nu2: 0.5, nu3: 0.5, t: 0.3, t2: 0.5,
                  m: 0.1, s: 0.1, d1: 'Exp', d2: 'Lin', f: 0.5, h: 0.3}

        for engine in all_engines():
            customfile = os.path.join(
                EXAMPLE_FOLDER, f"demographic_model_{engine.id}_3pops.py")

            spec = importlib.util.spec_from_file_location(
                f"module_{engine.id}", customfile)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[f'module_{engine.id}'] = module
            func = module.model_func
            variables = [nu1, nu2, nu3, m, t, t2]

            model4 = CustomDemographicModel(func, variables)

            for ind, model in enumerate([model1, model2, model3, model4]):
                for description, data in self._sfs_datasets():
                    msg = f"for model {ind + 1} and {description} data and "\
                          f"{engine.id} engine"
                    if engine.id == 'dadi':
                        options = {'pts': [5, 10, 15]}
                        args = (options['pts'],)
                    else:
                        options = {}
                        args = ()
                    model.mu = 1e-8
                    data.sequence_length = 1e10

                    true_ll = engine.set_and_evaluate(values, model,
                                                      data, options)
                    cmd = engine.generate_code(values, None, *args)

                    d = {}
                    exec(cmd, d)

                    msg += f": {true_ll} != {d['ll_model']}"
                    self.assertTrue(np.allclose(true_ll, d['ll_model']),
                                    msg=msg)
