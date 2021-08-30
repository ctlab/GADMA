import unittest
import pytest


from gadma import *
from gadma import ContinuousVariable
from gadma.models import *
from gadma.models.variables_combinations import BinaryOperation, operation_creation, Exp, Log
from gadma.models import TreeDemographicModel
from gadma.engines import DadiEngine
from gadma.cli import get_par_labels_from_file, get_variables

from .test_data import YRI_CEU_DATA
import importlib
import copy

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

        m = DemographicModel()
        self.assertRaises(NotImplementedError, m._get_Nanc_size, values=[])

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
                location = os.path.join(EXAMPLE_FOLDER, "MODELS", filename)
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

    def test_epoch_dm_init(self):
        dm = EpochDemographicModel()
        self.assertEqual(len(dm.variables), 0)
        self.assertFalse(dm.has_anc_size)
        self.assertTrue(dm.Nanc_size in dm.fixed_values)
        self.assertEqual(dm.fixed_values[dm.Nanc_size], 1)

        # no Nanc_size so has_anc_size is False
        dm = EpochDemographicModel(has_anc_size=True)
        self.assertFalse(dm.has_anc_size)

        dm = EpochDemographicModel(Nanc_size=10000)
        self.assertTrue(dm.has_anc_size)
        self.assertEqual(dm.get_Nanc_size(), 10000)

        var_phys = PopulationSizeVariable("nu1", units="physical")
        var_gen = PopulationSizeVariable("nu2", units="genetic")
        var_t = TimeVariable("t", units="physical")

        dm = EpochDemographicModel(has_anc_size=False)
        self.assertRaises(ValueError, dm.add_variable, var_phys)
        self.assertRaises(ValueError, dm.get_Nanc_size)

        self.assertRaises(ValueError, EpochDemographicModel, Nanc_size=var_gen)
        self.assertRaises(ValueError, EpochDemographicModel, Nanc_size=var_t)

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
        got = d.simulate([], ns, 1e6, None, pts)
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
        got = d.simulate(param, ns, 1e6, None, pts)
        self.assertTrue(np.allclose(got, real))
        self.assertEqual(dm.number_of_populations(), 1)
        dm.as_custom_string(param)

    def test_dadi_1pop_1_with_inbreeding(self):
        def inner(param, ns, pts):
            T, nu, sel, F = param

            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = dadi.Integration.one_pop(phi, xx, T=T, gamma=sel, nu=nu)

            sfs = dadi.Spectrum.from_phi_inbreeding(phi, ns, (xx,), (F,), (2,))

            return sfs

        ns = (20,)
        pts = [40, 50, 60]
        param = [1., 0.5, 1.5, 0.5]
        func_ex = dadi.Numerics.make_extrap_log_func(inner)
        real = func_ex(param, ns, pts)

        T = TimeVariable('T1')
        nu = PopulationSizeVariable('nu2')
        sl = SelectionVariable('sel')
        F = FractionVariable("F")
        dm = EpochDemographicModel()
        dm.add_epoch(T, [nu], sel_args=[sl])
        dm.add_inbreeding(inbr_args=[F])
        d = get_engine('dadi')
        d.set_model(dm)
        got = d.simulate(param, ns, 1e6, None, pts)

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
               'Tp': 0.363, 'T': 0.112, 'Dyn': 'Exp'}

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
        Nanc = PopulationSizeVariable("Nanc", units="physical")

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F], dyn_args=[Dyn2],
                     sel_args=[sel], dom_args=[dom])
        dm.add_split(0, [nu1F, Multiplication(f, nu2B)])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', Dyn])
        dm.mutation_rate = 1e-8

        dic = {'nu1F': 1.880, nu2B: 0.0724, 'f': 0.9, 'nu2F': 1.764,
               'm': 0.930, 'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp',
               'SudDyn': 'Sud', 's': 0.1, 'dom': 0.5, 'Nanc': 7300}

        data = SFSDataHolder(YRI_CEU_DATA, sequence_length=4e6)
        for engine in all_engines():
            with self.subTest(engine=engine.id):
                if engine.id == 'dadi':
                    args = ([5, 10, 15],)
                else:
                    args = ()
                model = copy.deepcopy(dm)
                if engine.id not in ['dadi', "moments"]:
                    model.Nanc_size = Nanc
                engine.set_model(model)
                engine.set_data(data)
                Nanc = engine.get_N_ancestral(dic, *args)
                tr = dm.translate_values("physical", dic, Nanc)
                _tr = dm.translate_values("genetic", dic, Nanc)
                model.Nref = 1
                model.translate_values("physical", dic, Nanc, rescale_back=True)
                model.as_custom_string(dic)

        # test failures
        # hack
        assert not dm.has_anc_size
        phys_var = PopulationSizeVariable("nu", units="physical")
        super(DemographicModel, dm).add_variable(phys_var)
        self.assertRaises(ValueError, dm.translate_values, "genetic", dic, Nanc=None)
        # other stuff
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
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER, 'DATA',
                                          'sfs', '3d_sfs.fs')))
        yield ("folded fs with mixed labels and downsizing",
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER, 'DATA',
                                          'sfs', '3d_sfs.fs'),
                             projections=[8, 8, 8],
                             population_labels=['YRI', 'CHB', 'CEU'],
                             outgroup=False))
        yield ("fs without pop labels",
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER, 'DATA',
                                          'sfs', '3d_sfs_no_name.fs'),
                             population_labels=['1', '2', '3']))
        yield ("dadi snp file",
               SFSDataHolder(os.path.join(EXAMPLE_FOLDER, 'DATA', 'sfs',
                                          'dadi_snp_file.txt')))
        yield ("vcf data",
               VCFDataHolder(
                    vcf_file=os.path.join(EXAMPLE_FOLDER, "DATA", "vcf", "out_of_africa_chr22_sim.vcf"),
                    popmap_file=os.path.join(EXAMPLE_FOLDER, "DATA", "vcf", "out_of_africa_chr22_sim_3pop.popmap")
               ))

    def test_add_split_after_inbreeding(self):
        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        f1 = FractionVariable('f1')
        Dyn2 = DynamicVariable('SudDyn')
        Tp = TimeVariable('Tp')

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1], dyn_args=[Dyn2])
        dm.add_inbreeding([f1])
        with pytest.raises(ValueError):
            dm.add_split(0, [nu1, nu2])

    def test_add_epoch_after_inbreeding(self):
        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        f1 = FractionVariable('f1')
        Dyn2 = DynamicVariable('SudDyn')
        T = TimeVariable('T')
        m = MigrationVariable('m')
        Tp = TimeVariable('Tp')

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1], dyn_args=[Dyn2])
        dm.add_inbreeding([f1])
        with pytest.raises(ValueError):
            dm.add_epoch(T, [nu1, nu2], [[None, m], [m, None]], ['Sud', 'Exp'])

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
        f1 = FractionVariable('f1')
        f2 = FractionVariable('f2')
        fxnu1 = Multiplication(f, nu1)
        tf = Multiplication(f, t)
        t_copy = copy.deepcopy(t)

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

        # create models with ancestral size
        Nanc = PopulationSizeVariable("Nanc", units="physical")
        Nu2 = PopulationSizeVariable("N2", units="physical")
        T3 = TimeVariable("T", units="physical")

        model4 = EpochDemographicModel(Nanc_size=Nanc)
        model4.add_epoch(t, [nu1])
        model4.add_split(0, [nu1, nu2])
        model4.add_epoch(t, [nu2, fxnu1], [[0, m], [0, 0]], [d1, d2])

        model5 = EpochDemographicModel(Nanc_size=Nanc)
        model5.add_epoch(t, [nu1])
        model5.add_split(0, [nu1, nu2])
        model5.add_epoch(T3, [0.5, fxnu1], [[0, m], [0, 0]], ["Sud", d2])
        model5.add_split(1, [nu2, nu1])
        model5.add_epoch(t, [nu1, nu2, nu1], None, None)

        model6 = EpochDemographicModel(Nanc_size=Nanc)
        model6.add_epoch(t, [nu1])
        model6.add_split(0, [nu1, nu2])
        model6.add_epoch(T3, [Nu2, fxnu1], [[0, m], [0, 0]], [d1, d2])
        model6.add_split(1, [nu2, nu1])
        model6.add_epoch(t, [nu1, nu2, nu1], None, None)
        model6.add_inbreeding([f1, f2, 0.5])

        values = {nu1: 2, nu2: 0.5, nu3: 0.5, t_copy: 0.3,
                  t2: 0.5, m: 0.1, s: 0.1, d1: 'Exp', d2: 'Lin',
                  f: 0.5, h: 0.3, f1: 0.1, f2: 0.3,
                  Nanc: 10000, Nu2: 5000, T3: 4000,
                  'N_1F':20000, 'r_2': -1e-5, 'N_2F': 5000, 'Tp': 500}

        for engine in [get_engine("momi")]:#all_available_engines():
            models = [model1, model2, model3, model5, model6]
            if engine.can_evaluate:
                customfile = os.path.join(
                    EXAMPLE_FOLDER, "MODELS",
                    f"demographic_model_{engine.id}_3pops.py")

                spec = importlib.util.spec_from_file_location(
                    f"module_{engine.id}", customfile)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[f'module_{engine.id}'] = module
                func = module.model_func
                par_labels = get_par_labels_from_file(customfile)
                variables = get_variables(par_labels, None, None, engine=engine.id)

                model7 = CustomDemographicModel(func, variables)
                models.append(model7)

            for ind, model in enumerate(models):
                for description, data in self._sfs_datasets():
                    msg = f"for model {ind + 1} and {description} data and " \
                          f"{engine.id} engine"
                    print(msg)
                    if engine.id == 'dadi':
                        options = {'pts': [4, 6, 8]}
                        args = (options['pts'],)
                    else:
                        options = {}
                        args = ()

                    if engine.id == "momi":
                        # there is an error in momi for that case
                        if description == "fs without pop labels":
                            continue
                        values['d2'] = "Exp"
                    data.sequence_length = 50818468
                    # we read data but save only updated data_holder
                    engine.data = data
                    engine.inner_data = None
                    engine.model = model
                    Nanc = None
                    if engine.id not in ["dadi", "moments"]:
                        Nanc = 10000

                    cmd = engine.generate_code(values, None, *args, nanc=Nanc)
                    # print(cmd)

                    if engine.can_evaluate:
                        if engine.id not in ['dadi', 'moments']:
                            model = copy.deepcopy(model)
                            model.Nanc_size = Nanc
                            engine.model = model
                        # read data
                        engine.data = data

                        true_ll = engine.evaluate(values, **options)

                        d = {}
                        exec(cmd, d)

                        msg += f": {true_ll} != {d['ll_model']}"
                        self.assertTrue(np.allclose(true_ll, d['ll_model']),
                                        msg=msg)
                        if (description == "dadi snp file" and
                                engine.id in ["dadi", "moments"]):
                            engine.data_holder.population_labels = None
                            self.assertRaises(ValueError, engine.generate_code,
                                              values, None, *args, nanc=Nanc)
                        if description == "fs without pop labels":
                            engine.data_holder.population_labels = None
                            engine.generate_code(values, None, *args, nanc=Nanc)
                    if engine.id == "momi":
                        values['d2'] = "Lin"


    def test_operation_eq(self):
        a = PopulationSizeVariable("a")
        b = PopulationSizeVariable("b")
        comb1 = operation_creation(Addition, a, b)
        comb2 = operation_creation(Addition, b, a)

        self.assertEqual(comb1, comb2)

        comb2 = operation_creation(Addition, a, b)
        self.assertEqual(comb1, comb2)
        c = TimeVariable("c")
        comb1 = Addition(c, Multiplication(a, b))
        comb2 = Addition(Multiplication(b, a), c)
        self.assertEqual(comb1, comb2)

        self.assertNotEqual(Division(a, b), Division(b, a))
        self.assertNotEqual(Subtraction(a, b), Subtraction(b, a))
        self.assertEqual(Addition(a, b), Addition(b, a))
        self.assertEqual(Multiplication(a, b), Multiplication(b, a))

        self.assertEqual(Exp(a), Exp(a))
        self.assertNotEqual(Exp(b), Exp(a))

        self.assertNotEqual(Log(b), Exp(b))
        self.assertNotEqual(Log(b), Multiplication(a, b))
        self.assertEqual(Exp(Multiplication(a, b)), Exp(Multiplication(b, a)))

    def test_operation_creation(self):
        self.test_creation_exp()
        self.test_creation_log()
        self.test_creation_add()
        self.test_creation_subtract()
        self.test_creation_multiplication()
        self.test_creation_division()

    def test_creation_exp(self):
        a = PopulationSizeVariable("a")
        b = PopulationSizeVariable("b")
        e1 = operation_creation(Exp, a)

        self.assertIsInstance(e1, Exp)
        log_b = Log(b)
        self.assertEqual(operation_creation(Exp, log_b), b)
        self.assertEqual(operation_creation(Exp, 0), 1)

        self.assertRaises(ValueError, operation_creation, Exp, 0, 0)

    def test_creation_log(self):
        a = PopulationSizeVariable("a")
        b = PopulationSizeVariable("b")
        l1 = operation_creation(Log, a)

        self.assertIsInstance(l1, Log)
        exp_b = Exp(b)
        self.assertEqual(operation_creation(Log, exp_b), b)
        self.assertEqual(operation_creation(Log, 1), 0)

        self.assertRaises(ValueError, operation_creation, Log, 0, 0)

    def test_creation_add(self):
        a = PopulationSizeVariable("a")
        b = PopulationSizeVariable("b")
        self.assertEqual(operation_creation(Addition, 0, 1), 1)

        self.assertEqual(operation_creation(Addition, a, 0), a)
        self.assertEqual(operation_creation(Addition, 0, b), b)

        self.assertIsInstance(operation_creation(Addition, a, 1), Addition)
        self.assertIsInstance(operation_creation(Addition, 1, b), Addition)
        self.assertIsInstance(operation_creation(Addition, a, b), Addition)
        self.assertRaises(ValueError, operation_creation, Addition, 0)

    def test_creation_subtract(self):
        a = PopulationSizeVariable("a")
        b = PopulationSizeVariable("b")
        self.assertEqual(operation_creation(Subtraction, 1, 0), 1)

        self.assertEqual(operation_creation(Subtraction, a, 0), a)
        # (1 + a * b) - ((b * a) + 1) = 0
        self.assertEqual(
            operation_creation(
                Subtraction,
                operation_creation(
                    Addition,
                    1,
                    operation_creation(
                        Multiplication,
                        a,
                        b
                    )
                ),
                operation_creation(
                    Addition,
                    operation_creation(
                        Multiplication,
                        b,
                        a
                    ),
                    1
                )
            ),
            0
        )

        self.assertRaises(ValueError, operation_creation, Subtraction, 0)

    def test_creation_multiplication(self):
        a = PopulationSizeVariable("a")
        b = PopulationSizeVariable("b")

        self.assertEqual(operation_creation(Multiplication, 2, 4), 8)

        self.assertEqual(operation_creation(Multiplication, 1, a), a)
        self.assertEqual(operation_creation(Multiplication, a, 1), a)

        self.assertEqual(operation_creation(Multiplication, 0, b), 0)
        self.assertEqual(operation_creation(Multiplication, b, 0), 0)

        self.assertIsInstance(operation_creation(Multiplication, a, b), Multiplication)

        self.assertRaises(ValueError, operation_creation, Multiplication, a)

    def test_creation_division(self):
        a = PopulationSizeVariable("a")
        b = PopulationSizeVariable("b")

        self.assertEqual(operation_creation(Division, 8, 2), 4)

        self.assertEqual(operation_creation(Division, a, 1), a)
        self.assertRaises(ValueError, operation_creation, Division, a, 0)

        self.assertEqual(operation_creation(Division, b, b), 1)

        self.assertRaises(ValueError, operation_creation, Division, a)

    def test_translation_from_epoch_to_tree1(self):
        from .test_tree_dem_model import TestTreeDemModel
        N_a, nu1, nu2, nu2F, t1, t2 = TestTreeDemModel.get_genetic_variables_model1()
        var2values = {
            'nu1': 0.4,
            'nu2F': 0.7,
            'nu2': 0.5,
            'N_a': 1e5,
            't1': 1,
            't2': 5
        }
        em = EpochDemographicModel(gen_time=29, Nanc_size=N_a, mutation_rate=1.25e-8)
        em.add_split(0, [nu1, nu2])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1, nu2F])

        cm = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        cm.add_leaf(0, size_pop=nu1)
        cm.add_leaf(1, size_pop=nu2F)
        cm.change_pop_size(1, t=t1, size_pop=nu2)
        cm.move_lineages(1, 0, t=t2, size_pop=N_a)

        translated_model, _ = em.translate_to(TreeDemographicModel, var2values)
        for event in translated_model.events:
            print(event)

        self.assertTrue(translated_model.equals(cm, var2values))

    @staticmethod
    def get_genetic_variables_model3():
        N_a = PopulationSizeVariable('N_a', units="physical")
        nu1B = PopulationSizeVariable('nu1B', units="genetic")
        nu1 = PopulationSizeVariable('nu1', units="genetic")
        nu1F = PopulationSizeVariable('nu1F', units="genetic")
        nu2B = PopulationSizeVariable('nu2B', units="genetic")
        nu2F = PopulationSizeVariable('nu2F', units="genetic")
        t1 = TimeVariable('t1', units="genetic")
        t2 = TimeVariable('t2', units="genetic")
        t3 = TimeVariable('t3', units="genetic")
        return N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3

    def test_translation_from_epoch_to_tree2(self):
        N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3 = self.get_genetic_variables_model3()
        var2values = {
            'N_a': 1e5,
            'nu1B': 0.4,
            'nu1': 0.8,
            'nu1F': 0.1,
            'nu2B': 0.5,
            'nu2F': 0.2,
            't1': 2,
            't2': 4,
            't3': 5
        }
        cm = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        cm.add_leaf(0, size_pop=nu1F)
        cm.add_leaf(1, size_pop=nu2F)
        cm.change_pop_size(0, t=t1, size_pop=nu1)
        cm.change_pop_size(0, t=t2, size_pop=nu1B)
        cm.change_pop_size(1, t=t2, size_pop=nu2B)
        cm.move_lineages(1, 0, t=t3, size_pop=N_a)

        em = EpochDemographicModel(gen_time=29, Nanc_size=N_a, mutation_rate=1.25e-8)
        em.add_split(0, [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t3, t2), [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2F])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1F, nu2F])
        translated_model, _ = em.translate_to(TreeDemographicModel, var2values)
        self.assertTrue(translated_model.equals(cm, var2values))

    def test_translation_from_epoch_to_tree3(self):
        N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3 = self.get_genetic_variables_model3()
        var2values = {
            'N_a': 1e5,
            'nu1B': 0.4,
            'nu1': 0.8,
            'nu1F': 0.1,
            'nu2B': 0.2,
            'nu2F': 0.5,
            't1': 2,
            't2': 4,
            't3': 5
        }
        cm = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        cm.add_leaf(0, size_pop=nu1F)
        cm.add_leaf(1, size_pop=nu2F)
        cm.change_pop_size(0, t=t1, size_pop=nu1)
        cm.change_pop_size(0, t=t2, size_pop=nu1B)
        cm.change_pop_size(1, t=t1, size_pop=nu2B)
        cm.move_lineages(1, 0, t=t3, size_pop=N_a)
        em = EpochDemographicModel(gen_time=29, Nanc_size=N_a, mutation_rate=1.25e-8)
        em.add_split(0, [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t3, t2), [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2B])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1F, nu2F])
        translated_model, _ = em.translate_to(TreeDemographicModel, var2values)
        self.assertTrue(translated_model.equals(cm, var2values))

    def test_translation_from_epoch_to_tree4(self):
        N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3 = self.get_genetic_variables_model3()
        t4 = TimeVariable('t4', units="genetic")
        var2values = {
            'N_a': 1e5,
            'nu1B': 0.4,
            'nu1': 0.8,
            'nu1F': 0.1,
            'nu2B': 0.5,
            'nu2F': 0.2,
            't1': 2,
            't2': 3,
            't3': 4,
            't4': 5
        }
        cm = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        cm.add_leaf(0, size_pop=nu1F)
        cm.add_leaf(1, size_pop=nu2F)
        cm.change_pop_size(0, t=t1, size_pop=nu1)
        cm.change_pop_size(0, t=t3, size_pop=nu1B)
        cm.change_pop_size(1, t=t2, size_pop=nu2B)
        cm.move_lineages(1, 0, t=t4, size_pop=N_a)
        em = EpochDemographicModel(gen_time=29, Nanc_size=N_a, mutation_rate=1.25e-8)
        em.add_split(0, [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t4, t3), [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t3, t2), [nu1, nu2B])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2F])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1F, nu2F])
        translated_model, _ = em.translate_to(TreeDemographicModel, var2values)
        self.assertTrue(translated_model.equals(cm, var2values))
