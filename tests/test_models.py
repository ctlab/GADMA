import unittest

from gadma import *

from .test_data import YRI_CEU_DATA

try:
    import dadi
    DADI_NOT_AVAILABLE = False
except ImportError:
    DADI_NOT_AVAILABLE = True

import numpy as np


class TestModels(unittest.TestCase):
    def dadi_wrapper(self, func):
        def wrapper(param, ns, pts):
            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)
            phi = func(param, xx, phi)
            sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
            return sfs
        return wrapper

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_1pop_0(self):
        @self.dadi_wrapper
        def inner(param, xx, phi):
            return phi

        ns = (20,)
        pts = [40, 50, 60]
        func_ex = dadi.Numerics.make_extrap_log_func(inner)
        real = func_ex([], ns, pts)

        dm = DemographicModel()
        pts = [40, 50, 60]
        d = get_engine('dadi')
        d.set_model(dm)
        got = d.simulate([], ns, pts)
        self.assertTrue(np.allclose(got, real))

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_1pop_(self):
        @self.dadi_wrapper
        def inner(param, xx, phi):
            T, nu = param
            phi = dadi.Integration.one_pop(phi, xx, T=T, nu=nu)
            return phi

        ns = (20,)
        pts = [40, 50, 60]
        param = [1., 0.5]
        func_ex = dadi.Numerics.make_extrap_log_func(inner)
        real = func_ex(param, ns, pts)

        T = TimeVariable('T1')
        nu = PopulationSizeVariable('nu2')
        dm = DemographicModel()
        dm.add_epoch(T, [nu])
        d = get_engine('dadi')
        d.set_model(dm)
        got = d.simulate(param, ns, pts)
        self.assertTrue(np.allclose(got, real))

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_gut_2pop(self):
        """
        Check loglikelihood of the demographic model from the YRI_CEU
        example of dadi.
        """
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')

        dm = DemographicModel()
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

    def test_fix_vars(self):
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')

        dm = DemographicModel()
        dm.add_epoch(Tp, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', Dyn])

        dic = {'nu1F': 1.880, 'nu2B': 0.0724, 'nu2F': 1.764, 'm': 0.930,
               'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp'}

        data = SFSDataHolder(YRI_CEU_DATA)
        d = DadiEngine(model=dm, data=data)
        values = [dic[var.name] for var in dm.variables]
        ll1 = d.evaluate(values, pts=[40, 50, 60])
        dm.fix_variable(Dyn, 'Exp')
        values = [dic[var.name] for var in dm.variables]
        d.model = dm
        ll2 = d.evaluate(dic, pts=[40, 50, 60])
        self.assertTrue(ll1, ll2)
