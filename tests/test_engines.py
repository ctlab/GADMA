import unittest
import gadma
from gadma import *
from gadma.engines import register_engine, Engine
from gadma.models import Model
import os
import numpy as np


DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


class TestEngines(unittest.TestCase):
    def test_existence(self):
        self.assertTrue(len(list(all_engines())) > 0)
        self.assertTrue(len(list(all_available_engines())) > 0)
        self.assertTrue(len(list(all_simulation_engines())) > 0)
        self.assertTrue(len(list(all_drawing_engines())) > 0)
        for engine in all_engines():
            options = {}
            if engine.id == 'dadi':
                options['pts'] = [10, 20, 30]
            if engine.can_evaluate:
                self.assertRaises(ValueError, engine.evaluate, [], **options)
            else:
                self.assertRaises(NotImplementedError,
                                  engine.evaluate, [], **options)

        self.assertRaises(ValueError, register_engine,
                          get_engine("dadi"))
        self.assertRaises(ValueError, get_engine, "some_strange_name")

        engine = Engine()
        self.assertRaises(NotImplementedError, engine.read_data, "some")
        self.assertRaises(NotImplementedError, engine.evaluate, [])
        self.assertRaises(NotImplementedError, engine.generate_code, [])

        dadi_engine = get_engine('dadi')
        model = Model()
        self.assertRaises(ValueError, dadi_engine.set_model, model)
        self.assertRaises(ValueError,
                          dadi_engine.set_and_evaluate, [], None, None)

        model = EpochDemographicModel()
        self.assertRaises(ValueError,
                          dadi_engine.set_and_evaluate, [], model, None)

    def model_3pop_dadi(self, ns, pts):
        import dadi
        def dadi_func(params, ns, pts):
            nu1, nu2, nu3, t1, t2, t3, m12 = params
            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)

            phi = dadi.Integration.one_pop(phi, xx, nu=nu1, T=t1)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t/t2)
            phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu1,
                                            nu2=nu2_func)
            phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)

            phi = dadi.Integration.three_pops(phi, xx, T=t3, nu1=nu1,
                                            nu2=nu2, nu3=nu3, m12=m12)

            sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
            return sfs

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        t1 = TimeVariable('t1')
        t2 = TimeVariable('t2')
        t3 = TimeVariable('t3')
        m12 = MigrationVariable('m12')
        dyn = DynamicVariable('dyn_exp')

        values = {'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  'nu3': nu3.resample(),
                  't1': t1.resample(),
                  't2': t2.resample(),
                  't3': t3.resample(),
                  'm12': m12.resample(),
                  'dyn_exp': 'Exp'}
        list_in_dadi = ['nu1', 'nu2', 'nu3', 't1', 't2', 't3', 'm12']

        func_ex = dadi.Numerics.make_extrap_log_func(dadi_func)
        data = func_ex([values[x] for x in list_in_dadi], ns, pts)
        ll = dadi.Inference.ll_multinom(data, data)

        dm = EpochDemographicModel()
        dm.add_epoch(t1, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t2, [nu1, nu2], dyn_args=['Sud', dyn])
        dm.add_split(0, [nu1, nu3])
        dm.add_epoch(t3, [nu1, nu2, nu3], mig_args=[[0, m12, 0], [0, 0, 0], [0, 0, 0]])

        return data, dm, values, ll

    def model_4pop_moments(self, ns, dt_fac):
        import moments
        def moments_func(params, ns, dt_fac):
            nu1, nu2, nu3, nu4, tf, m12 = params

            sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
            fs = moments.Spectrum(sts)

            fs.integrate(Npop=[nu1], tf=tf, dt_fac=dt_fac)
            fs =  moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t/tf)
            fs.integrate(Npop=lambda t: [nu1, nu2_func(t)], tf=tf, dt_fac=dt_fac)
            fs =  moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2] + ns[3])

            m = [[0, m12, 0], [0, 0, 0], [0, 0, 0]]
            fs.integrate(Npop=[nu1, nu2, nu3], tf=tf, m=m, dt_fac=dt_fac)
            fs =  moments.Manips.split_3D_to_4D_3(fs, ns[2], ns[3])

            fs.integrate(Npop=[nu1, nu2, nu3, nu4], tf=tf, dt_fac=dt_fac)

            return fs

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        nu4 = PopulationSizeVariable('nu4')
        t = TimeVariable('t')
        m12 = MigrationVariable('m12')
        dyn = DynamicVariable('dyn_exp')

        values = {'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  'nu3': nu3.resample(),
                  'nu4': nu4.resample(),
                  't': t.resample(),
                  'm12': m12.resample(),
                  'dyn_exp': 'Exp'}
        list_in_dadi = ['nu1', 'nu2', 'nu3', 'nu4', 't', 'm12']

        data = moments_func([values[x] for x in list_in_dadi], ns, dt_fac)
        ll = moments.Inference.ll_multinom(data, data)

        dm = EpochDemographicModel()
        dm.add_epoch(t, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])
        dm.add_split(1, [nu2, nu3])
        dm.add_epoch(t, [nu1, nu2, nu3], mig_args=[[0, m12, 0], [0, 0, 0], [0, 0, 0]])
        dm.add_split(2, [nu3, nu4])
        dm.add_epoch(t, [nu1, nu2, nu3, nu4])
        return data, dm, values, ll


    def test_dadi_engime(self):
        engine = get_engine('dadi')

        ns = (4, 4, 4)
        pts = [10, 20, 30]

        data, dm, values, ll = self.model_3pop_dadi(ns, pts)
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]

        engine.set_data(data)
        engine.set_model(dm)
        model = engine.simulate(vals, ns, pts)
        self.assertTrue(np.allclose(model, data), msg="Simulations differs in"
                                                      " engine and in dadi.")
        self.assertEqual(engine.set_and_evaluate(
            data=data,
            model=dm,
            values=vals,
            options={"pts": pts}
        ), ll)

    def test_moments_engine(self):
        engine = get_engine('moments')

        ns = (4, 4, 4, 4)
        dt_fac = 0.1

        data, dm, values, ll = self.model_4pop_moments(ns, dt_fac)
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]

        engine.set_data(data)
        engine.set_model(dm)
        model = engine.simulate(vals, ns, dt_fac)
        self.assertTrue(np.allclose(model, data), msg="Simulations differs in"
                                                      " engine and in dadi.")
        eng_ll = engine.set_and_evaluate(
            data=data,
            model=dm,
            values=vals,
            options={"dt_fac": dt_fac}
        )
        self.assertTrue(np.allclose(eng_ll, ll),
                        msg=f"{eng_ll} != {ll}")
        # check that multinom and constrain lead to error
        dm = EpochDemographicModel(Nanc_size=10000)
        dm.add_epoch(TimeVariable("t"), [PopulationSizeVariable("nu")])
        engine.model = dm
        engine.data = engine.simulate(values=[1, 1], ns=(10,))
        engine.model.linear_constrain = optimizers.LinearConstrain([[0, 1], [1, -1]], [0, 0], [1, 1])
        self.assertRaises(ValueError, engine.evaluate, values=[1, 1])

    def test_upper_bounds_on_split_time(self):
        import dadi
        settings = SettingsStorage()
        settings.initial_structure = [2, 1, 1]
        settings.upper_bound_of_first_split = 100
        settings.upper_bound_of_second_split = 50

        data = dadi.Spectrum.from_file(os.path.join(DATA_PATH, "DATA",
                                                    "sfs", "3d_sfs.fs"))
        engine = get_engine("moments")
        engine.set_data(data)
        engine.set_model(settings.get_model())

        self.assertRaises(AttributeError, engine.generate_code, [], "filename")

        values = [var.resample() for var in engine.model.variables]
        engine.get_theta(values)

        engine.model.linear_constrain.lb = [51, -1]
        engine.set_model(engine.model)
        self.assertRaises(ValueError, engine.get_theta, values)

    def test_demes_engine(self):
        data_holder = DataHolder(
            filename=None,
            projections=[10, 10, 10],
            outgroup=False,
            population_labels=["YRI", "CEU", "CHB"],
            sequence_length=None
        )

        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        m13 = MigrationVariable('m13')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')

        dm = EpochDemographicModel(Nanc_size=1000)
        dm.add_epoch(T, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(Tp, [nu1F, nu2F], dyn_args=['Sud', Dyn])
        dm.add_split(1, [nu2F, nu2F])
        migs = [[0, m, m13], [m, 0, 0], [0, 0, 0]]
        dm.add_epoch(T, [nu1F, nu2F, nu2F], mig_args=migs)

        values = {'nu1F': 1.880, nu2B: 0.0724, 'f': 0.9, 'nu2F': 1.764,
               'm': 0.930, 'Tp':  0.363, 'T': 0.112, 'Dyn': 'Lin',
               'SudDyn': 'Sud', 's': 0.1, 'dom': 0.5, 'm13': 1.5}

        engine = gadma.engines.demes_engine.DemesEngine()
        engine.data = data_holder
        engine.model = dm

        engine.generate_code(values=values)
        engine.draw_schematic_model_plot(values, save_file="plot.png")

        engine.generate_code(values=values,
                             gen_time=25, gen_time_units="years")
        engine.draw_schematic_model_plot(values, save_file="plot.png",
                                         gen_time=25, gen_time_units="years")

        dm = EpochDemographicModel(Nanc_size=1000)
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(Tp, [nu1F, nu2F], dyn_args=['Sud', Dyn])
        dm.add_split(1, [nu2F, nu2F])
        migs = [[0, m, m13], [m, 0, 0], [0, 0, 0]]
        dm.add_epoch(T, [nu1F, nu2F, nu2F], mig_args=migs)

        engine.model = dm
        engine.generate_code(values=values,
                             gen_time=25, gen_time_units="years")
        engine.draw_schematic_model_plot(values, save_file="plot.png",
                                         gen_time=25, gen_time_units="years")
        engine.draw_schematic_model_plot(values, save_file=None,
                                         gen_time=25, gen_time_units="years")

        # error when no nanc is set
        dm = EpochDemographicModel()
        engine.model = dm
        self.assertRaises(ValueError, engine.build_demes_graph,
                          values=values, nanc=None)

    def test_moments_drawing(self):
        data_holder = SFSDataHolder(
            os.path.join(DATA_PATH, "DATA", "sfs", "YRI_CEU.fs"),
        )

        engine = get_engine("moments")

        nu = PopulationSizeVariable('nu')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')

        dm = EpochDemographicModel(Nanc_size=1000)
        dm.add_split(0, [nu, nu])
        dm.add_epoch(T, [nu, nu], dyn_args=['Sud', Dyn])

        values = {'nu': 1.880, 'T': 0.112, 'Dyn': 'Exp'}

        engine.data = engine.read_data(data_holder)
        assert engine.data_holder is None
        engine.model = dm

        engine.draw_schematic_model_plot(values, save_file="plot.png")
        engine.draw_sfs_plots(values=values, grid_sizes=0.01, save_file=None)

        dm = EpochDemographicModel()
        dm.add_epoch(T, [nu])
        engine.model = dm
        engine.data = engine.simulate(values=values, ns=[20])
        engine.draw_sfs_plots(values=values, grid_sizes=0.01, save_file=None)
