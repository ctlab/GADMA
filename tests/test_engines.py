import unittest
import gadma
from gadma import *
from gadma.engines import register_engine, Engine
from gadma.models import Model
from gadma.data import create_bed_files_and_extract_chromosomes
import os
import pickle
import numpy as np
from pathlib import Path
import shutil
import pytest


DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

POP_MAP = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "pop_map.txt")
REC_MAPS_DIR = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "rec_maps")
VCF_DATA_LD = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "vcf_data.vcf")
TEST_OUTPUT = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "test_output")
SAVE_IMAGE = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "ld_curves.jpg")

R_BINS = np.array(
    [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])

DATA_HOLDER_FOR_MODELS = VCFDataHolder(
            vcf_file=VCF_DATA_LD,
            popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
            population_labels=["deme0", "deme1"]
)


class TestEngines(unittest.TestCase):
    def tearDown(self):
        if Path("./plot.png").exists():
            os.remove("./plot.png")

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
        self.assertRaises(NotImplementedError, engine._read_data, "some")
        self.assertRaises(NotImplementedError, engine.evaluate, [])
        self.assertRaises(NotImplementedError, engine.generate_code, [])
        self.assertRaises(NotImplementedError,
                          engine.update_data_holder_with_inner_data)

        dadi_engine = get_engine('dadi')
        model = Model()
        self.assertRaises(ValueError, dadi_engine.set_model, model)
        self.assertRaises(ValueError,
                          dadi_engine.set_and_evaluate, [], None, None)

        model = EpochDemographicModel()
        self.assertRaises(ValueError,
                          dadi_engine.set_and_evaluate, [], model, None)

        engine.supported_models.append(EpochDemographicModel)
        engine.model = EpochDemographicModel()
        self.assertRaises(NotImplementedError, engine.get_N_ancestral, [])

    def model_3pop_dadi(self, ns, pts):
        import dadi
        def dadi_func(params, ns, pts):
            nu1, nu2, nu3, t1, t2, t3, m12 = params
            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)

            phi = dadi.Integration.one_pop(phi, xx, nu=nu1, T=t1)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / t2)
            phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu1,
                                            nu2=nu2_func)
            phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)

            phi = dadi.Integration.three_pops(phi, xx, T=t3, nu1=nu1,
                                              nu2=nu2, nu3=nu3, m12=m12)

            sfs = dadi.Spectrum.from_phi(phi, ns, [xx] * len(ns))
            return sfs

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        t1 = TimeVariable('t1', domain=[1e-3, 5])
        t2 = TimeVariable('t2', domain=[1e-3, 5])
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
            fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
            fs.integrate(Npop=lambda t: [nu1, nu2_func(t)], tf=tf, dt_fac=dt_fac)
            fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2] + ns[3])

            m = [[0, m12, 0], [0, 0, 0], [0, 0, 0]]
            fs.integrate(Npop=[nu1, nu2, nu3], tf=tf, m=m, dt_fac=dt_fac)
            fs = moments.Manips.split_3D_to_4D_3(fs, ns[2], ns[3])

            fs.integrate(Npop=[nu1, nu2, nu3, nu4], tf=tf, dt_fac=dt_fac)

            return fs

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        nu4 = PopulationSizeVariable('nu4')
        t = TimeVariable('t', domain=[1e-3, 5])
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

    def test_dadi_engine(self):
        engine = get_engine('dadi')

        ns = (4, 4, 4)
        pts = [10, 20, 30]

        data, dm, values, ll = self.model_3pop_dadi(ns, pts)
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]

        engine.set_data(data)
        engine.set_model(dm)
        seq_len = 1e6
        pops = None
        model = engine.simulate(vals, ns, seq_len, pops, pts)
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
        seq_len = 1e6
        pops = None
        model = engine.simulate(vals, ns, seq_len, pops, dt_fac)
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
        engine.data = engine.simulate(
            values=[1, 1],
            ns=(10,),
            sequence_length=1e6,
            population_labels=["Pop1"]
        )
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
        Tp = TimeVariable('Tp', domain=[1e-3, 5])
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
                  'm': 0.930, 'Tp': 0.363, 'T': 0.112, 'Dyn': 'Lin',
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
        T = TimeVariable('T', domain=[1e-3, 5])
        Dyn = DynamicVariable('Dyn')

        dm = EpochDemographicModel(Nanc_size=1000)
        dm.add_split(0, [nu, nu])
        dm.add_epoch(T, [nu, nu], dyn_args=['Sud', Dyn])

        values = {'nu': 1.880, 'T': 0.112, 'Dyn': 'Exp'}

        engine.data = engine.read_data(data_holder)
        assert engine.data_holder is None
        engine.model = dm

        engine.draw_schematic_model_plot(values, save_file="plot.png")
        engine.draw_data_comp_plot(values=values, save_file=None)

        dm = EpochDemographicModel()
        dm.add_epoch(T, [nu])
        engine.model = dm
        engine.data = engine.simulate(
            values=values,
            ns=[20],
            sequence_length=1e6,
            population_labels=["Pop1"]
        )
        engine.draw_data_comp_plot(values=values, save_file=None)

    def get_data_holder(self):
        # we take the most common data
        vcf_path = os.path.join(DATA_PATH, "DATA", "vcf")
        data_holder = VCFDataHolder(
            os.path.join(vcf_path, "out_of_africa_chr22_sim.vcf"),
            popmap_file=os.path.join(vcf_path, "out_of_africa_chr22_sim_3pop.popmap"),
            projections=[4, 4, 4],
            population_labels=["YRI", "CEU", "CHB"],
            sequence_length=50818468,  # chr22
        )
        return data_holder

    def get_model(self):
        dm = StructureDemographicModel(
            initial_structure=[2, 1, 1],
            final_structure=[2, 1, 1],
            has_anc_size=True,
            has_migs=False,
            has_sels=False,
            has_dyns=True,
            sym_migs=False,
            frac_split=False,
        )
        dm.mutation_rate = 1e-8
        dm.recombination_rate = 1e-8
        return dm

    def test_drawing_engines(self):
        data_holder = self.get_data_holder()
        dm = self.get_model()

        values = []
        for var in dm.variables:
            if isinstance(var, ContinuousVariable):
                if var.units == "physical":
                    values.append(10000)
                else:
                    values.append(np.random.uniform(0.5, 1.5))
            else:
                values.append(np.random.choice(["Sud", "Exp"]))

        for engine in all_available_engines():
            if not engine.can_draw_comp:
                continue
            if engine.id == "momentsLD":
                # drawing test for momentsLD is separate
                continue
            engine.model = dm
            engine.data = data_holder
            kwargs = {}
            if engine.id == "dadi":
                kwargs["pts"] = [5, 10, 15]  # pts

            engine.draw_data_comp_plot(values, **kwargs, save_file=None)

            if engine.can_simulate:
                engine.data = engine.simulate(
                    values=values,
                    ns=engine.data_holder.projections,
                    sequence_length=engine.data_holder.sequence_length,
                    population_labels=engine.data_holder.population_labels,
                    **kwargs,
                )
                engine.draw_data_comp_plot(values, **kwargs, save_file=None)

        for engine in all_available_engines():
            if not engine.can_draw_model:
                continue
            engine.model = dm
            engine.data_holder = data_holder
            engine.draw_schematic_model_plot(
                values=values,
                gen_time=10,
                gen_time_units='Years',
                save_file="plot.png"
            )

    def test_evaluating_engines(self):
        data_holder = self.get_data_holder()
        dm = self.get_model()

        values = [var.resample() for var in dm.variables]
        values = [el if el != "Lin" else "Exp" for el in values]

        for engine in all_available_engines():
            if engine.id == "momentsLD":
                # Evaluation test for momentsLD is separate
                continue
            kwargs = {}
            if engine.id == "dadi":
                kwargs["pts"] = [5, 10, 15]  # pts

            if engine.can_evaluate:
                engine.data = data_holder
                engine.model = dm
                engine.evaluate(values, **kwargs)


class TestModelSimulation(unittest.TestCase):

    def tearDown(self):
        if Path(f"{TEST_OUTPUT}/_bed_files/").exists():
            shutil.rmtree(f"{TEST_OUTPUT}/_bed_files/")

    def model_one_pop_moments_ld(self):
        import moments.LD

        def moments_one_pop_func(params, rho, theta):
            nu, tf = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)
            ld_stats.integrate(tf=tf, nu=[nu], rho=rho, theta=theta)

            return ld_stats

        nu = PopulationSizeVariable('nu')
        tf = TimeVariable('tf')
        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5

        values = {'nu': nu.resample(),
                  'tf': tf.resample()}

        data = moments_one_pop_func([values[x] for x in values], rhos, theta)
        data = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(data[:-2], data[1:-1])]
            + [data[-1]],
            num_pops=data.num_pops,
            pop_ids=data.pop_ids,
        )
        data = moments.LD.Inference.sigmaD2(data)

        dm = EpochDemographicModel()
        dm.add_epoch(tf, [nu])

        return values, data, dm

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_one_pop_moments_ld(self):
        engine = get_engine('momentsLD')
        values, data, dm = self.model_one_pop_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_data(data)
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        model = engine.simulate(vals)

        self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
                                                            'engine and momentsLD')
        self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
                                                            'engine and momentsLD')

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def model_two_pops_moments_ld(self):
        import moments.LD

        def moments_two_pops_func(params, rho, theta):
            nu1, nu2, tf = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)
            ld_stats.integrate(tf=tf, nu=[nu1], rho=rho, theta=theta)
            ld_stats = ld_stats.split(0)
            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
            ld_stats.integrate(tf=tf, nu=lambda t: [nu1, nu2_func(t)], rho=rho, theta=theta)

            return ld_stats

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        t = TimeVariable('t')
        dyn = DynamicVariable('dyn_exp')

        values = {'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  't': t.resample(),
                  'dyn_exp': 'Exp'}

        list_for_moments_ld = ['nu1', 'nu2', 't']
        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5
        data = moments_two_pops_func([values[x] for x in list_for_moments_ld], rhos, theta)
        data = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(data[:-2], data[1:-1])]
            + [data[-1]],
            num_pops=data.num_pops,
            pop_ids=data.pop_ids,
        )
        data = moments.LD.Inference.sigmaD2(data)

        dm = EpochDemographicModel()
        dm.add_epoch(t, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])

        return values, data, dm

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_two_pops_moments_ld(self):
        engine = get_engine('momentsLD')
        values, data, dm = self.model_two_pops_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        model = engine.simulate(vals)
        self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
                                                            'engine and momentsLD')
        self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
                                                            'engine and momentsLD')

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def model_4pops_moments_ld(self):
        import moments.LD

        def moments_ld_func(params, rho, theta):
            nu1, nu2, nu3, nu4, tf, m12 = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)

            ld_stats.integrate(tf=tf, nu=[nu1], rho=rho, theta=theta)
            ld_stats = ld_stats.split(0)

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
            ld_stats.integrate(nu=lambda t: [nu1, nu2_func(t)], tf=tf, rho=rho, theta=theta)

            ld_stats = ld_stats.split(1)
            m = [[0, m12, 0], [0, 0, 0], [0, 0, 0]]
            ld_stats.integrate(nu=[nu1, nu2, nu3], tf=tf, m=m, rho=rho, theta=theta)

            ld_stats = ld_stats.split(2)

            ld_stats.integrate(nu=[nu1, nu2, nu3, nu4], tf=tf, rho=rho, theta=theta)
            return ld_stats

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
        list_for_moments_ld = ['nu1', 'nu2', 'nu3', 'nu4', 't', 'm12']

        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5

        simulated = moments_ld_func([values[x] for x in list_for_moments_ld], rhos, theta)
        simulated = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(simulated[:-2], simulated[1:-1])]
            + [simulated[-1]],
            num_pops=simulated.num_pops,
            pop_ids=simulated.pop_ids,
        )
        simulated = moments.LD.Inference.sigmaD2(simulated)

        dm = EpochDemographicModel()
        dm.add_epoch(t, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])
        dm.add_split(1, [nu2, nu3])
        dm.add_epoch(t, [nu1, nu2, nu3], mig_args=[[0, m12, 0], [0, 0, 0], [0, 0, 0]])
        dm.add_split(2, [nu3, nu4])
        dm.add_epoch(t, [nu1, nu2, nu3, nu4])

        return values, simulated, dm

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_4pops_moments_ld(self):
        values, simulated_by_moments, dm = self.model_4pops_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine = get_engine('momentsLD')
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        simulated_by_gadma = engine.simulate(vals)
        self.assertTrue(np.allclose(
            simulated_by_gadma[0],
            simulated_by_moments[0],
        ),
            msg='Simulations differs in '
                'engine and momentsLD')
        self.assertTrue(np.allclose(
            simulated_by_gadma[1],
            simulated_by_moments[1],
        ),
            msg='Simulations differs in '
                'engine and momentsLD')


class TestModelEvaluation(unittest.TestCase):

    def tearDown(self):
        moments.LD.Inference._varcov_inv_cache = {}
        if Path(f"{TEST_OUTPUT}/").exists():
            shutil.rmtree(f"{TEST_OUTPUT}/")
        os.mkdir(TEST_OUTPUT)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def model_for_gadma_moments_evaluation(self):
        def model_moments_ld(params, rho, theta):
            nu, nu1, nu2, tf, m12, m21 = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)

            ld_stats.integrate(tf=tf, nu=[nu], rho=rho, theta=theta)
            ld_stats = ld_stats.split(0)

            nu2_func = lambda t: nu * (nu2 / nu) ** (t / tf)
            nu1_func = lambda t: nu * (nu1 / nu) ** (t / tf)

            m = [[0, m12], [m21, 0]]
            ld_stats.integrate(nu=lambda t: [nu1_func(t), nu2_func(t)], m=m, tf=tf, rho=rho, theta=theta)

            return ld_stats

        nu = PopulationSizeVariable('nu')
        tf = TimeVariable('tf')
        m12 = MigrationVariable('m12')
        m21 = MigrationVariable('m21')
        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        dyn1 = DynamicVariable('dyn1')
        dyn2 = DynamicVariable('dyn2')

        dm = EpochDemographicModel()
        dm.add_epoch(tf, [nu])
        dm.add_split(0, [nu, nu])
        dm.add_epoch(
            tf, [nu1, nu2],
            dyn_args=[dyn1, dyn2],
            mig_args=[[0, m12], [m21, 0]])

        values = {'nu': nu.resample(),
                  'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  'tf': tf.resample(),
                  'm12': m12.resample(),
                  'm21': m21.resample(),
                  'dyn1': 'Exp',
                  'dyn2': 'Exp'}
        list_for_moments_ld = ['nu', 'nu1', 'nu2', 'tf', 'm12', 'm21']

        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5

        simulated = model_moments_ld([values[x] for x in list_for_moments_ld], rhos, theta)
        simulated = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(
                simulated[:-2],
                simulated[1:-1])] + [simulated[-1]],
            num_pops=simulated.num_pops,
            pop_ids=simulated.pop_ids,
        )
        simulated = moments.LD.Inference.sigmaD2(simulated)

        return values, simulated, dm

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    @pytest.mark.timeout(0)
    def test_evaluation_ld(self):
        # use data read with gadma
        engine = get_engine('momentsLD')

        preprocessed_test_data = os.path.join(
            DATA_PATH, 'DATA', 'vcf_ld',
            'preprocessed_data.bp'
        )
        with open(preprocessed_test_data, 'rb') as file:
            region_stats_moments_ld = pickle.load(file)
        engine.set_data(region_stats_moments_ld)
        self.assertRaises(AttributeError, engine.generate_code, [], "filename")

        engine.data_holder = VCFDataHolder(
            vcf_file=VCF_DATA_LD, popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
            population_labels=["deme0", "deme1"],
            sequence_length=1000000
        )
        # create bed files
        bed_files_dir = os.path.join(TEST_OUTPUT, "_bed_files")
        create_bed_files_and_extract_chromosomes(
            data_holder=engine.data_holder,
            output_dir=bed_files_dir
        )
        engine.data_holder.bed_files_dir = bed_files_dir
        engine.ld_kwargs = {"cM": True, "report": "False"}
        engine.set_data(engine.data_holder)
        data_gadma = engine.data
        data_moments = engine.data
        self.assertEqual(len(data_gadma), len(data_moments))
        for arr in range(len(data_moments['means'])):
            self.assertTrue(all(((a == b) | (np.isnan(a) & np.isnan(b))) for a, b in zip(
                data_moments['means'][arr], data_gadma['means'][arr]
            )))

        # simulate ld_stats with moments
        values, simulated_by_moments, dm = self.model_for_gadma_moments_evaluation()
        # check simulation ld_stats with GADMA
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        simulated_by_gadma = engine.simulate(values=vals)
        for ii in range(len(simulated_by_gadma)):
            self.assertTrue(np.allclose(
                simulated_by_gadma[ii],
                simulated_by_moments[ii],
            ),
                msg='Simulations differs in '
                    'engine and momentsLD')

        means, varcovs = moments.LD.Inference.remove_normalized_data(
            data_moments["means"],
            data_moments["varcovs"],
            num_pops=simulated_by_moments.num_pops)
        simulated_by_moments = moments.LD.Inference.remove_normalized_lds(
            simulated_by_moments, normalization=0)
        ll_moments = moments.LD.Inference.ll_over_bins(
            means,
            simulated_by_moments,
            varcovs
        )
        ll_gadma = engine.evaluate(vals)

        self.assertEqual(ll_gadma, ll_moments)

        theta = 4 * 10000 * 6.0e-5
        self.assertEqual(theta, engine.get_theta(vals))
        self.assertEqual(10000, engine.get_N_ancestral_from_theta(theta))

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    @pytest.mark.timeout(0)
    def test_draw_curves(self):
        preprocessed_test_data = os.path.join(
            DATA_PATH, 'DATA', 'vcf_ld',
            'preprocessed_data_YRI_CEU.bp'
        )
        engine = get_engine('momentsLD')
        engine.data_holder = VCFDataHolder(
            vcf_file=VCF_DATA_LD, popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
            population_labels=["deme0", "deme1"],
            preprocessed_data=preprocessed_test_data
        )

        values, simulated_by_moments, dm = self.model_for_gadma_moments_evaluation()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        engine.inner_data = engine._read_data(engine.data_holder)

        engine.draw_data_comp_plot(values=vals, save_file=SAVE_IMAGE,)
