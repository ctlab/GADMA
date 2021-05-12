import unittest
from gadma import *
from gadma.engines import register_engine, Engine
from gadma.models import Model
import os
import numpy as np
import jpype
import multiprocessing
from multiprocessing import Queue, Process
from gadma.engines import DiCal2Engine

from .test_data import CONTIG0, CONTIG0_POPMAP, REFERENCE

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

def evaluate_with_dical2_engine(q, er_q, data_holder, model, vals):
    try:
        engine = get_engine("diCal2")
        engine.data = engine.read_data(data_holder)
        engine.model = model
        q.put(engine.evaluate(vals))
    except Exception as e:
        er_q.put(e)
        raise e

def evaluate_with_dical2_cmd(q, er_q):
    try:
        # run JVM
        DiCal2Engine._startJVM()
        dical_pkg = DiCal2Engine.base_module
        def path(p):
            return  os.path.join(DATA_PATH, "MODELS", "dical2_models", p)
        def data_path(p):
            return os.path.join(DATA_PATH, "DATA", 'vcf', p)
        dical_args = ['--paramFile', path('mutRec.param'),
                      '--vcfFile', data_path('contig.0.vcf'),
                      '--vcfFilterPassString', 'PASS',
                      '--vcfReferenceFile', data_path('reference.fa'),
                      '--lociPerHmmStep', '32000',
                      '--configFile', path('2pop.config'),
                      '--demoFile', path('2pop.demo'),
                      '--intervalType', 'loguniform',
                      '--compositeLikelihood', 'lol',
                      '--intervalParams', '8,0.01,4',
    #                      '--startPoint', '0.2,0.5,0.5,0.02,1',
                      '--bounds', '0.002,20;0.01,20;0.01,20;0.01,20;0.01,20',
    #                      '--numberIterationsEM', '0',
    #                      '--numberIterationsMstep', '1',
    #                      '--verbose'
                        ]
        dical_param_set_class = dical_pkg.maximum_likelihood.DiCalParamSet
        dical_param_set = dical_param_set_class(dical_args, [], [],
                                                jpype.java.lang.System.out)
        chunk = 0
        extended_config = dical_param_set.extendedConfigInfoList.get(chunk)
        structuredConfig = extended_config.structuredConfig
        pSet = extended_config.pSet;
        fancyTransitionMap = extended_config.fancyTransitionMap
        csdConfigs = jpype.java.util.ArrayList()
        csdConfigs.add(jpype.java.util.ArrayList())
        csdConfigs.get(0).add(jpype.java.util.ArrayList())
        em_module = dical_pkg.maximum_likelihood.StructureEstimationEM
        csdConfigs.get(0).get(0).addAll(
            em_module.getLOLConfigList(structuredConfig,
                                       pSet,
                                       fancyTransitionMap,
                                       False)
        )

        # Create objective
        objectiveFunction = em_module.DiCalObjectiveFunction(
            csdConfigs,
            dical_param_set.demoFactory,
            dical_param_set.demoStateFactory,
            dical_param_set.conditionalObjectiveFunction,
            [0.2,0.5,0.5,0.02,1],
            False,
            dical_param_set.trunkFactory,
            False,
            csdConfigs.get(0).get(0).get(0).pSet.getMutationRate(0),
            0,
            False,
            True,
            dical_param_set.useEigenCore
        )
        cmd_ll = objectiveFunction.getLogLikelihood()
        q.put(float(cmd_ll))
    except Exception as e:
        er_q.put(e)
        raise e


def func_in_separate_process(function, *args):
    multiprocessing.set_start_method('spawn', force=True)
    queue = Queue()
    er_queue = Queue()
    p = Process(target=function,
                args=(queue, er_queue, *args))
    p.start()
    p.join() # this blocks until the process terminates
    if not er_queue.empty():
        raise RuntimeError() from er_queue.get(0)
    if not queue.empty():
        return queue.get(0)


class TestEngines(unittest.TestCase):
    def test_existence(self):
        self.assertTrue(len(list(all_engines())) > 0)
        for engine in all_engines():
            options = {}
            if engine.id == 'dadi':
                options['pts'] = [10, 20, 30]
            with self.assertRaises(ValueError):
                engine.evaluate([], **options)

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
        self.assertEqual(engine.evaluate(vals, pts), ll)

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
        eng_ll = engine.evaluate(vals, dt_fac)
        self.assertTrue(np.allclose(eng_ll, ll),
                        msg=f"{eng_ll} != {ll}")

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

    def test_dical2_engine_models_reading(self):
        from jpype import java
        from jpype import JDouble
        from java.io import FileReader
        #load model from file and compare it with created
        dical2_engine = engines.DiCal2Engine()
        model_file = os.path.join(DATA_PATH, "MODELS",
                                  "dical2_models", "2pop_no_vars.demo")

        with open(model_file) as f:
            correct_string = "".join(f.readlines())

        # create it with our engine
        # variables
        Nanc = PopulationSizeVariable("Nanc", units="physical")
        N1 = PopulationSizeVariable("N1", units="physical")
        N2 = PopulationSizeVariable("N2", units="genetic")
        T = TimeVariable("T", units="physical")
        m12 = MigrationVariable("m12", units="physical")
        model = EpochDemographicModel(has_anc_size=True, Nanc_size=Nanc)
        model.Nref = 10000
        model.add_split(pop_to_div=0, size_args=[N1, N2])
        migs = [[0, m12],[0.015, 0]]
        model.add_epoch(time_arg=T, size_args=[N1, N2], mig_args=migs)
        values={"T": 4000, "Nanc": 10000, "N1": 10000, "N2": 0.5, "m12": 1.25e-6}
        # model
        dical2_engine.model = model
        # demo model
        demo_string = dical2_engine._get_string_of_model(values)
        self.assertEqual(correct_string, demo_string)

    def test_dical2_engine_evaluation(self):

        dical2_engine = get_engine("diCal2")

        Nanc = PopulationSizeVariable("Nanc", units="physical")
        N1 = PopulationSizeVariable("N1", units="physical")
        N2 = PopulationSizeVariable("N2", units="genetic")
        T = TimeVariable("T", units="physical")
        m12 = MigrationVariable("m12", units="physical")

        model = EpochDemographicModel(has_anc_size=True, Nanc_size=Nanc)
        model.Nref = 10000
        model.mu = 1.25e-8
        model.add_split(pop_to_div=0, size_args=[N1, N2])
        migs = [[0, m12],[0, 0]]
        model.add_epoch(time_arg=T, size_args=[N1, N2], mig_args=migs)

        #0.2,0.5,0.5,0.02,1
        values={"T": 4000, "Nanc": 10000, "N1": 5000, "N2": 0.5, "m12": 5e-7}

        # data
        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP,
                             reference_file=REFERENCE)

        # we have to create new process to get valid performance with JVM
        ll = func_in_separate_process(evaluate_with_dical2_engine,
                                      data, model, values)
        # evaluate with dical2
        cmd_ll = func_in_separate_process(evaluate_with_dical2_cmd)

        self.assertEqual(ll, cmd_ll)
