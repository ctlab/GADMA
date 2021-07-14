import unittest
import os
import numpy as np
import gadma
from gadma import *
import jpype
import copy

DATA_PATH = os.path.join("tests", "test_data")
from .test_data import VCF_DATA, POPMAP
from .test_data import CONTIG0, CONTIG0_POPMAP, REFERENCE, SMALL_REFERENCE
CONTIG0_BED = os.path.join(DATA_PATH, "DATA", "vcf", "contig0_bed_file.bed")

class TestDiCal2(unittest.TestCase):
    def _run_dical2_cmd(self):
        dical_pkg = engines.DiCal2Engine.base_module
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
                      '--demoFile', path('2pop_2_2.demo'),
                      '--intervalType', 'loguniform',
                      '--compositeLikelihood', 'lol',
                      '--intervalParams', '8,0.01,4',
    #                      '--startPoint', '0.2,0.5,0.5,0.02,1',
                      '--bounds', '0.002,20;0.002,20;0.002,20;0.01,20;0.01,20;0.01,20;0.01,20',
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
            [0.2,0.4,0.6,0.5,0.5,0.02,1],
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
        return cmd_ll

    def test_dical2_engine_models_reading(self):
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

    def test_diCal2_engine(self):
        engine = get_engine("diCal2")

        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP,
                             reference_file=REFERENCE)
        # 0. set model and change it before JAVA is launched
        dm1 = EpochDemographicModel(Nanc_size=1000)
        dm1.mutation_rate = 1e-8
        dm1.recombination_rate = 1e-8
        engine.model = dm1
        dm2 = EpochDemographicModel(Nanc_size=100)
        dm2.mutation_rate = 1e-10
        dm2.recombination_rate = 1e-8
        engine.model = dm2
        engine.model = dm2

        # 1. read data
        engine.data_holder = data
        engine.data = gadma.engines.DiCal2Engine.read_data(data)
        assert isinstance(engine.data, tuple)
        assert len(engine.data) == 2

        # 2. dical2_engine evaluations
        # variables
        Nanc = PopulationSizeVariable("Nanc", units="physical")
        N1 = PopulationSizeVariable("N1", units="physical")
        N2 = PopulationSizeVariable("N2", units="genetic")
        T = TimeVariable("T", units="physical")
        m12 = MigrationVariable("m12", units="physical")
        # model
        model = EpochDemographicModel(has_anc_size=True, Nanc_size=Nanc)
        model.Nref = 10000
        model.mutation_rate = 1.25e-8
        model.recombination_rate = 1e-8
        model.add_epoch(time_arg=T, size_args=[N1])
        model.add_split(pop_to_div=0, size_args=[N1, N2])
        migs = [[0, m12],[0, 0]]
        model.add_epoch(time_arg=T, size_args=[N1, N2], mig_args=migs)
        model.add_epoch(time_arg=T, size_args=[N2, N1], mig_args=migs)
        engine.model = model
        # values 0.2,0.5,0.5,0.02,1
        values={"T": 4000, "Nanc": 10000, "N1": 5000, "N2": 0.5, "m12": 5e-7}
        # we have to create new process to get valid performance with JVM
        ll = engine.evaluate(values)
        # evaluate with dical2 cmd
        cmd_ll = self._run_dical2_cmd()
        # check
        self.assertEqual(ll, cmd_ll)

        # 3. Set model with same and different mu
        engine.model = model
        model = copy.deepcopy(model)
        model.mutation_rate = 1e-5
        model.recombination_rate = 1e-8
        engine.model = model

        # 4. fails
        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP)
        self.assertRaises(ValueError,
                          gadma.engines.DiCal2Engine.read_data, data)
        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP,
                             sequence_length=10, reference_file=REFERENCE)
        self.assertRaises(AssertionError,
                          gadma.engines.DiCal2Engine.read_data, data)
        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP,
                             reference_file=SMALL_REFERENCE)
        self.assertRaises(ValueError,
                          gadma.engines.DiCal2Engine.read_data, data)

        # 5. Empty model
        model = EpochDemographicModel(Nanc_size=100)
        model.mutation_rate = 1e-8
        model.recombination_rate = 1e-8
        model.Nref = 100
        engine.model = model
        engine._get_string_of_model(values=[])

        # end of test
        engine._stopJVM()
