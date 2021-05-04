from . import Engine
from ..models import DemographicModel, StructureDemographicModel,\
                     CustomDemographicModel
from ..utils import DiscreteVariable, cache_func
from ..utils import read_popinfo, get_list_of_names_from_vcf
from .. import VCFDataHolder
from .. import dadi_available, moments_available
from ..code_generator import id2printfunc

import warnings
import os
import numpy as np
from functools import wraps

import jpype
import jpype.imports
from jpype.types import *
from ..dical2_path import dical2_path


# Interface above dical2 functions that are used in DiCal2Engine
# all of them has dical2_pkg as first argument to use it as base package

def _dical2_config_info(
    dical2_pkg,
    multiplicities,
    numDemes,
    numLoci,
    numAlleles
):
    try:
        return dical2_pkg.csd.DemoConfiguration.ConfigInfo(
            multiplicities,
            numDemes,
            jpype.java.lang.Integer(numLoci),
            numAlleles
        )
    except SystemExit:
        raise ValueError("Creation of ConfigInfo failed")


def _dical2_read_vcf(
    dical2_pkg,
    reader,
    haplotypesToRead,
    numInternalAlleles,
    commentCharacters,
    acceptUnphasedAsMissing,
    pathToVcf,
    offset,
    bedReader,
    verbose,
    filterPassString,
    commandLineReferenceFileName,
    vcfIgnoreDoubleEntries
):
    return dical2_pkg.haplotype.ReadSequences.readVcf(
        reader,
        haplotypesToRead,
        numInternalAlleles,
        commentCharacters,
        acceptUnphasedAsMissing,
        pathToVcf,
        offset,
        bedReader,
        verbose,
        filterPassString,
        commandLineReferenceFileName,
        vcfIgnoreDoubleEntries
    )


def _extended_config_info(
    dical2_pkg,
    preSequenceList,
    preConfigInfo,
    numPerDeme,
    theta,
    rho,
    mutMatrix,
    useLocusSkipping,
    lociPerHmmStep,
    printLoci,
    useStationaryForPartially,
    additionalHapIdx,
    csdList,
):
    return dical2_pkg.maximum_likelihood.DiCalParamSet.ExtendedConfigInfo(
        preSequenceList,
        preConfigInfo,
        numPerDeme,
        theta,
        rho,
        mutMatrix,
        useLocusSkipping,
        lociPerHmmStep,
        printLoci,
        useStationaryForPartially,
        additionalHapIdx,
        csdList,
    )


def _create_objective_function(
    dical2_pkg,
    csdConfigList,
    demoFactory,
    demoStateFactory,
    objectiveType,
    startPoint,
    estimateRecomScaling,
    trunkFactory,
    estimateTheta,
    defaultTheta,
    thetaDim,
    lastIteration,
    verbose,
    useEigenCore
):
    base_module = dical2_pkg.maximum_likelihood.StructureEstimationEM
    return base_module.DiCalObjectiveFunction(
        csdConfigList,
        demoFactory,
        demoStateFactory,
        objectiveType,
        startPoint,
        estimateRecomScaling,
        trunkFactory,
        estimateTheta,
        defaultTheta,
        thetaDim,
        lastIteration,
        verbose,
        useEigenCore
    )

def _get_lol_config_list(
    dical2_pkg,
    structuredConfig,
    pSet,
    fancyTransitionMap,
    pcLol
):
    base_module = dical2_pkg.maximum_likelihood.StructureEstimationEM
    return base_module.getLOLConfigList(
        structuredConfig,
        pSet,
        fancyTransitionMap,
        pcLol
    )

def _create_dical_objective(
    dical2_pkg,
    csdConfigList,
    demoFactory,
    demoStateFactory,
    objectiveType,
    startPoint,
    estimateRecomScaling,
    trunkFactory,
    estimateTheta,
    defaultTheta,
    thetaDim,
    lastIteration,
    verbose,
    useEigenCore
):
    base_module = dical2_pkg.maximum_likelihood.StructureEstimationEM
    return base_module.DiCalObjectiveFunction(
        csdConfigList,
        demoFactory,
        demoStateFactory,
        objectiveType,
        startPoint,
        estimateRecomScaling,
        trunkFactory,
        estimateTheta,
        defaultTheta,
        thetaDim,
        lastIteration,
        verbose,
        useEigenCore
    )


def get_jar_files():
    if not os.path.exists(dical2_path):
        return ""
    jar_files = [os.path.join(dical2_path, 'diCal2.jar')]
    if os.path.exists(os.path.join(dical2_path, "diCal2_lib")):
        for name in os.listdir(os.path.join(dical2_path, "diCal2_lib")):
            if name.endswith(".jar"):
                jar_files.append(os.path.join(dical2_path, "diCal2_lib", name))
    return jar_files


class DiCal2Engine(Engine):
    """
    Engine for diCal2.
    """

    id = 'diCal2'  #:
    supported_models = [DemographicModel, StructureDemographicModel,
                        CustomDemographicModel]  #:
    supported_data = [VCFDataHolder]  #:
    inner_data_type = tuple  #: tuple of sequence_list and config info

    # run JVM
    if (dical2_path is not None and
            os.path.exists(os.path.join(dical2_path, 'diCal2.jar'))):
        jpype.startJVM(jpype.getDefaultJVMPath(),
                       "-ea",
                       "-Djava.class.path="+":".join(get_jar_files()))
    base_module = jpype.JPackage("edu.berkeley.diCal2")

    @staticmethod
    def read_data(data_holder):
        """
        Reads data from `data_holder.filename` in inner type.

        :param data_holder: Holder of data to read.
        :type data_holder: :class:`gadma.DataHolder

        :returns: readed data
        :rtype: ``Engine.inner_data_type``
        """
        # Check for reference file
        if data_holder.reference_file is None:
            raise ValueError("DiCal2 require reference file.")

        # 1. create config_info - information from config file of dical2
        # It contains information from popmap
        sample2pop = read_popinfo(data_holder.popmap_file)
        sample_list = get_list_of_names_from_vcf(data_holder.filename)
        # VCF data holder always has population_labels and sample_sizes
        populations = data_holder.population_labels
        num_demes = len(populations)
        # create matrix of 0 and 1 where M[i][j]==1 means that sample i is
        # from population j
        multiplicities = jpype.java.util.ArrayList()
        for sample in sample_list:
            for _ in range(data_holder.ploidy):
                mult = [0]*num_demes
                if sample in sample2pop:
                    pop = sample2pop[sample]
                    mult[populations.index(pop)] = 1
                multiplicities.add(jpype.java.util.ArrayList(mult))
        config_info = _dical2_config_info(
            dical2_pkg=DiCal2Engine.base_module,
            multiplicities=multiplicities,
            numDemes=num_demes,
            numLoci=data_holder.sequence_length,
            numAlleles=2,
        )

        # 2. Read VCF file
        sequence_reader = jpype.java.io.FileReader(data_holder.filename)
        # .bed file
        bed_reader = None
        if data_holder.bed_file is not None:
            bed_reader = jpype.java.io.FileReader(data_holder.bed_file)
        vcfReferenceFilename = data_holder.reference_file
        # what sequences we want to read
        seq_to_read = [sum(mult) > 0 for mult in multiplicities]
        # read seqs
        try:
            sequence_list = _dical2_read_vcf(
                dical2_pkg=DiCal2Engine.base_module,
                reader=sequence_reader,
                haplotypesToRead=jpype.java.util.ArrayList(seq_to_read),
                numInternalAlleles=2,
                commentCharacters=["#", ">"],   # was taken from dical2 code
                acceptUnphasedAsMissing=False,
                pathToVcf=data_holder.filename,
                offset=0,
                bedReader=bed_reader,
                verbose=False,
                filterPassString="PASS",
                commandLineReferenceFileName=data_holder.reference_file,
                vcfIgnoreDoubleEntries=True,
            )
        except jpype.java.io.IOException as e:
            raise ValueError(e)
        finally:
            # close streams
            sequence_reader.close()
        if bed_reader is not None:
            bed_reader.close()
        # Check that lengths are equal, code from dical2
        assert sequence_list.size() > 0, "No sequences were read from VCF"
        hap_ind = 0
        while (sequence_list.get(hap_ind) is None):
            hap_ind += 1
        seq_len = sequence_list.get(hap_ind).getNumLoci()
        if data_holder.sequence_length is not None:
            if data_holder.sequence_length != seq_len:
                raise AssertionError("Sequence length is not equal to the "
                                     "length of sequence in VCF file: "
                                     f"{data_holder.sequence_length} !="
                                     f" {seq_len}")
        else:
            config_info.numLoci = seq_len

        # Create local and extended config info
        # We have one VCF so it is one element for each list
        local_config_info = _dical2_config_info(
            dical2_pkg=DiCal2Engine.base_module,
            multiplicities=config_info.multiplicities,
            numDemes=config_info.numDemes,
            numLoci=sequence_list.get(hap_ind).getNumLoci(),
            numAlleles=config_info.numAlleles,
        )
        return sequence_list, config_info

    def _create_demo_model(self, values, trunk_factory):
        """
        Creates demographic model with diCal2 from model of GADMA.

        Based on dical2.demography.DemographyFactory.ParamFileDemoFactory
        """
        var2value = self.model.var2value(values)
        def get_value(entity):
            return self.model.get_value_from_var2value(var2value, entity)
        demography_module = self.base_module.demography


        #create DemoFactory
        demo_factory = demography_module.DemographyFactory.DemoFactory(
            self.base_module.csd.UberDemographyCore.DEMO_FACTORY_EPSILON,
            trunk_factory.halfMigrationRate,
            None
        )

        params_to_vary = jpype.java.util.TreeMap()
        variables = self.model.variables

        def get_mutable_double(entity):
            return demography_module.DemographyFactory.getMutableDouble(
                str(entity),
                params_to_vary
            )
        last_end = get_mutable_double(0.0)
        partition = [ [pop] for pop in enumerate(self.model.events[-1].size_args)]
        for event in reversed(self.model.events):
            if isinstance(event, Epoch):
                # time of events
                epoch_time = get_value(event.time_arg)
                curr_break_point = last_end + epoch_time
                demo_factory.currEpochList.add(
                    SimplePair(last_end, curr_break_point)
                )
                # partition
                pop_list = jpype.java.util.ArrayList()
                for deme in partition:
                    ancient_deme = TreeSet(deme)
                    pop_list.add(ancient_deme)
                demo_factory.treePartitionList.add(popList)
                # pop sizes
                # TODO exponential
                pop_sizes = [get_mutable_double(get_value(arg))
                             for arg in event.size_args]
                demo_factory.currPopSizesList.add(pop_sizes)
                # TODO pulse matrix
                # adds pop_sizes equal to None and add additional rates
                mig_matrix = []
                for row in event.mig_args:
                    mig_matrix.append([])
                    for mig in row:
                        mig_matrix[-1].append(
                            get_mutable_double(get_value(mig))
                        )
                demo_factory.currMigrationMatrixList.add(mig_matrix)
                demo_factory.currPulseMigrationMatrixList.add(None)
                # Exponential rates
                # TODO
                if event.dyn_args is None:
                    dynamics = ["Sud" for _ in event.size_args]
                else:
                    dynamics = [get_value(arg) for arg in event.dyn_args]
                rates = []
                for dyn in dynamics:
                    assert dyn == "Sud"
                    rates.append(get_mutable_double(0.0))
                demo_factory.currExpRatesList.add(rates)
            elif isinstance(event, Split):
                # fix partition
                pop_that_split = event.pop_to_div
                partition[pop_that_split].append(partition[-1])
                partition = partition[:-1]
        demo_factory.currEpochList.add(
            SimplePair(last_end, get_mutable_double(Double.POSITIVE_INFINITY))
        )
        return demo_factory

    def evaluate(self, values, **options):
        """
        Evaluation of the objective function of the engine.

        :param values: values of variables of setted demographic model.
        """
        mut_matrix = DiCal2Engine.base_module.Jama.Matrix([[0, 1], [1, 0]])
        extended_config_info = _extended_config_info(
            dical2_pkg=DiCal2Engine.base_module,
            preSequenceList=sequence_list,
            preConfigInfo=local_config_info,
            numPerDeme=None,  # looks like it is null
            theta=self.model.mu,
            rho=0,#self.model.recombination_rate,  # TODO
            mutMatrix=mut_matrix,
            useLocusSkipping=True,
            lociPerHmmStep=self.loci_per_HMM_step,  # TODO
            printLoci=False,
            useStationaryForPartially=False,  # TODO
            additionalHapIdx=None,
            csdList=None,  # is not None when csd are from file, we have LOL
        )

        # Demography
        csd_module = self.base_module.csd
        # 1. Trunks TODO: what is it?
        default_trunc_style_name = "migratingEthan"
        default_cake_style_name = "average"
        cake_style = csd_module.TrunkProcess.getCakeStyle(
            default_cake_style_name
        )
        trunk_factory = csd_module.TrunkProcessFactory(
            default_trunc_style_name,
            cake_style
        )
        assert trunk_factory is not None
        # interval type
        default_interval_type_name = "loguniform"
        default_interval_factory_params = "8,0.01,4" # TODO may be different for 3 pops
        default_print_intervals = False
        interval_factory = csd_module.IntervalFactory.getIntervalFactory(
            default_interval_type_name,
            default_interval_factory_params,
            default_print_intervals
        )
        assert interval_factory is not None
        # demo state factory
        default_add_trunk_intervals = 0
        demo_state_factory = csd_module.DemoState.DemoStateFactory(
            interval_factory,
            default_add_trunk_intervals,
            csd_module.UberDemographyCore.ONELOCUS_EPSILON
        )

        demo_factory = self._create_demo_model(values, trunk_factory)

        # create config list for lol objective
        chunk = 0
        structured_config = extended_config_info.structuredConfig
        p_set = extended_config_info.pSet;
        fancy_transition_map = extended_config_info.fancyTransitionMap
        csd_configs = jpype.java.util.ArrayList()
        csd_configs.add(jpype.java.util.ArrayList())
        csd_configs.get(0).add(jpype.java.util.ArrayList())
        csd_configs.get(0).get(0).addAll(
            _get_lol_config_list(structuredConfig=structured_config,
                                 pSet=p_set,
                                 fancyTransitionMap=fancy_transition_map,
                                 pcLol=False)
        )
        local_base = self.base_module.maximum_likelihood.StructureEstimationEM
        cond_obj_type = local_base.ConditionalObjectiveFunctionType
        # in Dical2 there are several switches for choice of this function
        # We leave switches off
        cond_obj_fun = cond_obj_type.ConditionLineage
        # Create objective
        objective_function = _create_dical_objective(
            csdConfigList=csd_configs,
            demoFactory=demo_factory,
            demoStateFactory=demo_state_factory,
            objectiveType=cond_obj_fun,
            startPoint=[],
            estimateRecomScaling=False,  # TODO check that we do not need it
            trunkFactory=trunk_factory,
            estimateTheta=False,  # TODO check that we do not need it
            defaultTheta=p_set.getMutationRate(0),
            thetaDim=0,  # ? TODO
            lastIteration=False,
            verbose=True,
            useEigenCore=False,  # switch is off be default ODECore
        )
        return objective_function.getLogLikelihood()
