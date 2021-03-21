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
            numLoci,
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
            if data_holder.sequence_length == seq_len:
                raise AssertionError("Sequence length is not equal to the "
                                     "length of sequence in VCF file: "
                                     f"{data_holder.sequence_length} !="
                                     f" {seq_len}")
        else:
            config_info.numLoci = seq_len
        return sequence_list, config_info

    def evaluate(self, values, **options):
        """
        Evaluation of the objective function of the engine.

        :param values: values of variables of setted demographic model.
        """
        raise NotImplementedError
