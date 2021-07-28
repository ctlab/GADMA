import unittest
import sys
import os
import numpy as np
import copy
import shutil
import itertools
from collections import namedtuple
import warnings

import gadma
from gadma.cli.arg_parser import ArgParser, get_settings,\
    check_required_settings
from gadma.cli import SettingsStorage, get_variables
from gadma.cli import settings as default_settings
from gadma.utils import StdAndFileLogger
from gadma.utils import PopulationSizeVariable, MigrationVariable,\
    TimeVariable, FractionVariable, ContinuousVariable
from gadma import *
from gadma.core.shared_dict import SharedDict, SharedDictForCoreRun
from gadma.cli import arg_parser

try:
    import moments.LD
    MOMENTS_LD_NOT_AVAILABLE = False
except ImportError:
    MOMENTS_LD_NOT_AVAILABLE = True

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data", "DATA")

POP_MAP_LD = os.path.join(DATA_PATH, 'vfc_ld', "pop_map.txt")
REC_MAP_LD = os.path.join(DATA_PATH, 'vfc_ld', "rec_map.txt")
VCF_DATA_LD = os.path.join(DATA_PATH, 'vfc_ld', "two_pop.vcf")


# class TestVCFDataHolderLD(unittest.TestCase):
#
#     def test_vcf_data_holder_ld_init(self):
#         ld_data = VCFDataHolder(vcf_file=VCF_DATA_LD, popmap_file=POP_MAP_LD,
#                                 recombination_map=REC_MAP_LD)
#         self.assertEqual(ld_data.filename, VCF_DATA_LD)
#         self.assertEqual(ld_data.popmap_file, POP_MAP_LD)
#         self.assertEqual(ld_data.recombination_map, REC_MAP_LD)


def get_settings_test():
    settings, args = get_settings()
    check_required_settings(settings)
    return settings, args


# class TestSettingStorageLDStats(unittest.TestCase):
#
#     def test_param_file_wit_ld(self):
#
#         # param_file = os.path.join(DATA_PATH, "PARAMS",
#         #                           'example_params_old')
#         # sys.argv = ['gadma', '-p', param_file, '-o' 'some_dir']
#         # settings, _ = get_settings_test()
#         #
#         # self.assertEqual(settings.output_directory, abspath('some_dir'))
#
#         param_file = os.path.join(DATA_PATH, "PARAMS", 'another_test_params')
#         another_fs = os.path.join(DATA_PATH, "DATA", "sfs", 'YRI_CEU.fs')
#
#         sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir',
#                     '-i', another_fs, '--only_models']
#         settings, _ = get_settings_test()
#
#         self.assertEqual(settings.output_directory, abspath('some_dir'))
#         self.assertEqual(settings.input_data, abspath(another_fs))

file_path = '/home/stas/git/gadma_moments/devel_fork/GADMA/tests/tests/test_data/DATA/sfs/small_1pop.fs'
print(check_file_existence(file_path))
