import unittest
import pytest
import itertools
import sys
import shutil
import pickle
import gadma
from os import listdir
from collections import namedtuple
import os
import numpy as np
from pathlib import Path
from gadma import *
from tests.test_fast_data_read import rewrite_params_file
import warnings # we ignore warning of unclosed files in dadi

warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.dadi_moments_common', lineno=284)

try:
    import dadi
    DADI_NOT_AVAILABLE = False
except ImportError:
    DADI_NOT_AVAILABLE = True

try:
    import moments
    MOMENTS_NOT_AVAILABLE = False
except ImportError:
    MOMENTS_NOT_AVAILABLE = True

try:
    import momi
    MOMI_NOT_AVAILABLE = False
except ImportError:
    MOMI_NOT_AVAILABLE = True

# Test data
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data", "DATA")
YRI_CEU_DATA = os.path.join(DATA_PATH, "sfs", "YRI_CEU.fs")
YRI_CEU_NO_LABELS_DATA = os.path.join(DATA_PATH, "sfs", "YRI_CEU_old.fs")
YRI_CEU_F_DATA = os.path.join(DATA_PATH, "sfs", "YRI_CEU_folded.fs")
SNP_DATA = os.path.join(DATA_PATH, "sfs", "data.txt")
NO_OUT_SNP_DATA = os.path.join(DATA_PATH, "sfs", "data_no_outgroup.txt")
DAMAGED_SNP_DATA = os.path.join(DATA_PATH, "sfs", "damaged_data.txt")
STRANGE_DATA = os.path.join(DATA_PATH, "sfs", "some_strange_data")

VCF_DATA = os.path.join(DATA_PATH, "vcf", "data.vcf")
POPMAP = os.path.join(DATA_PATH, "vcf", "popmap")
VCF_SIM_YRI_CEU_DATA =  os.path.join(DATA_PATH, "vcf",
                                     "out_of_africa_chr22_sim.vcf")
BAD_FILTER_VCF_DATA = os.path.join(DATA_PATH, "vcf", "bad_filter.vcf")
NO_AA_INFO_VCF = os.path.join(DATA_PATH, "vcf", "no_aa_info.vcf")
ONE_PLOIDY_VCF = os.path.join(DATA_PATH, "vcf", "one_ploidy.vcf")
NOT_NUCL_VCF = os.path.join(DATA_PATH, "vcf", "not_nucl.vcf")
POPMAP_SIM_YRI_CEU =  os.path.join(DATA_PATH, "vcf",
                                   "out_of_africa_chr22_sim.popmap")
BAD_POPMAP = os.path.join(DATA_PATH, "vcf", "bad.popmap")

POP_MAP = os.path.join(
    DATA_PATH, 'vcf_ld', "pop_map.txt")
REC_MAPS_DIR = os.path.join(
    DATA_PATH, 'vcf_ld', "rec_maps")
VCF_DATA_LD = os.path.join(
    DATA_PATH, 'vcf_ld', "vcf_data.vcf")
TEST_OUTPUT = os.path.join(
    DATA_PATH, 'vcf_ld', "test_output")
TEST_BED_FILES = os.path.join(
    DATA_PATH, 'vcf_ld', "test_bed_files")
SFS_DATA = os.path.join(
    DATA_PATH, 'vcf_ld', "wrong_data.fs")

PREPROCESSED_DATA = os.path.join(
    DATA_PATH, 'vcf_ld', "preprocessed_data.bp")

SAVE_IMAGE = os.path.join(
    DATA_PATH, 'vcf_ld', "ld_curves.jpg")

DATA_HOLDER_FOR_MODELS = VCFDataHolder(
            vcf_file=VCF_DATA_LD,
            popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
)


class TestDataHolder(unittest.TestCase):

    def setUp(self):
        if Path(TEST_OUTPUT).exists():
            shutil.rmtree(TEST_OUTPUT)

    def tearDown(self):
        if Path("./output_dir").exists():
            shutil.rmtree("./output_dir")
        if Path(TEST_OUTPUT).exists():
            shutil.rmtree(TEST_OUTPUT)
            os.remove(TEST_OUTPUT)

    def _check_data(self, data_holder, pop_labels, outgroup, sample_sizes):
        self.assertTrue(all(data_holder.projections == sample_sizes),
                        msg=f"{data_holder.projections} != {sample_sizes}")
        self.assertEqual(list(data_holder.population_labels), list(pop_labels),
                         msg=f"{data_holder.population_labels} != {pop_labels}")
        self.assertEqual(data_holder.outgroup, outgroup,
                          msg=f"{data_holder.outgroup} != {outgroup}")

    def _load_with_dadi(self, data_file, size, labels, outgroup):
        warnings.filterwarnings(action="ignore", message="unclosed",
                         category=ResourceWarning)
        if data_file.split('.')[-1] == 'txt':
            d = dadi.Misc.make_data_dict(data_file)
            data = dadi.Spectrum.from_data_dict(d, labels, size,
                                                polarized=outgroup)
            return data
        data = dadi.Spectrum.from_file(YRI_CEU_DATA)
        if labels == ["CEU", "YRI"] and data_file != YRI_CEU_NO_LABELS_DATA:
            data = np.transpose(data, [1,0])
            data.pop_ids = labels
        if data_file == YRI_CEU_NO_LABELS_DATA:
            assert len(labels) == 2
            data.pop_ids = labels
        if len(labels) == 1:
            marg = []
            for i, lab in enumerate(data.pop_ids):
                if lab not in labels:
                    marg.append(i)
            data = data.marginalize(marg)
        data = data.project(size)
        if not outgroup:
            data = data.fold()
        return data

    def test_vcf_dataholder_init(self):
        sample_sizes = (2,1)
        outgroup = True
        d = VCFDataHolder(vcf_file=VCF_DATA, popmap_file=POPMAP,
                          projections=sample_sizes, outgroup=outgroup)
        self.assertEqual(d.projections, sample_sizes)
        self.assertEqual(d.outgroup, outgroup)

    def _test_sfs_reading(self, id):
        sizes = [None, (20,20), (10, 10), (10, 5), (5,), (10,)]
        if id == "momi2":
            sizes = sizes[:2]
        outgroup = [None, True, False]
        labels = [None, ["YRI", "CEU"], ["CEU", "YRI"], ["CEU"]]
        seq_lens = [None, 1e6]
        data = [YRI_CEU_DATA, SNP_DATA, NO_OUT_SNP_DATA]
        if id != "momi2":
            data.append(YRI_CEU_NO_LABELS_DATA)
        for dat, siz, lab, seq, out in itertools.product(data, sizes, labels,
                                                         seq_lens, outgroup):
            if lab is not None and siz is not None and len(lab) != len(siz):
                continue
            if lab is None and siz is not None and len(siz) == 1:
                continue
            if dat == YRI_CEU_NO_LABELS_DATA:
                if lab is not None and len(lab) == 1:
                    continue
            if dat == NO_OUT_SNP_DATA and out:
                continue
            with self.subTest(data=dat, size=siz, labels=lab,
                              seq_len=seq, outgroup=out):
                sfs_holder = SFSDataHolder(dat, projections=siz, outgroup=out,
                                           population_labels=lab,
                                           sequence_length=seq)
                engine = get_engine(id)
                engine.data = sfs_holder  # Reading is here

                corr_size = None
                if dat == SNP_DATA or dat == NO_OUT_SNP_DATA:
                    if lab == ["CEU", "YRI"]:
                        corr_size = [44, 24]
                    else:
                        corr_size = (24, 44)
                else:
                    corr_size = (20, 20)
                if lab is not None and len(lab) == 1:
                    corr_size = (corr_size[1],)
                siz = siz or corr_size
                lab = lab or (
                    ["Pop 1", "Pop 2"] if dat == YRI_CEU_NO_LABELS_DATA
                    else ["YRI", "CEU"])
                out = True if out is None else out
                if dat == NO_OUT_SNP_DATA:
                    out = False
                self._check_data(engine.data_holder, lab, out, siz)
                if engine.id in ["dadi", "moments"]:
                    sfs = self._load_with_dadi(dat, siz, lab, out)
                    self.assertTrue(np.allclose(engine.inner_data, sfs))

    def _test_vcf_reading(self, id):
        sizes = [None, (4, 6), (4, 2), (6,), (4,)]
        outgroup = [None, True, False]
        labels = [None, ["YRI", "CEU"], ["CEU", "YRI"], ["CEU"]]
        seq_lens = [None, 51304566]
        data = [(VCF_SIM_YRI_CEU_DATA, POPMAP_SIM_YRI_CEU)]
        for dat, siz, lab, out in itertools.product(data, sizes, labels,
                                                    outgroup):
            # print(siz, lab, out)
            seq = seq_lens[-1]
            vcf_file, popmap_file = dat
            data_holder = VCFDataHolder(
                vcf_file=vcf_file,
                popmap_file=popmap_file,
                population_labels=lab,
                projections=siz,
                sequence_length=seq,
                outgroup=out,
            )
            if lab is not None and siz is not None and len(lab) != len(siz):
                self.assertRaises((ValueError, AssertionError),
                                  get_engine(id).read_data, data_holder)
                continue
            if lab == ["CEU", "YRI"] and siz is not None and list(siz) == [4, 6]:
                self.assertRaises(AssertionError,
                                  get_engine(id).read_data, data_holder)
                continue
            if lab is None and siz is not None and len(siz) != 2:
                self.assertRaises((ValueError, AssertionError),
                                  get_engine(id).read_data, data_holder)
                continue

            # momi2 cannot downsize SFS
            if (id == "momi2" and siz is not None and
                   (list(siz) == [4, 2] or list(siz) == [4])):
                continue

            engine = get_engine(id)
            engine.data = data_holder
            if lab is None:
                lab = ["YRI", "CEU"]
            if siz is None:
                if len(lab) == 2:
                    siz = (4, 6)
                    if lab == ["CEU", "YRI"]:
                        siz = [6, 4]
                else:
                    assert lab == ["CEU"]
                    siz = (6,)
            if out is None:
                out = True
            self._check_data(engine.data_holder, lab, out, siz)

        if engine.id == "momi2":
            return
        # for some reason there is a problem for momi2 to read this file
        data_holder = VCFDataHolder(
            vcf_file=VCF_DATA,
            popmap_file=POPMAP,
            projections=None,
            outgroup=None,
        )
        sfs = get_engine(id).read_data(data_holder)
        self.assertEqual(sfs.S(), 1)

    def _test_read_fails(self, id):
        warnings.filterwarnings(action="ignore", message="unclosed",
                         category=ResourceWarning)
        data_holder = SFSDataHolder(YRI_CEU_DATA, population_labels=[1, 2])
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(YRI_CEU_DATA, projections=(40, 50))
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(YRI_CEU_F_DATA, outgroup=True)
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(STRANGE_DATA)
        self.assertRaises(SyntaxError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(DAMAGED_SNP_DATA)
        self.assertRaises(
            ValueError,
            engines.dadi_moments_common._get_default_from_snp_format,
            DAMAGED_SNP_DATA
        )
        self.assertRaises(SyntaxError, get_engine(id).read_data, data_holder)

        # Bad data holder
        not_data_holder = get_engine(id)
        self.assertRaises(ValueError, get_engine(id).read_data, not_data_holder)

        # VCF data
        data_holder = VCFDataHolder(vcf_file=BAD_FILTER_VCF_DATA,
                                    popmap_file=POPMAP_SIM_YRI_CEU)
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = VCFDataHolder(vcf_file=ONE_PLOIDY_VCF,
                                    popmap_file=POPMAP_SIM_YRI_CEU)
        self.assertRaises(AssertionError, get_engine(id).read_data, data_holder)
        data_holder = VCFDataHolder(vcf_file=VCF_SIM_YRI_CEU_DATA,
                                    popmap_file=BAD_POPMAP)
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = VCFDataHolder(vcf_file=NOT_NUCL_VCF,
                                    popmap_file=POPMAP_SIM_YRI_CEU)
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = VCFDataHolder(vcf_file=NO_AA_INFO_VCF,
                                    popmap_file=POPMAP_SIM_YRI_CEU,
                                    outgroup=True)
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)

    @pytest.mark.timeout(0)
    def test_sfs_data_reading(self):
        for engine in all_engines():
            if SFSDataHolder in engine.supported_data:
                self._test_sfs_reading(engine.id)

    @pytest.mark.timeout(0)
    def test_vcf_data_reading(self):
        for engine in all_engines():
            if engine.id != "momentsLD":
                self._test_vcf_reading(engine.id)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_reading_fails(self):
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.data_utils', lineno=106)
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.data_utils', lineno=110)
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.dadi_moments_common', lineno=693)
        self._test_read_fails('dadi')

    @unittest.skipIf(MOMENTS_NOT_AVAILABLE, "Moments module is not installed")
    def test_moments_reading_fails(self):
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.data_utils', lineno=106)
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.data_utils', lineno=110)
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.dadi_moments_common', lineno=693)
        self._test_read_fails('moments')

    @unittest.skipIf(MOMENTS_NOT_AVAILABLE or DADI_NOT_AVAILABLE,
                     'moments and dadi are not installed')
    def test_data_transform(self):
        data_holder = SFSDataHolder(YRI_CEU_DATA)
        dadi_eng = get_engine('dadi')
        dadi_eng.set_data(data_holder)
        moments_eng = get_engine('moments')
        moments_eng.set_data(dadi_eng.data)
        dadi_eng.set_data(moments_eng.data)
        moments_eng.set_data(np.array([[1, 2], [2, 3]]))
        self.assertRaises(ValueError, moments_eng.set_data, set())


def test_fsc_reading():
    # old format
    warnings.filterwarnings(action='ignore', category=UserWarning,
                    module='.*\.dadi_moments_common', lineno=350)
    # missed lines in vcf
    warnings.filterwarnings(action='ignore', category=UserWarning,
                    module='.*\.dadi_moments_common', lineno=693)
    warnings.filterwarnings(action='ignore', category=UserWarning,
                    module='.*\.dadi_moments_common', lineno=703)
    # repeats in vcf file
    warnings.filterwarnings(action='ignore', category=UserWarning,
                    module='.*\.dadi_moments_common', lineno=710)
    # repeats in vcf file
    warnings.filterwarnings(action='ignore', category=UserWarning,
                    module='.*\.dadi_moments_common', lineno=521)

    TestInfo = namedtuple('TestInfo',
        'file population_labels projections outgroup')

    vcf = os.path.join(DATA_PATH, 'fsc', 'test_data.vcf')
    for engine in all_engines():
        if engine.id == "momentsLD":
            continue

        tests = [
            # joint SFSs for 2 demes; minor and derived allele
            # also test projections
            TestInfo('jointMAFpop1_0', ('YRI', 'CEU'), None, False),
            TestInfo('jointDAFpop1_0', ('YRI', 'CEU'), [4, 2], True),
            # SFS for single population; minor and derived allele
            TestInfo('YRI_DAFpop0', ('YRI',), None, True),
            TestInfo('YRI_MAFpop0', ('YRI',), None, False),
            # multi-SFS for minor and derived allele
            TestInfo('MSFS', ('dwarves', 'elves', 'orcs'), None, False),
            TestInfo('DSFS', ('dwarves', 'elves', 'orcs'), None, True)
        ]

        for test in tests:
            obs, popmap = (os.path.join(DATA_PATH,'fsc', f'{test.file}.{ext}')
                for ext in ['obs', 'popmap'])

            fsc_data_holder = SFSDataHolder(
                obs,
                population_labels=test.population_labels,
                projections=test.projections,
                sequence_length=None,
                outgroup=test.outgroup
            )

            vcf_data_holder = VCFDataHolder(
                vcf_file=vcf,
                popmap_file=popmap,
                population_labels=test.population_labels,
                projections=test.projections,
                sequence_length=None,
                outgroup=test.outgroup
            )
            if engine.id == "momi":
                if test.projections is not None and test.projections == [4, 2]:
                    continue

            fsc_data = engine.read_data(fsc_data_holder)
            vcf_data = engine.read_data(vcf_data_holder)

            debug = False
            if debug:
                print(test.file, np.sum(np.abs(fsc_data - vcf_data)))
                print('from fsc:')
                print(fsc_data)
                print('from vcf:')
                print(vcf_data)

                print('difference:')
                print(fsc_data - vcf_data)
                diff = fsc_data - vcf_data
                print('mask')
                print(fsc_data.mask)
                print(vcf_data.mask)

            if engine.id in ["dadi", "moments"]:
                assert np.all(fsc_data.mask == vcf_data.mask), \
                        test.file + ": mask did not match"

                # on current test data SFSs read from vcf and fsc across all elements
                # differ by (max) 4 total, probably because easySFS implementation
                # deviates from dadi somehow
                delta = np.sum(np.abs(fsc_data - vcf_data))
                assert delta <= 4, "difference between SFSs is larger than expected"

            assert fsc_data.folded == vcf_data.folded, \
                    test.file + ": folded did not match"

        # test the case with multiple observations per file
        multiple = SFSDataHolder(
            os.path.join(DATA_PATH, 'fsc', 'MO_jointDAFpop1_0.obs'),
            population_labels = ('YRI', 'CEU'),
            projections=None,
            sequence_length=None,
            outgroup=True
        )
        single = SFSDataHolder(
            os.path.join(DATA_PATH, 'fsc', 'jointDAFpop1_0.obs'),
            population_labels = ('YRI', 'CEU'),
            projections=None,
            sequence_length=None,
            outgroup=True
        )
        multiple_SFS = engine.read_data(multiple)
        single_SFS = engine.read_data(single)
        assert np.all(multiple_SFS == single_SFS), "Several observations per file"


class TestVCFDataHolderLD(unittest.TestCase):

    def tearDown(self):
        if Path(f"{TEST_OUTPUT}/_bed_files/").exists():
            shutil.rmtree(f"{TEST_OUTPUT}/_bed_files/")

    def test_vcf_data_holder_ld_init(self):
        ld_data = VCFDataHolder(
            vcf_file=VCF_DATA_LD,
            popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
        )
        self.assertEqual(ld_data.filename, VCF_DATA_LD)
        self.assertEqual(ld_data.popmap_file, POP_MAP)
        self.assertEqual(ld_data.recombination_maps, REC_MAPS_DIR)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_sfs_data_holder(self):
        ld_wrong_data = SFSDataHolder(
            sfs_file=SFS_DATA)

        settings = SettingsStorage()
        settings.engine = "momentsLD"
        settings.data_holder = ld_wrong_data
        self.assertRaises(AssertionError, settings.read_data)


def get_settings_test():
    settings, args = get_settings()
    check_required_settings(settings)
    return settings, args


class TestSettingStorageLDStats(unittest.TestCase):
    def tearDown(self):
        if Path(f"{TEST_OUTPUT}/_bed_files/").exists():
            shutil.rmtree(f"{TEST_OUTPUT}/_bed_files/")

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_param_file_with_ld(self):
        DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_test_correct')

        sys.argv = ['gadma', '-p', param_file]
        settings, _ = get_settings_test()
        settings.read_data()

        self.assertEqual(settings.output_directory, abspath(TEST_OUTPUT))
        self.assertEqual(settings.data_holder.filename, VCF_DATA_LD)
        self.assertEqual(settings.data_holder.popmap_file, POP_MAP)
        self.assertEqual(settings.data_holder.recombination_maps, REC_MAPS_DIR)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_errors_in_param_file(self):
        DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_param_file_with_wrong_keys_in_dict')
        sys.argv = ['gadma', '-p', param_file]
        self.assertRaises(KeyError, get_settings_test)

        DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_test_no_mutation_rate')
        sys.argv = ['gadma', '-p', param_file]
        self.assertRaises(ValueError, get_settings_test)

        # param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_param_file_with_dict_and_wrong_engine')
        # sys.argv = ['gadma', '-p', param_file]
        # self.assertRaises(AssertionError, get_settings_test)

        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_without_anc_size_as_parameter')
        sys.argv = ['gadma', '-p', param_file]
        settings, args = get_settings_test()
        self.assertTrue(settings.ancestral_size_as_parameter, True)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    @pytest.mark.timeout(0)
    def test_correct_LD_data_processing(self):
        data_reading_case_list = ['without_rec_map', 'with_rec_map', 'with_rec_maps_in_one_file']
        for case in data_reading_case_list:
            param_file = os.path.join(
                os.path.dirname(__file__), "test_data",
                'PARAMS', 'ld_data_parsing_params',
                f'ld_data_parsings_params_{case}'
            )
            sys.argv = ['gadma', '-p', param_file]
            settings, _ = get_settings_test()
            try:
                data_gadma = settings.read_data()

                preprocessed_test_data = os.path.join(
                    DATA_PATH, 'vcf_ld',
                    f'parsing_test_data_{case}.bp'
                )

                with open(preprocessed_test_data, 'rb') as file:
                    region_stats_moments_ld = pickle.load(file)
                    data_moments_ld = moments.LD.Parsing.bootstrap_data(region_stats_moments_ld)

                self.assertEqual(
                    len(data_gadma['means']),
                    len(data_moments_ld['means'])
                )
                for arr in range(len(data_gadma["means"])):
                    if not (any(np.isnan(
                            data_moments_ld["means"][arr]) |
                             np.isnan(data_gadma["means"][arr]))):
                        self.assertTrue(np.allclose(
                            data_moments_ld["means"][arr],
                            data_gadma["means"][arr]))
            finally:
                if Path(f"{TEST_OUTPUT}/_bed_files/").exists():
                    shutil.rmtree(f"{TEST_OUTPUT}/_bed_files/")
                rewrite_params_file(param_file)
