import unittest
import itertools
import os
import numpy as np
from gadma import *
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



class TestDataHolder(unittest.TestCase):
    def _check_data(self, data, pop_labels, outgroup, sample_sizes):
        self.assertTrue(all(data.sample_sizes == sample_sizes),
                        msg=f"{data.sample_sizes} != {sample_sizes}")
        self.assertEqual(list(data.pop_ids), list(pop_labels),
                         msg=f"{data.pop_ids} != {pop_labels}")
        self.assertEqual(not data.folded, outgroup,
                          msg=f"{not data.folded} != {outgroup}")

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
        outgroup = [None, True, False]
        labels = [None, ["YRI", "CEU"], ["CEU", "YRI"], ["CEU"]]
        seq_lens = [None, 1e6]
        data = [YRI_CEU_DATA, YRI_CEU_NO_LABELS_DATA, SNP_DATA, NO_OUT_SNP_DATA]
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
                data = get_engine(id).read_data(sfs_holder)
                corr_size = None
                if dat == SNP_DATA or dat == NO_OUT_SNP_DATA:
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
                self._check_data(data, lab, out, siz)
                sfs = self._load_with_dadi(dat, siz, lab, out)
                self.assertTrue(np.allclose(data, sfs))

    def _test_vcf_reading(self, id):
        sizes = [None, (4, 6), (4, 2), (4,), (3,)]
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
                self.assertRaises(ValueError,
                                  get_engine(id).read_data, data_holder)
                continue
            if lab == ["CEU", "YRI"] and siz is not None and list(siz) == [4, 6]:
                self.assertRaises(AssertionError,
                                  get_engine(id).read_data, data_holder)
                continue
            if lab is None and siz is not None and len(siz) != 2:
                self.assertRaises(ValueError,
                                  get_engine(id).read_data, data_holder)
                continue
            sfs = get_engine(id).read_data(data_holder)
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
            self._check_data(sfs, lab, out, siz)

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


    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_reading(self):
        self._test_sfs_reading('dadi')
        self._test_vcf_reading('dadi')

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_reading_fails(self):
        self._test_read_fails('dadi')

    @unittest.skipIf(MOMENTS_NOT_AVAILABLE, "Moments module is not installed")
    def test_moments_reading(self):
        self._test_sfs_reading('moments')
        self._test_vcf_reading('moments')


    @unittest.skipIf(MOMENTS_NOT_AVAILABLE, "Moments module is not installed")
    def test_moments_reading_fails(self):
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
