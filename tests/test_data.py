import unittest
import itertools
import os
import numpy as np
import gadma
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

from gadma import dical2_available

# Test data
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data", "DATA")
YRI_CEU_DATA = os.path.join(DATA_PATH, "sfs", "YRI_CEU.fs")
YRI_CEU_NO_LABELS_DATA = os.path.join(DATA_PATH, "sfs", "YRI_CEU_old.fs")
YRI_CEU_F_DATA = os.path.join(DATA_PATH, "sfs", "YRI_CEU_folded.fs")
SNP_DATA = os.path.join(DATA_PATH, "sfs", "data.txt")
DAMAGED_SNP_DATA = os.path.join(DATA_PATH, "sfs", "damaged_data.txt")
STRANGE_DATA = os.path.join(DATA_PATH, "sfs", "some_strange_data")

VCF_DATA = os.path.join(DATA_PATH, "vcf", "data.vcf")
POPMAP = os.path.join(DATA_PATH, "vcf", "popmap")

CONTIG0 =  os.path.join(DATA_PATH, "vcf", "contig.0.vcf")
CONTIG0_POPMAP = os.path.join(DATA_PATH, "vcf", "contig_0_popmap")
REFERENCE = os.path.join(DATA_PATH, "vcf", "reference.fa")
SMALL_REFERENCE = os.path.join(DATA_PATH, "vcf", "small_reference.fa")


def run_dical2_reading(q, er_q, data):
    try:
        res = gadma.engines.DiCal2Engine.read_data(data)
        assert isinstance(res, tuple)
        assert len(res) == 2
        # q.put(res)  # Not working as java objects are unpickleable
    except Exception as e:
        er_q.put(e)
        raise e


class TestDataHolder(unittest.TestCase):

    def _check_data(self, data, pop_labels, outgroup, sample_sizes):
        self.assertTrue(all(data.sample_sizes == sample_sizes),
                        msg=f"{data.sample_sizes} != {sample_sizes}")
        self.assertEqual(data.pop_ids, pop_labels,
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
        sample_sizes = [4,2]
        outgroup = True
        d = VCFDataHolder(vcf_file=VCF_DATA, popmap_file=POPMAP,
                          sample_sizes=sample_sizes, outgroup=outgroup)
        self.assertEqual(d.population_labels, ['Pop1', 'Pop2'])
        self.assertEqual(d.projections, sample_sizes)
        self.assertEqual(d.outgroup, outgroup)

        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP)
        self.assertEqual(data.population_labels, ["pop1", "pop2"])
        self.assertEqual(data.projections, [4, 2])

        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP,
                             reference_file=REFERENCE)
        self.assertRaises(AssertionError, VCFDataHolder, vcf_file=CONTIG0,
                          popmap_file=CONTIG0_POPMAP,
                          population_labels=["1", "pop2"])
        self.assertRaises(AssertionError, VCFDataHolder, vcf_file=CONTIG0,
                          popmap_file=CONTIG0_POPMAP, sample_sizes=[2, 1])
 
    def _test_sfs_reading(self, id):
        sizes = [None, (20,20), (10, 10), (10, 5), (5,), (10,)]
        outgroup = [None, True, False]
        labels = [None, ["YRI", "CEU"], ["CEU", "YRI"], ["CEU"]]
        seq_lens = [None, 1e6]
        data = [YRI_CEU_DATA, YRI_CEU_NO_LABELS_DATA, SNP_DATA]
        for dat, siz, lab, seq, out in itertools.product(data, sizes, labels,
                                                         seq_lens, outgroup):
            if lab is not None and siz is not None and len(lab) != len(siz):
                continue
            if lab is None and siz is not None and len(siz) == 1:
                continue 
            if dat == YRI_CEU_NO_LABELS_DATA:
                if lab is not None and len(lab) == 1:
                    continue
            with self.subTest(data=dat, size=siz, labels=lab,
                              seq_len=seq, outgroup=out):
                sfs_holder = SFSDataHolder(dat, projections=siz, outgroup=out,
                                           population_labels=lab,
                                           sequence_length=seq)
                data = get_engine(id).read_data(sfs_holder)
                corr_size = None
                if dat == SNP_DATA:
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
                self._check_data(data, lab, out, siz)
                sfs = self._load_with_dadi(dat, siz, lab, out)
                self.assertTrue(np.allclose(data, sfs))

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
        self.assertRaises(SyntaxError, get_engine(id).read_data, data_holder)
        vcf_data = VCFDataHolder(VCF_DATA, POPMAP, (4, 2), True)
        self.assertRaises(ValueError, get_engine(id).read_data, vcf_data)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_reading(self):
        self._test_sfs_reading('dadi')

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_reading_fails(self):
        self._test_read_fails('dadi')

    @unittest.skipIf(MOMENTS_NOT_AVAILABLE, "Moments module is not installed")
    def test_moments_reading(self):
        self._test_sfs_reading('moments')

    @unittest.skipIf(MOMENTS_NOT_AVAILABLE, "Moments module is not installed")
    def test_moments_reading_fails(self):
        self._test_read_fails('moments')

    @unittest.skipIf(dical2_available, "DiCal2 is not installed")
    def test_dical2_reading(self):
        from .test_engines import func_in_separate_process
        data = VCFDataHolder(vcf_file=CONTIG0, popmap_file=CONTIG0_POPMAP,
                             reference_file=REFERENCE)
        func_in_separate_process(run_dical2_reading, data)  # checks are inside

    @unittest.skipIf(dical2_available, "DiCal2 is not installed")
    def test_dical2_reading_fails(self):
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
        gadma,engines.DiCal2Engine._startJVM()

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
