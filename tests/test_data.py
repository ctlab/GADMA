import unittest
import itertools
import os
import numpy as np
from gadma import *
import warnings # we ignore warning of unclosed files in dadi

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
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
YRI_CEU_DATA = os.path.join(DATA_PATH, "YRI_CEU.fs")
YRI_CEU_F_DATA = os.path.join(DATA_PATH, "YRI_CEU_folded.fs")
SNP_DATA = os.path.join(DATA_PATH, "data.txt")
DAMAGED_SNP_DATA = os.path.join(DATA_PATH, "damaged_data.txt")
STRANGE_DATA = os.path.join(DATA_PATH, "some_strange_data")


class TestDataHolder(unittest.TestCase):

    def _check_data(self, data, pop_labels, outgroup, sample_sizes):
        self.assertTrue(all(data.sample_sizes == sample_sizes))
        self.assertEqual(data.pop_ids, pop_labels)
        self.assertEqual(not data.folded, outgroup)

    def _load_with_dadi(self, data, size, labels, outgroup):
        warnings.filterwarnings(action="ignore", message="unclosed", 
                         category=ResourceWarning)
        if data.split('.')[-1] == 'txt':
            d = dadi.Misc.make_data_dict(data)
            data = dadi.Spectrum.from_data_dict(d, labels, size, outgroup)
            return data
        data = dadi.Spectrum.from_file(YRI_CEU_DATA)
        if labels == ["CEU", "YRI"]:
            data = np.transpose(data, [1,0])
            data.pop_ids = labels
        if not outgroup:
            data = data.fold()
        data = data.project(size)
        return data
 
    def _test_sfs_reading(self, id):
        sizes = [None, (20,20), (10, 10), (10, 5)]
        outgroup = [None, True, False]
        labels = [None, ["YRI", "CEU"], ["CEU", "YRI"]]
        seq_lens = [None, 1e6]
        data = [YRI_CEU_DATA, SNP_DATA]
        for dat, siz, lab, seq, out in itertools.product(data, sizes, labels,
                                                         seq_lens, outgroup):
            with self.subTest(data=dat, size=siz, labels=lab,
                              seq_len=seq, outgroup=out):
                sfs_holder = SFSDataHolder(dat, siz, out, lab, seq)
                data = get_engine(id).read_data(sfs_holder)
                siz = siz or ((20,20) if (dat == YRI_CEU_DATA) else (24, 44))
                lab = lab or ["YRI", "CEU"]
                out = out or True
                self._check_data(data, lab, out, siz)
                sfs = self._load_with_dadi(dat, siz, lab, out)
                self.assertTrue(np.allclose(data, sfs))

    def _test_read_fails(self, id):
        warnings.filterwarnings(action="ignore", message="unclosed", 
                         category=ResourceWarning)
        data_holder = SFSDataHolder(YRI_CEU_DATA, population_labels=[1, 2])
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(YRI_CEU_DATA, projections=[40, 50])
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(YRI_CEU_F_DATA, outgroup=True)
        self.assertRaises(ValueError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(STRANGE_DATA)
        self.assertRaises(SyntaxError, get_engine(id).read_data, data_holder)
        data_holder = SFSDataHolder(DAMAGED_SNP_DATA)
        self.assertRaises(SyntaxError, get_engine(id).read_data, data_holder)

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_reading(self):
        self._test_sfs_reading('dadi')

    @unittest.skipIf(DADI_NOT_AVAILABLE, "Dadi module is not installed")
    def test_dadi_reading_fails(self):
        self._test_read_fails('dadi')
