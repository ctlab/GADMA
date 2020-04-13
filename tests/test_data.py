import unittest
import itertools
import os
import numpy as np
from gadma import *

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
YRI_CEU_DATA = os.path.join(DATA_PATH, "YRI_CEU.fs")
YRI_CEU_F_DATA = os.path.join(DATA_PATH, "YRI_CEU_folded.fs")
SNP_DATA = os.path.join(DATA_PATH, "data.txt")
DAMAGED_SNP_DATA = os.path.join(DATA_PATH, "damaged_data.txt")
STRANGE_DATA = os.path.join(DATA_PATH, "some_strange_data")


class TestDataHolder(unittest.TestCase):

    def _check_data_holder(self, data_holder, pop_labels, seq_len, outgroup, sample_sizes):
        self.assertTrue(all(data_holder.sample_sizes == sample_sizes))
        self.assertEqual(data_holder.pop_labels, pop_labels)
        self.assertEqual(data_holder.seq_len, seq_len)
        self.assertEqual(data_holder.outgroup, outgroup)

    def _load_yri_ceu_dadi(self, size, labels, outgroup):
        import dadi
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
        for siz, lab, seq, out in itertools.product(sizes, labels, seq_lens, outgroup):
            sfs_holder = SFSDataHolder(YRI_CEU_DATA, out, lab, seq, siz)
            sfs_holder.prepare_for_engine(get_engine(id))
            siz = siz or (20,20)
            lab = lab or ["YRI", "CEU"]
            out = out or True
            self._check_data_holder(sfs_holder, lab, seq, out, siz)
            self.assertTrue(np.allclose(sfs_holder.data,self._load_yri_ceu_dadi(siz, lab, out)))

    def _test_read_fails(self, id):
        data_holder = SFSDataHolder(YRI_CEU_DATA, pop_labels=[1, 2])
        self.assertRaises(ValueError, data_holder.prepare_for_engine, get_engine(id))
        data_holder = SFSDataHolder(YRI_CEU_DATA, sample_sizes=[40, 50])
        self.assertRaises(ValueError, data_holder.prepare_for_engine, get_engine(id))
        data_holder = SFSDataHolder(YRI_CEU_F_DATA, outgroup=True)
        self.assertRaises(ValueError, data_holder.prepare_for_engine, get_engine(id))
        data_holder = SFSDataHolder(STRANGE_DATA)
        self.assertRaises(SyntaxError, data_holder.prepare_for_engine, get_engine(id))
        data_holder = SFSDataHolder(DAMAGED_SNP_DATA)
        self.assertRaises(SyntaxError, data_holder.prepare_for_engine, get_engine(id))

    def test_dadi_reading(self):
        self._test_sfs_reading('dadi')

    def test_dadi_reading_fails(self):
        self._test_read_fails('dadi')
