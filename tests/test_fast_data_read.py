import os
import sys
import gzip
import allel
import shutil
import pickle
import pytest
import tempfile
import unittest
from unittest import mock
import moments.LD
import numpy as np
from os import listdir
import copy
from pathlib import Path
import gadma
from gadma.data.data import VCFDataHolder
from gadma.engines.engine import get_engine
from gadma.data.data_utils import create_bed_files_and_extract_chromosomes
from gadma.data.data_utils import extract_chromosomes_from_vcf, get_chrom2len
from gadma.precompute_ld_data import main
from gadma.engines.moments_ld_engine import _read_data_one_job, create_h5_file
from gadma.engines.moments_ld_engine import extract_rec_map_name_and_extension

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

POP_MAP = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "pop_map.txt")
REC_MAP = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "rec_maps", 'rec_map_1.txt')
TEST_BED_FILE = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "test_bed_files", 'bed_file_1_1.bed')
VCF_DATA_LD = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "vcf_data.vcf")
H5_FILE = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "vcf_data.h5")
FEW_REC_MAPS = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "rec_maps.txt")
TEST_OUTPUT = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "test_output")
VCF_DATA_FEW_CHR = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "vcf_data_few_chr.vcf")
BED_FILES_DIR = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "test_bed_files")

if gadma.moments_LD_available:
    engine = get_engine('momentsLD')
    kwargs = engine.get_kwargs()

    REG_NUM = 1
    PARSING_KWARGS = {
        "vcf_file": VCF_DATA_LD,
        "pop_file": POP_MAP,
        "bed_file": TEST_BED_FILE,
        "chromosome": "1",
        "pops": ['deme0', 'deme1'],
        "rec_map_file": None,
        **kwargs
    }

    DATA_HOLDER = VCFDataHolder(
        vcf_file=VCF_DATA_LD,
        popmap_file=POP_MAP,

    )


class TestFastDataRead(unittest.TestCase):
    def tearDown(self):
        if Path('./preprocessed_data.bp').exists():
            os.remove('./preprocessed_data.bp')

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_rec_map_and_extension_extraction(self):
        rec_map = 'rec_map_1.map'
        self.assertEqual(
            ('rec_map', 'map'),
            extract_rec_map_name_and_extension(rec_map)
        )

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_read_data_without_rec_map_function(self):
        KWARGS = copy.deepcopy(PARSING_KWARGS)
        KWARGS["rec_map_file"] = None
        KWARGS["bp_bins"] = engine.bp_bins

        ld_kwargs = copy.deepcopy(kwargs)
        ld_kwargs["bp_bins"] = engine.bp_bins

        read_function_results = _read_data_one_job(
            [REG_NUM, KWARGS]
        )['1']

        results = moments.LD.Parsing.compute_ld_statistics(
            vcf_file=VCF_DATA_LD,
            pop_file=POP_MAP,
            bed_file=TEST_BED_FILE,
            pops=['deme0', 'deme1'],
            chromosome="1",
            **ld_kwargs
        )

        self.assertEqual(
            len(results['sums']),
            len(read_function_results['sums'])
        )
        for ii in range(len(results['sums'])):
            self.assertTrue(
                np.allclose(
                    results['sums'][ii],
                    read_function_results['sums'][ii]
                )
            )

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_read_data_function(self):
        KWARGS = copy.deepcopy(PARSING_KWARGS)
        KWARGS["rec_map_file"] = REC_MAP
        KWARGS["r_bins"] = engine.r_bins

        ld_kwargs = copy.deepcopy(kwargs)
        ld_kwargs["r_bins"] = engine.r_bins

        read_function_results = _read_data_one_job([REG_NUM, KWARGS])['1']

        results = moments.LD.Parsing.compute_ld_statistics(
            vcf_file=VCF_DATA_LD,
            pop_file=POP_MAP,
            bed_file=TEST_BED_FILE,
            pops=['deme0', 'deme1'],
            rec_map_file=REC_MAP,
            chromosome="1",
            **ld_kwargs
        )
        self.assertTrue(isinstance(results, dict))
        self.assertEqual(
            len(results['sums']),
            len(read_function_results['sums'])
        )
        for ii in range(len(results['sums'])):
            self.assertTrue(
                np.allclose(
                    results['sums'][ii],
                    read_function_results['sums'][ii]
                )
            )

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_read_data_rec_maps_in_one_file_function(self):
        KWARGS = copy.copy(PARSING_KWARGS)
        KWARGS["rec_map_file"] = FEW_REC_MAPS
        KWARGS["r_bins"] = engine.r_bins

        ld_kwargs = copy.copy(kwargs)
        ld_kwargs["r_bins"] = engine.r_bins

        read_function_results = _read_data_one_job(
            [REG_NUM, KWARGS]
        )['1']

        results = moments.LD.Parsing.compute_ld_statistics(
            vcf_file=VCF_DATA_LD,
            pop_file=POP_MAP,
            bed_file=TEST_BED_FILE,
            rec_map_file=FEW_REC_MAPS,
            pops=['deme0', 'deme1'],
            chromosome="1",
            map_name="1",
            **ld_kwargs
        )

        self.assertEqual(
            len(results['sums']),
            len(read_function_results['sums'])
        )
        for ii in range(len(results['sums'])):
            self.assertTrue(
                np.allclose(
                    results['sums'][ii],
                    read_function_results['sums'][ii]
                )
            )

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    @pytest.mark.timeout(0)
    def test_main_func(self):
        param_file = os.path.join(DATA_PATH, 'PARAMS', 'another_test_params')
        sys.argv = ['python', '-p', param_file]
        self.assertRaises(ValueError, main)

        data_reading_case_list = ['without_rec_map', 'with_rec_map', 'with_rec_maps_in_one_file', 'with_rec_rate']
        for case in data_reading_case_list:
            param_file = os.path.join(
                DATA_PATH, 'PARAMS', 'ld_data_parsing_params',
                f'ld_data_parsings_params_{case}'
            )
            sys.argv = ['gadma-parsing_ld_stats', '-p', param_file]
            try:
                main()
                self.assertTrue(Path('./preprocessed_data.bp').exists())
                with open('./preprocessed_data.bp', 'rb') as fin:
                    region_stats_evaluated, data_evaluated = pickle.load(fin)

                preprocessed_test_data = os.path.join(
                    DATA_PATH, 'DATA', 'vcf_ld',
                    f'parsing_test_data_{case}.bp'
                )

                with open(preprocessed_test_data, 'rb') as file:
                    region_stats_stored, data_stored = pickle.load(file)

                self.assertEqual(
                    len(region_stats_evaluated['0']),
                    len(region_stats_stored['0'])
                )
                # Was evaluated for eld version of momentsLD - differ a little
                for region in range(15):
                    for ii in range(len(region_stats_evaluated[f'{region}']['sums'])):
                        self.assertTrue(
                            np.allclose(
                                region_stats_evaluated[f'{region}']['sums'][ii],
                                region_stats_stored[f'{region}']['sums'][ii],
                                rtol=1e-1
                            )
                        )
            finally:
                rewrite_params_file(param_file)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_h5_creation(self):
        vcf_file = VCF_DATA_LD
        if Path(H5_FILE).exists():
            os.remove(H5_FILE)
        self.assertTrue(not Path(H5_FILE).exists())
        create_h5_file(vcf_file)
        self.assertTrue(Path(H5_FILE).exists())

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_h5_file_after_data_reading(self):
        """
        H5 file that is created to read LD statistics from VCF file is
        removed in the end by the name it was created with. H5 file that
        existed before the reading is kept.
        """
        from gadma.engines import moments_ld_engine
        ld_engine = get_engine('momentsLD')
        dirname = tempfile.mkdtemp()

        def read_data(vcf_file):
            """Reads data and returns h5 files that were used for it."""
            h5_files = []

            # evaluation of statistics takes time, we do not need it here
            def one_job(args):
                h5_files.append(sorted(name for name in os.listdir(dirname)
                                       if name.endswith(".h5")))
                return {str(args[0]): {}}

            data_holder = VCFDataHolder(vcf_file=vcf_file,
                                        popmap_file=POP_MAP,
                                        bed_files_dir=BED_FILES_DIR)
            with mock.patch.object(moments_ld_engine, "_read_data_one_job",
                                   side_effect=one_job), \
                    mock.patch("moments.LD.Parsing.bootstrap_data",
                               side_effect=lambda stats: stats):
                ld_engine.read_data(data_holder)
            return h5_files[0]

        try:
            # h5 file has another name than VCF file without its extension
            vcf_file = os.path.join(dirname, "x.vcf.filtered.vcf")
            shutil.copy(VCF_DATA_LD, vcf_file)
            self.assertEqual(read_data(vcf_file), ["x.h5"])
            self.assertEqual(os.listdir(dirname), ["x.vcf.filtered.vcf"])

            # h5 file that was created before
            self.assertTrue(create_h5_file(vcf_file))
            self.assertFalse(create_h5_file(vcf_file))
            self.assertEqual(read_data(vcf_file), ["x.h5"])
            self.assertEqual(sorted(os.listdir(dirname)),
                             ["x.h5", "x.vcf.filtered.vcf"])
        finally:
            shutil.rmtree(dirname)


class TestChromosomesOfVCF(unittest.TestCase):
    def test_chromosomes_of_vcf(self):
        """
        Chromosomes are taken from the first column of VCF file and they are
        the same as scikit-allel gives. The file could be compressed and
        could have empty lines.
        """
        vcf_with_20_chr = os.path.join(
            DATA_PATH, 'DATA', 'vcf_ld', "YRI_CEU_sim_data.vcf")
        for vcf_file, n_chrom in [(VCF_DATA_LD, 1), (VCF_DATA_FEW_CHR, 2),
                                  (vcf_with_20_chr, 20)]:
            chromosomes = extract_chromosomes_from_vcf(vcf_file)
            vcf_data = allel.read_vcf(vcf_file, fields=["variants/CHROM"])
            self.assertEqual(chromosomes,
                             sorted(set(vcf_data["variants/CHROM"])))
            self.assertEqual(len(chromosomes), n_chrom)

        with open(VCF_DATA_FEW_CHR, "rb") as fl:
            lines = fl.readlines()
        dirname = tempfile.mkdtemp()
        try:
            # compressed file
            vcf_file = os.path.join(dirname, "data.vcf.gz")
            with gzip.open(vcf_file, "wb") as fl:
                fl.writelines(lines)
            self.assertEqual(extract_chromosomes_from_vcf(vcf_file),
                             ["1", "2"])
            # empty lines are not chromosomes
            vcf_file = os.path.join(dirname, "empty_lines.vcf")
            with open(vcf_file, "wb") as fl:
                fl.writelines(lines + [b"\n", b"\n"])
            self.assertEqual(extract_chromosomes_from_vcf(vcf_file),
                             ["1", "2"])
            # file without variants
            vcf_file = os.path.join(dirname, "no_variants.vcf")
            with open(vcf_file, "wb") as fl:
                fl.writelines(
                    [line for line in lines if line.startswith(b"#")])
            self.assertRaises(ValueError, extract_chromosomes_from_vcf,
                              vcf_file)
        finally:
            shutil.rmtree(dirname)

    def test_vcf_is_read_once_for_chromosomes(self):
        """
        Chromosomes of VCF file are remembered: the file is read for them
        once and one more time only if it was changed. Genotypes are not
        loaded to get lengths of chromosomes.
        """
        dirname = tempfile.mkdtemp()
        vcf_file = os.path.join(dirname, "data.vcf")
        opened = []
        loaded_fields = []
        builtin_open = open
        read_vcf = allel.read_vcf

        def open_and_remember(file, *args, **kwargs):
            if file == vcf_file:
                opened.append(file)
            return builtin_open(file, *args, **kwargs)

        def read_vcf_and_remember(*args, **kwargs):
            vcf_data = read_vcf(*args, **kwargs)
            loaded_fields.append(sorted(vcf_data.keys()))
            return vcf_data

        try:
            shutil.copyfile(VCF_DATA_FEW_CHR, vcf_file)
            with mock.patch("builtins.open", open_and_remember), \
                    mock.patch("allel.read_vcf", read_vcf_and_remember):
                for _ in range(3):
                    chromosomes = extract_chromosomes_from_vcf(vcf_file)
                    self.assertEqual(chromosomes, ["1", "2"])
                    # change of the returned list changes nothing
                    chromosomes.append("3")
                self.assertEqual(len(opened), 1)

                # lengths of chromosomes are checked by positions of
                # variants, genotypes are not needed for it
                lengths = {"1": 1000000, "2": 1000000}
                data_holder = VCFDataHolder(
                    vcf_file=vcf_file,
                    popmap_file=POP_MAP,
                    sequence_length=lengths,
                )
                self.assertEqual(get_chrom2len(data_holder), lengths)
                self.assertEqual(loaded_fields,
                                 [["variants/CHROM", "variants/POS"]])

                # the file is changed, so it is read again
                shutil.copyfile(VCF_DATA_LD, vcf_file)
                n_opened = len(opened)
                for _ in range(3):
                    self.assertEqual(extract_chromosomes_from_vcf(vcf_file),
                                     ["1"])
                self.assertEqual(len(opened), n_opened + 1)
        finally:
            shutil.rmtree(dirname)


class TestBedFilesCreation(unittest.TestCase):
    def tearDown(self):
        if Path(f"{TEST_OUTPUT}").exists():
            shutil.rmtree(f"{TEST_OUTPUT}")

    data_holder = VCFDataHolder(
        vcf_file=VCF_DATA_LD,
        popmap_file=POP_MAP,
    )
    output_dir = os.path.join(TEST_OUTPUT, "auto_bed_files")
    region_len = 50000
    vcf_data = allel.read_vcf(data_holder.filename)
    chromosome_len = max(vcf_data['variants/POS'])

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_one_chrom_normal_region_num(self):
        self.data_holder.filename = VCF_DATA_LD
        self.data_holder.sequence_length = 1000000
        self.region_len = 50000

        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder,
            output_dir=self.output_dir,
            region_len=self.region_len
        )
        self.assertEqual(1, len(chromosomes))
        regions_num = len(
            listdir(self.output_dir)
        )
        self.assertEqual(regions_num, round(
            self.chromosome_len / self.region_len))

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_one_chrom_small_region_num(self):
        self.data_holder.filename = VCF_DATA_LD
        self.data_holder.sequence_length = 10000000
        self.region_len = 100000000

        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder,
            output_dir=self.output_dir,
            region_len=self.region_len,
        )
        self.assertEqual(1, len(chromosomes))
        regions_num = len(
            listdir(self.output_dir)
        )
        self.assertEqual(regions_num, 15)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_few_chrom_normal_region_num(self):
        self.data_holder.filename = VCF_DATA_FEW_CHR
        self.data_holder.sequence_length = {"1": 1000000, "2": 1000000}
        self.region_len = 50000

        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder,
            output_dir=self.output_dir,
            region_len=self.region_len,
        )
        vcf_data = allel.read_vcf(self.data_holder.filename)
        chromosome_len = max(vcf_data['variants/POS'])
        regions_num = len(
            listdir(self.output_dir)
        )
        self.assertEqual(
            len(set(vcf_data['variants/CHROM'])), len(chromosomes))
        self.assertEqual(regions_num, round(
            self.chromosome_len / self.region_len) * 2)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_few_chrom_small_region_num(self):
        self.data_holder.filename = VCF_DATA_FEW_CHR
        self.region_len = 500000
        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder,
            output_dir=self.output_dir,
            region_len=self.region_len,
        )
        vcf_data = allel.read_vcf(self.data_holder.filename)
        self.assertEqual(
            len(set(vcf_data['variants/CHROM'])), len(chromosomes))
        regions_num = len(
            listdir(self.output_dir)
        )
        self.assertEqual(regions_num, 16)


def rewrite_params_file(params_file):
    remove_line = 'preprocessed_data: ./preprocessed_data.bp\n'
    with open(params_file, 'r') as file:
        lines = file.readlines()
    with open(params_file, 'w') as file:
        for line in lines:
            if line != remove_line:
                file.write(line)
