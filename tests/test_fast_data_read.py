import os
import sys
import allel
import shutil
import pickle
import pytest
import unittest
import moments.LD
import numpy as np
from os import listdir
from pathlib import Path
import gadma
from gadma.data.data import VCFDataHolder
from gadma.engines.engine import get_engine
from gadma.data.data_utils import create_bed_files_and_extract_chromosomes
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
    kwargs = engine.kwargs

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
        PARSING_KWARGS["rec_map_file"] = None
        read_function_results = _read_data_one_job(
            [REG_NUM, PARSING_KWARGS]
        )['1']

        results = moments.LD.Parsing.compute_ld_statistics(
            vcf_file=VCF_DATA_LD,
            pop_file=POP_MAP,
            bed_file=TEST_BED_FILE,
            pops=['deme0', 'deme1'],
            chromosome="1",
            **kwargs
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
        PARSING_KWARGS["rec_map_file"] = REC_MAP

        read_function_results = _read_data_one_job([REG_NUM, PARSING_KWARGS])['1']

        results = moments.LD.Parsing.compute_ld_statistics(
            vcf_file=VCF_DATA_LD,
            pop_file=POP_MAP,
            bed_file=TEST_BED_FILE,
            pops=['deme0', 'deme1'],
            rec_map_file=REC_MAP,
            chromosome="1",
            **kwargs
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
        PARSING_KWARGS["rec_map_file"] = FEW_REC_MAPS

        read_function_results = _read_data_one_job(
            [REG_NUM, PARSING_KWARGS]
        )['1']

        results = moments.LD.Parsing.compute_ld_statistics(
            vcf_file=VCF_DATA_LD,
            pop_file=POP_MAP,
            bed_file=TEST_BED_FILE,
            rec_map_file=FEW_REC_MAPS,
            pops=['deme0', 'deme1'],
            chromosome="1",
            map_name="1",
            **kwargs
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
                    region_stats = pickle.load(fin)

                preprocessed_test_data = os.path.join(
                    DATA_PATH, 'DATA', 'vcf_ld',
                    f'parsing_test_data_{case}.bp'
                )

                with open(preprocessed_test_data, 'rb') as file:
                    region_stats_moments_ld = pickle.load(file)

                self.assertEqual(
                    len(region_stats['0']),
                    len(region_stats_moments_ld['0'])
                )
                for region in range(15):
                    for ii in range(len(region_stats[f'{region}']['sums'])):
                        self.assertTrue(
                            np.allclose(
                                region_stats[f'{region}']['sums'][ii],
                                region_stats_moments_ld[f'{region}']['sums'][ii]
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
