import os
import sys
import allel
import shutil
import unittest
import moments.LD
import numpy as np
from os import listdir
from pathlib import Path
from gadma.data.data import VCFDataHolder
from gadma.engines.engine import get_engine
from gadma.utils.utils import create_bed_files_and_extract_chromosomes
from gadma.parsing_ld_data import (
    main, read_data, extract_rec_map_name_and_extension, ReadInfo
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

POP_MAP = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "pop_map.txt")
REC_MAP = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "rec_maps", 'rec_map_1.txt')
TEST_BED_FILE = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "test_bed_files", 'bed_file_1_1.bed')
VCF_DATA_LD = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "vcf_data.vcf")
FEW_REC_MAPS = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "rec_maps.txt")
TEST_OUTPUT = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "test_output")
VCF_DATA_FEW_CHR = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "vcf_data_few_chr.vcf")

engine = get_engine('momentsLD')
kwargs = engine.kwargs

read_info = ReadInfo(
    reg_num=1,
    filename=VCF_DATA_LD,
    pop_file=POP_MAP,
    bed_file=TEST_BED_FILE,
    chromosome="1",
    pops=['deme0', 'deme1'],
    rec_map=None,
    kwargs=kwargs
)


class TestFastDataRead(unittest.TestCase):

    def test_rec_map_and_extension_extraction(self):
        rec_map = 'rec_map_1.map'
        self.assertEqual(
            ('rec_map', 'map'),
            extract_rec_map_name_and_extension(rec_map)
        )

    def test_read_data_without_rec_map_function(self):
        read_info.rec_map = None
        read_function_results = read_data(read_info)['1']

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

    def test_read_data_function(self):
        read_info.rec_map = REC_MAP

        read_function_results = read_data(read_info)['1']

        results = moments.LD.Parsing.compute_ld_statistics(
            vcf_file=VCF_DATA_LD,
            pop_file=POP_MAP,
            bed_file=TEST_BED_FILE,
            pops=['deme0', 'deme1'],
            rec_map_file=REC_MAP,
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

    def test_read_data_rec_maps_in_one_file_function(self):
        read_info.rec_map = FEW_REC_MAPS

        read_function_results = read_data(read_info)['1']

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

    def test_main_func(self):
        param_file = os.path.join(DATA_PATH, "PARAMS", 'another_test_params')
        sys.argv = ['python', '-p', param_file]
        self.assertRaises(ValueError, main)


class TestBedFilesCreation(unittest.TestCase):
    def tearDown(self):
        if Path(f"{TEST_OUTPUT}").exists():
            shutil.rmtree(f"{TEST_OUTPUT}")

    data_holder = VCFDataHolder(
        vcf_file=VCF_DATA_LD,
        popmap_file=POP_MAP,
        output_directory=TEST_OUTPUT,
        region_len=50000
    )
    vcf_data = allel.read_vcf(data_holder.filename)
    chromosome_len = max(vcf_data['variants/POS'])

    def test_one_chrom_normal_region_num(self):
        self.data_holder.filename = VCF_DATA_LD
        self.data_holder.region_len = 50000

        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder)
        self.assertEqual(1, len(chromosomes))
        regions_num = len(
            listdir(f'{self.data_holder.output_directory}/bed_files/'))
        self.assertEqual(regions_num, round(
            self.chromosome_len/self.data_holder.region_len))

    def test_one_chrom_small_region_num(self):
        self.data_holder.filename = VCF_DATA_LD
        self.data_holder.region_len = 100000000
        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder)
        self.assertEqual(1, len(chromosomes))
        regions_num = len(
            listdir(f'{self.data_holder.output_directory}/bed_files/'))
        self.assertEqual(regions_num, 15)

    def test_few_chrom_normal_region_num(self):
        self.data_holder.filename = VCF_DATA_FEW_CHR
        self.data_holder.region_len = 50000

        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder)
        vcf_data = allel.read_vcf(self.data_holder.filename)
        chromosome_len = max(vcf_data['variants/POS'])
        regions_num = len(
            listdir(f'{self.data_holder.output_directory}/bed_files/'))
        self.assertEqual(
            len(set(vcf_data['variants/CHROM'])), len(chromosomes))
        self.assertEqual(regions_num, round(
            self.chromosome_len/self.data_holder.region_len) * 4)

    def test_few_chrom_small_region_num(self):
        self.data_holder.filename = VCF_DATA_FEW_CHR
        self.data_holder.region_len = 500000
        chromosomes = create_bed_files_and_extract_chromosomes(
            data_holder=self.data_holder)
        vcf_data = allel.read_vcf(self.data_holder.filename)
        self.assertEqual(
            len(set(vcf_data['variants/CHROM'])), len(chromosomes))
        regions_num = len(
            listdir(f'{self.data_holder.output_directory}/bed_files/'))
        self.assertEqual(regions_num, 16)
