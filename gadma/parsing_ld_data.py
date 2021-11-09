import multiprocessing
from collections import ChainMap
import moments.LD
import shutil
import allel
from os import listdir
import pickle
from gadma.utils.utils import create_bed_files_and_extract_chromosomes,\
    check_file_existence
from gadma.cli import arg_parser
from gadma.data.data import VCFDataHolder
from gadma.engines.engine import get_engine


def main():
    settings_storage, args = arg_parser.get_settings()
    print("Data reading")
    data_holder = settings_storage.data_holder

    if isinstance(data_holder, VCFDataHolder):
        pops = data_holder.population_labels
    else:
        raise ValueError("Wrong type of data_holder: "
                         f"{data_holder.__class__}")

    engine = get_engine(settings_storage.engine)
    kwargs = engine.kwargs

    if data_holder.ld_kwargs:
        for key in data_holder.ld_kwargs:
            try:
                kwargs[key] = eval(data_holder.ld_kwargs[key])
            except:  # NOQA
                kwargs[key] = data_holder.ld_kwargs[key]

    chromosomes = create_bed_files_and_extract_chromosomes(data_holder)

    if data_holder.recombination_maps:
        rec_map = listdir(data_holder.recombination_maps)[0]
        rec_map_name, extension = extract_rec_map_name_and_extension(rec_map)
        rec_map_exist = True
        one_rec_map_few_chromosomes = False
        if (len(listdir(data_holder.recombination_maps)) == 1 and
                len(chromosomes) > 1):
            one_rec_map_few_chromosomes = True
            rec_map_name = listdir(data_holder.recombination_maps)[0]
    else:
        rec_map_exist = False

    bed_files = data_holder.output_directory + "/bed_files/"
    read_information_list = []
    region_number = 0

    filename = data_holder.filename
    popmap_file = data_holder.popmap_file
    sorted_choromosomes_list = sorted(chrom for chrom in chromosomes)
    for chrom in sorted_choromosomes_list:
        if rec_map_exist:
            if one_rec_map_few_chromosomes:
                rec_map = f"{data_holder.recombination_maps}/" \
                          f"{rec_map_name}"
            else:
                rec_map = f"{data_holder.recombination_maps}/" \
                          f"{rec_map_name}_{chrom}.{extension}"
        else:
            rec_map = None
        for ii in range(1, chromosomes[chrom] + 1):
            bed_file = f"{bed_files}bed_file_{chrom}_{ii}.bed"
            read_information_list.append(
                ReadInfo(
                    reg_num=region_number,
                    filename=filename,
                    rec_map=rec_map,
                    bed_file=bed_file,
                    pop_file=popmap_file,
                    pops=pops,
                    kwargs=kwargs,
                    chromosome=chrom
                )
            )
            region_number += 1

    create_h5_file(data_holder.filename)
    n_processes = settings_storage.number_of_processes
    pool = multiprocessing.Pool(processes=n_processes)

    try:
        number_of_rec_maps = len(listdir(data_holder.recombination_maps))
        if data_holder.recombination_maps is not None:
            if number_of_rec_maps == len(chromosomes):
                result = pool.map(read_data, read_information_list)
            else:
                result = pool.map(
                    read_data_rec_maps_in_one_file, read_information_list
                )
        else:
            result = pool.map(
                read_data_without_rec_map, read_information_list
            )

        pool.close()
        regions = dict(ChainMap(*result))

        with open("./preprocessed_data.bp", "wb+") as fout:
            pickle.dump(regions, fout)

        params_file = vars(args)["params"]
        with open(f"{params_file}", "a") as params:
            params.write("\npreprocessed_data: ./preprocessed_data.bp\n")
    finally:
        shutil.rmtree(settings_storage.output_directory)


def read_data(item):
    """
    Function for reading data using multiprocessing
    :param item: ReadInfo object contains information about region to read
    """
    results = {
        f"{item.reg_num}":
            moments.LD.Parsing.compute_ld_statistics(
                vcf_file=item.filename,
                rec_map_file=item.rec_map,
                pop_file=item.pop_file,
                bed_file=item.bed_file,
                pops=item.pops,
                chromosome=item.chromosome,
                **item.kwargs
            )
    }
    return results


def read_data_without_rec_map(item):
    """
    Function for reading data using multiprocessing without recombination map
    :param item: ReadInfo object contains information about region to read
    """
    results = {
        f"{item.reg_num}":
            moments.LD.Parsing.compute_ld_statistics(
                vcf_file=item.filename,
                pop_file=item.pop_file,
                bed_file=item.bed_file,
                pops=item.pops,
                **item.kwargs
            )
    }

    return results


def read_data_rec_maps_in_one_file(item):
    """
    Function for reading data using multiprocessing
    with one file containing few recombination maps
    :param item: ReadInfo object contains information about region to read
    """
    results = {
        f"{item.reg_num}":
            moments.LD.Parsing.compute_ld_statistics(
                vcf_file=item.filename,
                rec_map_file=item.rec_map,
                map_name=item.chromosome,
                pop_file=item.pop_file,
                bed_file=item.bed_file,
                chromosome=item.chromosome,
                pops=item.pops,
                **item.kwargs
            )
    }
    return results


def extract_rec_map_name_and_extension(rec_map):
    extension = rec_map.split(".")[1]
    rec_map_name = rec_map.split(".")[0]
    rec_map_name = "_".join(rec_map_name.split('_')[:-1])
    return rec_map_name, extension


def create_h5_file(vcf_file):
    h5_file_path = vcf_file.split(".vcf")[0] + ".h5"
    if not check_file_existence(h5_file_path):
        allel.vcf_to_hdf5(
            vcf_file,
            h5_file_path,
            fields="*",
            exclude_fields=["calldata/GQ"],
            overwrite=True,
        )


class ReadInfo:
    """
    Class for storing data about region used in Parsing LD stats
    """
    def __init__(
            self, reg_num, filename, rec_map,
            bed_file, pop_file, pops, kwargs,
            chromosome
    ):
        self.reg_num = reg_num
        self.filename = filename
        self.rec_map = rec_map
        self.bed_file = bed_file
        self.pop_file = pop_file
        self.pops = pops
        self.kwargs = kwargs
        self.chromosome = chromosome


if __name__ == "__main__":
    main()
