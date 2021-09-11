import collections
import multiprocessing
from collections import ChainMap
import moments.LD
import numpy as np
from os import listdir
import time
import pickle
from gadma.utils.utils import create_bed_files_and_extract_chromosomes
from gadma.cli import arg_parser
from gadma.data.data import VCFDataHolder
from gadma.data.data_utils import check_and_return_projections_and_labels
from gadma.engines.engine import get_engine


def main(chrom_ii_pair_list, n_processes=6):
    t2 = time.time()
    pool = multiprocessing.Pool(processes=n_processes)
    result = pool.map(read_data, chrom_ii_pair_list)
    pool.close()
    regions = dict(ChainMap(*result))

    ld_stats = moments.LD.Parsing.bootstrap_data(regions)
    with open("./preprocessed_data.bp", "wb+") as fout:
        pickle.dump(ld_stats, fout)

    params_file = vars(args)["params"]
    with open(f"./{params_file}", "a") as params:
        params.write("preprocessed_data: ./preprocessed_data.bp\n")



def read_data(item):
    results = {
        f"{item.reg_num}":
            moments.LD.Parsing.compute_ld_statistics(
                vcf_file=item.filename,
                rec_map_file=item.rec_map,
                pop_file=item.pop_file,
                bed_file=item.bed_file,
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


if __name__ == "__main__":
    settings_storage, args = arg_parser.get_settings()
    print("Data reading")
    data_holder = settings_storage.data_holder

    if isinstance(data_holder, VCFDataHolder):
        projections, pops = check_and_return_projections_and_labels(data_holder)
    else:
        raise ValueError("Wrong type of data_holder: "
                         f"{data_holder.__class__}")

    engine_name = settings_storage.engine
    engine = get_engine(engine_name)

    kwargs = engine.kwargs

    if data_holder.ld_kwargs:
        for key in data_holder.ld_kwargs:
            try:
                kwargs[key] = eval(data_holder.ld_kwargs[key])
            except:  # NOQA
                kwargs[key] = data_holder.ld_kwargs[key]

    chromosomes = create_bed_files_and_extract_chromosomes(data_holder)

    read_info = collections.namedtuple(
        "read_info", [
            "reg_num", "filename", "rec_map",
            "bed_file", "pop_file", "pops",
            "kwargs"
        ]
    )
    rec_map = listdir(data_holder.recombination_maps)[0]
    rec_map_name, extension = extract_rec_map_name_and_extension(rec_map)
    bed_files = data_holder.output_directory + "/bed_files/"

    read_information_list = []

    region_number = 1

    filename = data_holder.filename
    popmap_file = data_holder.popmap_file
    for chrom in chromosomes:
        rec_map = f"{data_holder.recombination_maps}/" \
                  f"{rec_map_name}_{chrom}.{extension}"
        for ii in range(1, chromosomes[chrom]):
            bed_file = f"{bed_files}bed_file_{chrom}_{ii}.bed"
            read_information_list.append(
                read_info(
                    reg_num=region_number,
                    filename=filename,
                    rec_map=rec_map,
                    bed_file=bed_file,
                    pop_file=popmap_file,
                    pops=pops,
                    kwargs=kwargs
                )
            )
            region_number += 1

    n_processes = settings_storage.number_of_processes

    main(read_information_list, n_processes)

