import multiprocessing
from collections import ChainMap
import moments.LD
import shutil
import allel
from os import listdir
import pickle
from gadma.cli import SettingsStorage
from gadma.data.data import VCFDataHolder
from gadma.engines.engine import get_engine
from gadma.utils import ensure_dir_existence
from gadma.cli import arg_parser
import os
import gadma
gadma.cli.arg_parser.version = lambda: "GADMA module for data preprocessing "\
                                       "with momentsLD engine\n"
gadma.cli.arg_parser.tool_name = "gadma-precompute_ld_data"


def main():
    settings_storage, args = arg_parser.get_settings()

    if isinstance(settings_storage.data_holder, VCFDataHolder):
        pops = settings_storage.data_holder.population_labels
    else:
        raise ValueError("Wrong type of data_holder: "
                         f"{settings_storage.data_holder.__class__}")
    ensure_dir_existence(
        settings_storage.output_directory, check_emptiness=True
    )
    assert settings_storage.engine == "momentsLD"

    try:
        print("Data reading")
        settings_storage._prepare_for_moments_ld()
        engine = get_engine(settings_storage.engine)

        regions = engine._get_region_stats(settings_storage.data_holder)

        with open("./preprocessed_data.bp", "wb+") as fout:
            pickle.dump(regions, fout)

        params_file = vars(args)["params"]
        with open(f"{params_file}", "a") as params:
            params.write("\npreprocessed_data: ./preprocessed_data.bp\n")
    finally:
        shutil.rmtree(settings_storage.output_directory)
