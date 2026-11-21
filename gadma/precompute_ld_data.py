import moments.LD
import os
import shutil
import pickle
from gadma.data.data import VCFDataHolder
from gadma.engines.engine import get_engine
from gadma.utils import ensure_dir_existence
from gadma.cli import arg_parser
import gadma
gadma.cli.arg_parser.version = lambda: "GADMA module for data preprocessing "\
                                       "with momentsLD engine\n"
gadma.cli.arg_parser.tool_name = "gadma-precompute_ld_data"


def main():
    settings_storage, args = arg_parser.get_settings()

    if not isinstance(settings_storage.data_holder, VCFDataHolder):
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
        data = moments.LD.Parsing.bootstrap_data(regions)

        preprocessed_data = os.path.abspath("preprocessed_data.bp")
        with open(preprocessed_data, "wb+") as fout:
            pickle.dump(tuple([regions, data]), fout)

        # The file with data is set in the params file. It is set once: the
        # line of the previous run is replaced, as the params file could
        # not be read with two such lines.
        params_file = vars(args)["params"]
        with open(params_file) as fl:
            lines = [line for line in fl
                     if not line.strip().lower().replace(" ", "_").startswith(
                         "preprocessed_data")]
        while len(lines) > 0 and len(lines[-1].strip()) == 0:
            lines.pop()
        if len(lines) > 0 and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"\npreprocessed_data: {preprocessed_data}\n")
        with open(params_file, "w") as fl:
            fl.writelines(lines)
    finally:
        shutil.rmtree(settings_storage.output_directory)
