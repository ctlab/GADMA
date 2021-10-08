import argparse
from gadma.utils.utils import abspath, check_file_existence, module_name_from_path
import moments.LD
import pickle
import importlib.util
import sys

# Need to import lib
#
print("Hello")


def main():
    print("START ARGPARSE")
    parser = argparse.ArgumentParser("GADMA module for calculating confidence "
                                     "intervals from calculated LD params")

    parser.add_argument('input_filename', metavar='<filename>',
                        help="Filename (.py) with result from run "
                             "on data. Output of gadma.")
    parser.add_argument('--method',required=True, type=str,
                        help='Method for CI evaluation. '
                             'You can choose Fisher Information Matrix (FIM) or '
                             'the Godambe Information Matrix (GIM)')

    args = parser.parse_args()
    filename = abspath(args.input_filename)
    if not check_file_existence(filename):
        raise ValueError(f"Input file ({filename}) does not exist.")
    module_name = module_name_from_path(filename)
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    if all([
        hasattr(module, "model_func"),
        hasattr(module, "rs"),
        hasattr(module, "opt_params"),
        hasattr(module, "rep_data_file")
    ]):
        model_func = getattr(module, "model_func")
        rs = getattr(module, "rs")
        opt_params = getattr(module, "opt_params")
        rep_data_file = getattr(module, "rep_data_file")
    else:
        raise ValueError("Data for CI evaluation is not valid! Check it!")

    with open(rep_data_file, "rb") as file:
        region_stats = pickle.load(file)
    data = moments.LD.Parsing.bootstrap_data(region_stats)

    if args.method == "FIM":
        uncerts_FIM = moments.LD.Godambe.FIM_uncert(
            model_func,
            opt_params,
            data["means"],
            data["varcovs"],
            r_edges=rs,
        )
        lower = opt_params - 1.96 * uncerts_FIM
        upper = opt_params + 1.96 * uncerts_FIM

        print(lower)
        print(upper)
    elif args.method == "GIM":
        num_boots = 100
        norm_idx = 0
        bootstrap_sets = moments.LD.Parsing.get_bootstrap_sets(
            region_stats, num_bootstraps=num_boots, normalization=norm_idx)

        uncerts_GIM = moments.LD.Godambe.GIM_uncert(
            model_func,
            bootstrap_sets,
            opt_params,
            data["means"],
            data["varcovs"],
            r_edges=rs,
        )

        lower = opt_params - 1.96 * uncerts_GIM
        upper = opt_params + 1.96 * uncerts_GIM
        print(lower)
        print(upper)

    else:
        raise ValueError("Unknown method for CI.")


if __name__ == "__main__":
    main()
