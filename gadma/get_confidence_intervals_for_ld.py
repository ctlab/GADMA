import argparse
import os.path

from gadma.utils.utils import abspath, \
    check_file_existence, module_name_from_path
import moments.LD
import pickle
import importlib.util
import sys
import copy
import pandas as pd


def main():
    """
    Main function of script for momentsLD CI evaluation.
    Reads read generated data_for_ci.py file and
    calculate confidence intervals from it.
    """
    parser = argparse.ArgumentParser("GADMA module for calculating confidence "
                                     "intervals from calculated LD params")

    parser.add_argument('input_filename', metavar='<filename>',
                        help="Filename (.py) with result from run "
                             "on data. Output of gadma.")

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
        hasattr(module, "rep_data_file"),
        hasattr(module, "param_names")
    ]):
        model_func = getattr(module, "model_func")
        rs = getattr(module, "rs")
        opt_params = getattr(module, "opt_params")
        rep_data_file = getattr(module, "rep_data_file")
        param_names = getattr(module, "param_names")
    else:
        raise ValueError("Data for CI evaluation is not valid! Check it!")

    with open(rep_data_file, "rb") as file:
        region_stats = pickle.load(file)
    data = moments.LD.Parsing.bootstrap_data(region_stats)

    uncerts_fim = moments.LD.Godambe.FIM_uncert(
        model_func,
        opt_params,
        data["means"],
        data["varcovs"],
        r_edges=rs,
    )
    print("uncerts_fim")
    print(uncerts_fim)
    lower_fim = opt_params - 1.96 * uncerts_fim
    upper_fim = opt_params + 1.96 * uncerts_fim

    num_boots = 100
    norm_idx = 0
    bootstrap_sets = moments.LD.Parsing.get_bootstrap_sets(
        region_stats, num_bootstraps=num_boots, normalization=norm_idx)

    uncerts_gim = moments.LD.Godambe.GIM_uncert(
        model_func,
        bootstrap_sets,
        opt_params,
        data["means"],
        data["varcovs"],
        r_edges=rs,
    )

    lower_gim = opt_params - 1.96 * uncerts_gim
    upper_gim = opt_params + 1.96 * uncerts_gim

    lower_fim_phys_units = copy.deepcopy(lower_fim)
    upper_fim_phys_units = copy.deepcopy(upper_fim)
    lower_gim_phys_units = copy.deepcopy(lower_gim)
    upper_gim_phys_units = copy.deepcopy(upper_gim)

    phys_units_boundaries_list = [
        lower_fim_phys_units,
        upper_fim_phys_units,
        lower_gim_phys_units,
        upper_gim_phys_units
    ]

    gen_units_boundaries_list = [
        lower_fim,
        upper_fim,
        lower_gim,
        upper_gim
    ]

    for bound in gen_units_boundaries_list:
        for num, param in enumerate(param_names):
            bound[num] = round(bound[num], 4)

    # opt_params[-1] is Nref
    for bound in phys_units_boundaries_list:
        for num, param in enumerate(param_names):
            if param.startswith("t"):
                bound[num] *= (2 * opt_params[-1])
            elif param.startswith("nu"):
                bound[num] *= opt_params[-1]
            elif param.startswith("m"):
                bound[num] /= (2 * opt_params[-1])
            bound[num] = round(bound[num], 4)

    # create pandas dataframe

    fim_bounds_list = [
        f"{lower_fim[num]} "
        f"- {upper_fim[num]}" for num in range(len(param_names))
    ]
    gim_bounds_list = [
        f"{lower_gim[num]} "
        f"- {upper_gim[num]}" for num in range(len(param_names))
    ]
    fim_bounds_list_phys_units = [
            f"{lower_fim_phys_units[num]} "
            f"- {upper_fim_phys_units[num]}" for num in range(len(param_names))
        ]
    gim_bounds_list_phys_units = [
            f"{lower_gim_phys_units[num]} "
            f"- {upper_gim_phys_units[num]}" for num in range(len(param_names))
        ]

    all_ci_data = {
        "Param names": param_names,
        "Opt params": opt_params,
        "FIM": fim_bounds_list,
        "GIM": gim_bounds_list,
        "FIM phys units": fim_bounds_list_phys_units,
        "GIM phys units": gim_bounds_list_phys_units
    }
    results = os.path.join(os.path.dirname(filename), "ci_results.xlsx")
    all_ci_dataframe = pd.DataFrame(data=all_ci_data)
    all_ci_dataframe.to_csv(results, index=False)
    print(all_ci_dataframe)


if __name__ == "__main__":
    main()
