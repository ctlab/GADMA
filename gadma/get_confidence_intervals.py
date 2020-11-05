import numpy as np
import sys
import argparse
from scipy import stats
from .utils import abspath, check_file_existence


def main():
    """
    Main function of script. Reads command-line arguments, reads saved table
    and calculate confidence intervals from it.
    """
    import pandas as pd
    parser = argparse.ArgumentParser("GADMA module for calculating confidence "
                                     "intervals from the result table of local"
                                     " search runs on bootstrapped data.")
    parser.add_argument('input_filename', metavar='<filename>',
                        help="Filename (.csv or .pkl) with result from local "
                             "search runs on bootstrapped data. Output of "
                             "gadma-run_ls_on_boot_data.")
    parser.add_argument('--log', required=False, action='store_true',
                        help="If log then logarithm will be used to calculate "
                             "confidence intervals.")
    parser.add_argument('--tex', required=False, action='store_true',
                        help="LaTex output.")
    parser.add_argument('--acc', required=False, metavar='N', type=int,
                        default=5, help="Precision of an output (default: 5).")

    args = parser.parse_args()
    filename = abspath(args.input_filename)
    if not check_file_existence(filename):
        raise ValueError(f"Input file ({filename}) does not exist.")
    ext = filename.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(filename, index_col=0)
    elif ext == 'pkl':
        df = pd.read_pickle(filename)
    else:
        raise ValueError(f"Unknown extension of input file ({filename}). "
                         "Valid extension are: .csv and .pkl")

    a = df.values
    if args.log:
        a = np.log(a)

    means = np.mean(a, axis=0)
    stds = np.std(a, axis=0)

    for m, s, x, par_name in zip(means, stds, a.T, df.columns):
        low = m - 1.96 * s
        upp = m + 1.96 * s
        if args.log:
            low = np.exp(low)
            upp = np.exp(upp)
        k2, p = stats.normaltest(x)
        if p < 0.05:
            normtest_str = 'data looks '
            if args.log:
                normtest_str += 'log-normal (fail to reject H0)'
            else:
                normtest_str += 'normal (fail to reject H0)'
        else:
            normtest_str = 'data does not look '
            if args.log:
                normtest_str += 'log-normal (reject H0)'
            else:
                normtest_str += 'normal (reject H0)'
        normtest_str += f' p-value={p:.2e}'
        if args.tex:
            format_string = '%s:\t$[%.' + str(args.acc) + 'f - %.' +\
                            str(args.acc) + 'f]$\t%s'
        else:
            format_string = '%s:\t%.' + str(args.acc) + 'f\t%.' +\
                            str(args.acc) + 'f\t%s'
        print(format_string % (par_name.strip(), low, upp, normtest_str))
