import numpy as np
import pandas as pd
import sys
import argparse
from scipy import stats
from gadma import support

def main():
    parser = argparse.ArgumentParser('GADMA module for calculating confidence intervals from the result table of local search runs on bootstrapped data.')
    parser.add_argument('input_filename', metavar='<filename>', help='Filename (.csv or .pkl) with result from local search runs on bootstrapped data. Output of gadma-run_ls_on_boot_data.')
    parser.add_argument('--log', required=False, action='store_true', help='If log then logarithm will be used to calculate confidence intervals.')
    parser.add_argument('--tex', required=False, action='store_true', help='Tex output.')
    parser.add_argument('--acc', required=False, metavar='N', type=int, default=5, help='Accuracy of output (dafault: 5).')

    args = parser.parse_args()
    filename = support.check_file_existence(args.input_filename)
    ext = filename.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(filename, index_col = 0)
    elif ext == 'pkl':
        df = pd.read_pickle(filename)

    a = df.values
    if args.log:
        a = np.log(a)

    means = np.mean(a, axis=0)
    stds = np.std(a, axis=0)

    for m,s, x,par_name in zip(means, stds, a.T, df.columns):
        l = m - 1.96 * s
        u = m + 1.96 * s
        if args.log:
            l = np.exp(l)
            u = np.exp(u)
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
        normtest_str += ' p-value=%.2e' % p
        if args.tex:
            format_string = '%s:\t$[%.' + str(args.acc) + 'f - %.' + str(args.acc) + 'f]$\t%s'
        else:
            format_string = '%s:\t%.' + str(args.acc) + 'f\t%.' + str(args.acc) + 'f\t%s'
        print(format_string % (par_name.strip(), l, u, normtest_str))
