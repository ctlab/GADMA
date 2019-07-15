import numpy as np
import pandas as pd
import sys
import argparse
from gadma import support

parser = argparse.ArgumentParser()
parser.add_argument('input_filename', metavar='<filename>', help='Filename (.csv or .pkl) with result from local search runs on bootstraped data. Output of run_ls_on_boot_data.py.')
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

for m,s, par_name in zip(means, stds, df.columns):
	l = m - 1.96 * s
	u = m + 1.96 * s
	if args.log:
		l = np.exp(l)
		u = np.exp(u)
	if args.tex:
		format_string = '%s:\t$[%.' + str(args.acc) + 'f - %.' + str(args.acc) + 'f]$'
	else:
		format_string = '%s:\t%.' + str(args.acc) + 'f\t%.' + str(args.acc) + 'f'
	print(format_string % (par_name.strip(), l, u))
