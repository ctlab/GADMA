#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

import argparse
import os, sys
import signal
import numpy as np
from gadma import options
from gadma import support
import gadma
import pandas as pd
import imp

def worker_init():
    """Graceful way to interrupt all processes by Ctrl+C."""
    # ignore the SIGINT in sub process
    def sig_int(signal_num, frame):
        pass

    signal.signal(signal.SIGINT, sig_int)
import copy

def run_one_job(params):
    fs_filename, fs, p0, model_func, lower_bound, upper_bound, pts, opt_name, output = params
    np.random.seed()
    fs_filename = os.path.basename(os.path.normpath(fs_filename))
    prefix = opt_name + '_' + fs_filename
    out_file = os.path.join(output, "opt_" + str(prefix) + ".out")
    res_file = os.path.join(output, str(prefix) + "_params.npy")
    if os.path.isfile(res_file):
        return fs_filename, np.load(res_file)
 
    kwargs = {'p0': p0,
        'data': fs,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'verbose': 1,
        'output_file': out_file}


    if pts is not None:
        import dadi as sim_lib
        func_ex = sim_lib.Numerics.make_extrap_log_func(model_func)
        if opt_name != 'powell':
            kwargs['pts'] = pts
    else:
        import moments as sim_lib
        func_ex = model_func

    kwargs['model_func'] = func_ex

    if opt_name == 'powell' and pts is not None:
        import moments as sim_lib
        def my_func(*args, **kwargs):
            kwargs['pts'] = pts
            return func_ex(*args, **kwargs)
        kwargs['model_func'] = my_func
        popt = moments.Inference.optimize_powell(**kwargs)
    else:
        if opt_name == 'powell':
            popt = sim_lib.Inference.optimize_powell(**kwargs)
        elif opt_name == 'log':
            popt = sim_lib.Inference.optimize_log(**kwargs)

    popt = np.array(popt)
    if pts is not None:
        model_fs = func_ex(popt, fs.sample_sizes, pts)
    else:
        model_fs = func_ex(popt, fs.sample_sizes)
    theta = sim_lib.Inference.optimal_sfs_scaling(model_fs, fs)
    popt = np.append(popt, theta)
    np.save(res_file, popt)
    return  fs_filename, popt

def get_params_names(dem_model_filename, func_name):
    with open (dem_model_filename) as f:
        for line in f:
            if line.strip().startswith('def'):
                local_func_name = line.strip().split()[1].split('(')[0]
                if local_func_name == func_name:
                    str_after_bracket = line.strip().split('(', 1)[1].strip()
                    print str_after_bracket
                    if str_after_bracket.startswith('('):
                        return str_after_bracket[1:].split(')')[0].split(',')

def load_parameters_from_python_file(filename, as_module=False):
    if not as_module:
        f = support.check_file_existence(filename)
        try:
            module = imp.load_source('module', f)
        except Exceprint, e:
            support.error('File ' + filename + " is not valid python file.", error_instance=e)
    else:
        module = filename
    try:
        lower_bound = module.lower_bound
    except:
        lower_bound = None

    try:
        upper_bound = module.upper_bound
    except:
        upper_bound = None

    try:
        p0 = module.popt
    except:
        try:
            p0 = module.p0
        except:
            p0 = None
    try:
        pts = module.pts
    except:
        pts = None
    try:
        par_labels = module.par_labels
    except:
        try:
            par_labels = module.param_labels
        except:
            par_labels = None
    return lower_bound, upper_bound, p0, par_labels, pts 


def main():
    parser = argparse.ArgumentParser('GADMA module for runs of local search on bootstrapped data. Is needed for calculating confidence intervals.')
    parser.add_argument(
        '-b', '--boots', metavar="<dir>", required=True, help='Directory where bootstrapped data is located.')
    parser.add_argument(
        '-d', '--dem_model', metavar="<filename>", required=True, help='File with demographic model. Should contain `model_func` or `generated_model` function. One can put there several extra parameters and they will be taken automatically, otherwise one will need to enter them manually. Such parameters are:\n\t1) p0 (or popt) - initial parameters values\n\t2) lower_bound - list of lower bounds for parameters values\n\t4) upper_bound - list of upper bounds for parameters values\n\t5) par_labels/param_labels - list of string names for parameters 6) pts - pts for dadi (if there is no pts then moments will be run automatically).')
    parser.add_argument(
        '-o', '--output', metavar="<dir>", required=True, help='Output directory.')
    parser.add_argument(
        '-j', '--jobs', metavar="N", type=int, default=1, help='Number of threads for parallel run.')

    parser.add_argument(
        '--opt', metavar="log/powell", type=str, default='log', help='Local search algorithm, by now it can be:\n\t1) `log` - Inference.optimize_log\n\t2) `powell` - Inference.optimize_powell.')
    parser.add_argument(
        '-p', '--params', metavar="<filename>", type=str, default=None, help='Filename with parameters, should be valid python file. Parameters are presented in -d/--dem_model option description upper.')

    args = parser.parse_args()

    output = support.ensure_dir_existence(args.output, False)
    all_boot = gadma.Inference.load_bootstrap_data_from_dir(args.boots, return_filenames=True)
    
    print(str(len(all_boot)) + ' bootstrapped data found.')

    #extract model_func, we cannot put it to function as multiprocessing need pickleable functions
    args.dem_model = support.check_file_existence(args.dem_model)
    try:
        file_with_model_func = imp.load_source('module', args.dem_model)
    except Exception, e:
        support.error('File ' + args.dem_model + " is not valid python file.", error_instance=e)
    try:
        model_func = file_with_model_func.model_func
        name = 'model_func'
    except:
        try:
            model_func = file_with_model_func.generated_model
            name = 'generated_model'
        except:
            support.error(
                "File " + dem_model_filename + ' does not contain function named `model_func` or `generated_model`.')
    par_labels = get_params_names(args.dem_model, name)
    lower_bound, upper_bound, p0, new_par_labels, pts = load_parameters_from_python_file(file_with_model_func, as_module=True)
    if new_par_labels is not None:
        par_labels = new_par_labels

    if args.params is not None:
        new_lower_bound, new_upper_bound, new_p0, new_par_labels, new_pts = load_parameters_from_python_file(args.params)
        if new_lower_bound is not None:
            lower_bound = new_lower_bound
        if new_upper_bound is not None:
            upper_bound = new_upper_bound
        if new_p0 is not None:
            p0 = new_p0
        if new_par_labels is not None:
            par_labels = new_par_labels
        if new_pts is not None:
            pts = new_pts
    if p0 is not None:
        print('Found initial parameters values: ' + str(p0))
    if par_labels is not None:
        print('Found parameters labels (+ `Theta` will be at the end): ' + str(par_labels))
    if p0 is None:
        print("Could not detect p0/popt in files. Please enter:")
        p0 = support.check_comma_sep_list(raw_input(), is_int=False, is_float=True)
    if lower_bound is None:
        print("Could not detect lower_bound in files. Please enter:")
        lower_bound = support.check_comma_sep_list(raw_input(), is_int=False, is_float=True)
    if upper_bound is None:
        print("Could not detect upper_bound in files. Please enter:")
        upper_bound = support.check_comma_sep_list(raw_input(), is_int=False, is_float=True)

    if pts is None:
        print("Cannot detect `pts` parameter. Moments will be run.")
    
    if len(lower_bound) != len(upper_bound):
        support.error("Lengths of upper and lower bound must be the same.")
    
    if par_labels is not None and len(par_labels) != len(lower_bound):
        support.error("Lengths of `par_labels` and lower/upper_bound must be the same.")

    if len(p0) != len(lower_bound):
        support.error("Lengths of initial values (p0) and lower/upper_bound must be the same.")

    for l, p, u in zip(lower_bound, p0, upper_bound):
        if l > p:
            support.error("Lower bound is greater than p0.")
        if p > u:
            support.error("Upper bound is less than p0.")
    if par_labels is None or len(lower_bound) != len(par_labels):
        par_labels = ['par_' + str(i) for i in xrange(len(lower_bound))]
    
    par_labels.append('Theta')
    print('Parameters labels will be: ' + ','.join(par_labels))

    import multiprocessing as mp

    #run_one_job((all_boot[0][0],all_boot[0][1], p0, model_func, lower_bound, upper_bound, pts, args.opt, output))
    pool = mp.Pool(processes=args.jobs, initializer=worker_init)
    try:    
        result = []
        map_result = pool.map_async(run_one_job, [(fs_filename, fs, p0, model_func, lower_bound, upper_bound, pts, args.opt, output) for fs_filename, fs in all_boot], callback=result.extend)
        while True:
            try:
                map_result.get(timeout=1)
                break
            except mp.TimeoutError as ex:
                pass
            except Exception, e:    
                pool.terminate()
                support.error(str(e))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        sys.exit(1)
    
    fs_filenames = [x[0] for x in result]
    result = np.array([x[1] for x in result])
    df = pd.DataFrame(data=result, index=fs_filenames, columns=par_labels)
    csv_path = os.path.join(output, 'result_table.csv')
    pkl_path = os.path.join(output, 'result_table.pkl')
    df.to_csv(csv_path)
    df.to_pickle(pkl_path)
    print df
    print('DONE. Results are saved as csv file (' + csv_path + ') and as pandas dataframe (' + pkl_path + ').')
