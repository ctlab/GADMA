import argparse
import os
import sys
import numpy as np
import gadma
import importlib
import copy
import multiprocessing as mp
from importlib.machinery import SourceFileLoader


def run_job(params):
    """
    Run one local search optimization on one bootstrapped data.

    :param params: Tuple of (filename with spectrum, loaded spectrum,
                    initial best values, settings storage).
    """
    boot_fs_filename, boot_fs, p0, settings = params
    np.random.seed()
    fs_filename = gadma.utils.abspath(boot_fs_filename)
    prefix = str(settings.local_optimizer) + '_' + boot_fs_filename
    out_file = os.path.join(settings.output_directory,
                            "opt_" + str(prefix) + ".out")
    res_file = os.path.join(settings.output_directory,
                            str(prefix) + "_params.npy")
    if os.path.isfile(res_file):
        return boot_fs_filename, np.load(res_file, allow_pickle=True)

    kwargs = {'verbose': 1,
              'report_file': out_file}

    custom_model = settings.get_model()

    grid_size = settings.get_engine_args()

    engine = gadma.get_engine(settings.engine)
    engine.set_data(boot_fs)
    engine.set_model(custom_model)

    f = engine.evaluate

    optimizer = gadma.get_local_optimizer(settings.local_optimizer)
    optimizer.maximize = True

    result = optimizer.optimize(f, custom_model.variables, x0=p0,
                                args=grid_size, **kwargs)
    popt = result.x
    theta = engine.get_theta(popt, *grid_size)

    popt = np.append(popt, theta)
    np.save(res_file, popt)
    return (boot_fs_filename, popt)


def load_module(filename):
    """
    Loads module from file.
    """
    filename = gadma.utils.abspath(filename)
    name = "strange_name_of." +\
           os.path.basename(filename).replace('/', '_').rstrip('.py')
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name,
                                                                  filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parameters_from_python_file(filename):
    """
    Loads ``lower_bound``, ``upper_bound``, ``p0`` or ``popt`` and ``pts`` from
    valid python file if they are there.

    :returns: tuple(p0, settings)
    """
    settings = gadma.SettingsStorage()
    if filename is None:
        return None, settings
>>>>>>> devel
    try:
        module = load_module(filename)
    except Exception as e:
        raise ValueError('File ' + filename + " is not valid python file." +
                         str(e))
    try:
        settings.lower_bound = module.lower_bound
    except Exception as e:
        settings.lower_bound = None

    try:
        settings.upper_bound = module.upper_bound
    except Exception as e:
        settings.upper_bound = None

    try:
        p0 = module.popt
    except Exception as e:
        try:
            p0 = module.p0
        except Exception as e:
            p0 = None
    try:
        settings.pts = module.pts
    except Exception as e:
        settings.pts = None
    try:
        settings.parameter_identifiers = module.par_labels
    except Exception as e:
        try:
            settings.parameter_identifiers = module.param_labels
        except Exception as e:
            settings.parameter_identifiers = None
    return p0, settings


def main():
    """
    Base function of script parse command-line arguments, loads all
    bootstrapped data and runs local search optimization for each data in
    parallel.

    All output is saved in output directory.
    """
    import pandas as pd
    parser = argparse.ArgumentParser(
        "GADMA module for runs of local search on bootstrapped data. "
        "Is needed for calculating confidence intervals.\n")
    parser.add_argument('-b', '--boots', metavar="<dir>", required=True,
                        help='Directory where bootstrapped data is located.')
    parser.add_argument('-d', '--dem_model', metavar="<filename>",
                        required=True,
                        help="File with demographic model. Should contain "
                             "`model_func` or `generated_model` function. One "
                             "can put there several extra parameters and they "
                             "will be taken automatically, otherwise one will "
                             "need to enter them manually. Such parameters "
                             "are:\n"
                             "\t1) p0 (or popt) - initial parameters values\n"
                             "\t2) lower_bound - list of lower bounds for "
                             "parameters values\n"
                             "\t3) upper_bound - list of pper bounds for "
                             "parameters values\n"
                             "\t4) par_labels/param_labels - list of string "
                             "names for parameters 6) pts - pts for dadi "
                             "(if there is no pts then moments will be run "
                             "automatically).")
    parser.add_argument('-o', '--output', metavar="<dir>", required=True,
                        help='Output directory.')
    parser.add_argument('-j', '--jobs', metavar="N", type=int, default=1,
                        help='Number of threads for parallel run.')

    parser.add_argument('--opt', metavar="log/powell", type=str, default='log',
                        help="Local search algorithm, by now it can be:\n"
                             "\t1) `log` - Inference.optimize_log\n"
                             "\t2) `powell` - Inference.optimize_powell.")
    parser.add_argument('-p', '--params', metavar="<filename>", type=str,
                        default=None,
                        help="Filename with parameters, should be valid "
                             "python file. Parameters are presented in "
                             "-d/--dem_model option description upper.")

    args = parser.parse_args()

<<<<<<< HEAD
    output = support.ensure_dir_existence(args.output, False)
    all_boot = gadma.Inference.load_bootstrap_data_from_dir(args.boots, return_filenames=True)
    
    print(str(len(all_boot)) + ' bootstrapped data found.')

    #extract model_func, we cannot put it to function as multiprocessing need pickleable functions
    args.dem_model = support.check_file_existence(args.dem_model)
    try:
        file_with_model_func = imp.load_source('module', args.dem_model)
    except Exception as e:
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
=======
    p0, settings = load_parameters_from_python_file(args.dem_model)
    settings.output_directory = args.output
    if args.opt == 'log':
        settings.local_optimizer = 'optimize_log'
    elif args.opt == 'powell':
        settings.local_optimizer = 'optimize_log_powell'
    else:
        settings.local_optimizer = args.opt
>>>>>>> devel

    loaded_attrs = ['parameter_identifiers', 'pts',
                    'lower_bound', 'upper_bound']
    if args.params is not None:
        fresh_p0, fresh_settings = load_parameters_from_python_file(
            args.params)

        for attr_name in loaded_attrs:
            if getattr(fresh_settings, attr_name) is not None:
                settings.__setattr__(attr_name,
                                     getattr(fresh_settings, attr_name))
        if fresh_p0 is not None:
            p0 = fresh_p0

    settings.custom_filename = args.dem_model
    settings.get_model()

    for attr_name in loaded_attrs:
        if attr_name != 'pts' and getattr(settings, attr_name) is None:
            raise ValueError(f"Parameter `{attr_name}` is missed both in "
                             f"demographic model and params files")
    if p0 is None:
<<<<<<< HEAD
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
        par_labels = ['par_' + str(i) for i in range(len(lower_bound))]
    
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
            except Exception as e:    
                pool.terminate()
                support.error(str(e))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        sys.exit(1)
    
=======
        raise ValueError(f"Parameter `p0` (or `popt`) is missed both in "
                         "demographic model and params files")

    if settings.pts is None:
        settings.engine = 'moments'
    else:
        settings.engine = 'dadi'
    print(f"Chosen engine: {settings.engine}")

    output = gadma.utils.ensure_dir_existence(args.output)

    settings.directory_with_bootstrap = args.boots
    all_boots = list(settings.read_bootstrap_data(True))

    p_ids = list(settings.parameter_identifiers)
    p_ids.append('Theta')

    print(f"{len(all_boots)} bootstrapped data found.")
    print(f"Found initial parameters values: {p0}")
    print(f"Parameters labels will be: {', '.join(p_ids)}")
    print(f"Lower bound will be: {settings.lower_bound}")
    print(f"Upper bound will be: {settings.upper_bound}")
    print(f"Optimization to run: {settings.local_optimizer}")

    if args.jobs == 1:
        result = []
        for fs_filename, fs in all_boots:
            res = run_job((all_boots[0][0], all_boots[0][1], p0, settings))
            result.append(res)
    else:
        pool = mp.Pool(processes=args.jobs)
        try:
            result = []
            map_result = pool.map_async(run_job,
                                        [(fs_filename, fs, p0, settings)
                                         for fs_filename, fs in all_boots],
                                        callback=result.extend)
            while True:
                try:
                    map_result.get(timeout=1)
                    break
                except mp.TimeoutError as ex:
                    pass
                except Exception as e:
                    pool.terminate()
                    raise e
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            sys.exit(1)

    fs_filenames = [x[0] for x in result]
    result = np.array([x[1] for x in result])
    df = pd.DataFrame(data=result, index=fs_filenames, columns=p_ids)
    csv_path = os.path.join(output, 'result_table.csv')
    pkl_path = os.path.join(output, 'result_table.pkl')
    df.to_csv(csv_path)
    df.to_pickle(pkl_path)
    print(df)
    print(f"DONE. Results are saved as csv file ({csv_path}) and "
          f"as pandas dataframe ({pkl_path}).")
