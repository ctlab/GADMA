from ..cli import arg_parser
from ..utils import parallel_wrap, StdAndFileLogger, bcolors

from .draw_and_generate_code import print_runs_summary
from .core_run import CoreRun
from .shared_dict import SharedDictForCoreRun

from functools import partial
import numpy as np
import os
import sys

from datetime import datetime
import operator
import multiprocessing
import signal
from multiprocessing import Manager, Pool
import math
import warnings
import time
import traceback


SUPPORT_STRING = "\nIn case of any questions or problems, "\
                 "please contact: ekaterina.e.noskova@gmail.com\n"


def job(index, shared_dict, settings):
    try:
        obj = CoreRun(index, shared_dict, settings)
        obj.run(settings.get_optimizers_init_kwargs())
    except Exception as e:
        print(f"{bcolors.FAIL}Run {index} failed due to following exception:"
              f"{bcolors.ENDC}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise e


def main():
    settings_storage, args = arg_parser.get_settings()

    # Form output directory
    log_file = os.path.join(settings_storage.output_directory, 'GADMA.log')
    params_file = os.path.join(settings_storage.output_directory,
                               'params_file')
    extra_params_file = os.path.join(settings_storage.output_directory,
                                     'extra_params_file')

    # Change output stream both to stdout and log file
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    sys.stdout = StdAndFileLogger(log_file)
    sys.stderr = StdAndFileLogger(log_file)

    print(f"{bcolors.OKGREEN}--Successful arguments parsing--{bcolors.ENDC}")

    # Data reading
    print("\nData reading")
    data = settings_storage.read_data()
    print(f"Number of populations: {settings_storage.number_of_populations}")
    print(f"Projections: {settings_storage.projections}")
    print(f"Population labels: {settings_storage.population_labels}")
    print(f"Outgroup: {settings_storage.outgroup}")
    print(f"{bcolors.OKGREEN}--Successful data reading--{bcolors.ENDC}")

    if settings_storage.directory_with_bootstrap is not None:
        print("\nBootstrap data reading")
        boot_data = settings_storage.read_bootstrap_data()
        print(f"Number of files found: "
              f"{len(settings_storage.bootstrap_data)}")
        print(f"{bcolors.OKGREEN}--Successful bootstrap data reading--"
              f"{bcolors.ENDC}")

    # Save parameters
    settings_storage.to_files(params_file, extra_params_file)
    if not args.test:
        print(f"\nParameters of launch are saved in output directory: "
              f"{params_file}")
        print(f"All output is saved in output directory: {log_file}")

    print(f"{bcolors.OKBLUE}--Start pipeline--{bcolors.ENDC}")

    # Change output stream both to stdout and log file
    sys.stdout = StdAndFileLogger(log_file, settings_storage.silence)

    # Create shared dictionary
    shared_dict = SharedDictForCoreRun()

    # Start pool of processes
    start_time = datetime.now()

    # For debug
#    shared_dict = SharedDictForCoreRun(multiprocessing=False)
#    job(0, shared_dict, settings_storage)
#    os._exit(0)

    pool = Pool(processes=settings_storage.number_of_processes)

    results = []
    for proc_args in [(i + 1, shared_dict, settings_storage)
                      for i in range(settings_storage.number_of_repeats)]:
        results.append(pool.apply_async(job, proc_args))

    pool.close()

    check_time = time.time()
    time_diff = 60 * settings_storage.time_to_print_summary
    time_bias = 0
    get_time = 10
    while True:
        time.sleep(get_time - time_bias)
        time_bias = 0
        all_finished = True
        for r in results:
            try:
                r.get(0)
            except multiprocessing.TimeoutError as e:
                all_finished = False
            except Exception as e:
                pool.terminate()
                print(f"{bcolors.FAIL}Main run failed due to following "
                      f"exception:{bcolors.ENDC}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                print(SUPPORT_STRING)
                sys.stdout = saved_stdout
                sys.stderr = saved_stderr
                os._exit(1)
        if all_finished:
            break
        if (time.time() - check_time) >= time_diff:
            check_time = time.time()
            print_runs_summary(start_time, shared_dict,
                               settings_storage, None, None, None)
            time_bias = time.time() - check_time
            time_bias %= get_time

    pool.join()
#        # graceful way to interrupt all processes by Ctrl+C
#        min_counter = 0
#        while True:
#            try:
#                multiple_results = pool_map.get(60)
#                    # 60 * settings_storage.time_to_print_summary)
#                break
#            # catch TimeoutError and get again
#            except multiprocessing.TimeoutError as ex:
#                print_runs_summary(start_time, shared_dict,
#                                   settings_storage, None,
#                                   precision, None)
#            except Exception as e:
#                print("Catch exception in main")
#                pool.terminate()
#                pool.close()
#                raise RuntimeError(str(e))
#        pool.close()
    print_runs_summary(start_time, shared_dict, settings_storage, None,
                       None, None)

    print('\n--Finish pipeline--\n')
    if args.test:
        print('--Test passed correctly--')
    if settings_storage.theta0 is None:
        print("\nYou didn't specify theta at the beginning. If you want "
              "change it and rescale parameters, please see tutorial.\n")
#        if (params.resume_dir is not None and
#                (params.initial_structure != params.final_structure).any()):
#            print('\nYou have resumed from another launch. Please, check '
#                  'best AIC model, as information about it was lost.\n')

    print('Thank you for using GADMA!')
    print(SUPPORT_STRING)

    sys.stdout = saved_stdout
    sys.stderr = saved_stderr


if __name__ == '__main__':
    main()
