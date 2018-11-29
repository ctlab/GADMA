#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################
from __future__ import print_function
import numpy as np
import os

from gadma import options
from gadma import support
from gadma.genetic_algorithm import GA
from datetime import datetime

import operator
from collections import defaultdict
import multiprocessing
import signal
from multiprocessing import Manager, Pool
import math


def run_genetic_algorithm(params_tuple):
    """Main function to run genetic algorithm with prefix number.

    params_tuple :    tuple of 3 objects:
        number :    number of GA to run (moreover its prefix).
        log_file :  log file to write output.
        shared_dict :   dictionary to share information between processes.
    """
    np.random.seed()
    number = 'ID'
    try:
        def write_func(string): return support.write_log(log_file, string,
                                                         write_to_stdout=not params.silence)

        number, params, log_file, shared_dict = params_tuple

        write_func('Run genetic algorithm number ' + str(number))

        ga_instance = GA(params, prefix=str(number))
        ga_instance.run(shared_dict=shared_dict)

        write_func('Finish genetic algorithm number ' + str(number))
        return number, ga_instance.best_model()
    except Exception, e:
        raise RuntimeError('GA number ' + str(number) + ': ' + str(e))


def worker_init():
    """Graceful way to interrupt all processes by Ctrl+C."""
    # ignore the SIGINT in sub process
    def sig_int(signal_num, frame):
        pass

    signal.signal(signal.SIGINT, sig_int)


def print_best_solution_now(start_time, shared_dict, params,
                            log_file, precision, draw_model):
    """Prints best demographic model by logLL among all processes.

    start_time :    time when equation was started.
    shared_dict :   dictionary to share information between processes.
    log_file :      file to write logs.
    draw_model :    plot model best by logll and best by AIC.
    """

    def write_func(string): return support.write_log(log_file, string,
                                                     write_to_stdout=not params.silence)

    def my_str(x): return support.float_representation(x, precision)

    all_models_data = dict(shared_dict)
    if not all_models_data:
        return
    all_models = [(i, all_models_data[i][0]) for i in all_models_data]
    all_models = sorted(all_models, key=lambda x: x[
                        1].get_fitness_func_value())

    s = (datetime.now() - start_time).total_seconds()
    write_func('\n[%(hours)03d:%(minutes)02d:%(seconds)02d]' % {
        'hours': s / 3600,
        'minutes': s % 3600 / 60,
        'seconds': s % 60
    })

    write_func('All best logLL models:')
    write_func('GA number\tlogLL\t\tAIC\t\tModel')
    for number, res in all_models:
        write_func(('Model %3s' % number) + '\t' +
                   my_str(-res.get_fitness_func_value()) + '\t' +
                   my_str(res.get_aic_score()) + '\t' +
                   str(res))

    if params.initial_structure is not None and (params.final_structure != params.initial_structure).any():
        all_aic_models = [(i, all_models_data[i][1]) for i in all_models_data]
        all_aic_models = sorted(all_aic_models, key=lambda x: x[1].get_aic_score())
        write_func('\nAll best AIC models:')
        write_func('GA number\tlogLL\t\tAIC\t\tModel')
        for number, res in all_aic_models:
            write_func(('Model %3s' % number) + '\t' +
                       my_str(-res.get_fitness_func_value()) + '\t' +
                       my_str(res.get_aic_score()) + '\t' +
                       str(res))

    write_func('\n--Best model by log likelihood--')
    write_func('Log likelihood:\t' +
               my_str(-all_models[0][1].get_fitness_func_value()))
    write_func('with AIC score:\t' +
               my_str(all_models[0][1].get_aic_score()))
    write_func('Model:\t' + str(all_models[0][1]))

    if params.model_func_file is None and (params.final_structure != params.initial_structure).any():
        write_func('\n--Best model by AIC score--')
        write_func('Log likelihood:\t' +
                   my_str(-all_aic_models[0][1].get_fitness_func_value()))
        write_func('with AIC score:\t' +
                   my_str(all_aic_models[0][1].get_aic_score()))
        write_func('Model:\t' + str(all_aic_models[0][1]))

    if draw_model:
        all_models[0][1].draw(
            os.path.join(params.output_dir,
                         'best_logLL_model.png'),
            title='logLL: ' +
            support.float_representation(-all_models[0][1].get_fitness_func_value(), precision) +
            ', AIC: ' + support.float_representation(all_models[0][1].get_aic_score(), precision))
        if params.model_func_file is None and (params.final_structure != params.initial_structure).any():
            all_aic_models[0][1].draw(
                os.path.join(params.output_dir,
                             'best_aic_model.png'),
                title='logLL: ' +
                support.float_representation(-all_aic_models[0][1].get_fitness_func_value(), precision) +
                ', AIC: ' + support.float_representation(all_aic_models[0][1].get_aic_score(), precision))

    if not params.initial_structure is None or not params.moments_scenario:
        all_models[0][1].dadi_code_to_file(
            os.path.join(params.output_dir, 'best_logLL_model_dadi_code.py'))
    if not params.initial_structure is None or params.moments_scenario:
        all_models[0][1].moments_code_to_file(
            os.path.join(params.output_dir, 'best_logLL_model_moments_code.py'))
    if params.model_func_file is None and (params.final_structure != params.initial_structure).any():
        if not params.initial_structure is None or not params.moments_scenario:
            all_aic_models[0][1].dadi_code_to_file(
                os.path.join(params.output_dir, 'best_aic_model_dadi_code.py'))
        if not params.initial_structure is None or params.moments_scenario:
            all_aic_models[0][1].moments_code_to_file(
                os.path.join(params.output_dir, 'best_aic_model_moments_code.py'))

    if not params.test:
        write_func(
            '\nYou can find its picture and python code in output directory')


def main():
    params = options.parse_args()

    log_file = os.path.join(
        params.output_dir, 'GADMA.log')
    open(log_file, 'w').close()

    support.write_log(log_file, "--Successful arguments' parsing--\n")
    params_filepath = os.path.join(
        params.output_dir, 'params')
    params.save(params.output_dir)
    if not params.test:
        support.write_log(
            log_file, 'You can find all parameters of this run in:\t\t' + params_filepath + '\n')
        support.write_log(log_file, 'All output is saved (without warnings and errors) in:\t' +
                          os.path.join(params.output_dir, 'GADMA.log\n'))

    support.write_log(log_file, '--Start pipeline--\n')

    # For debug
#    run_genetic_algorithm((1, params, log_file, None))

    # Create shared dictionary
    m = Manager()
    shared_dict = m.dict()

    # Start pool of processes
    start_time = datetime.now()
    
    pool = Pool(processes=params.processes,
                initializer=worker_init)
    try:
        pool_map = pool.map_async(
            run_genetic_algorithm,
            [(i + 1, params, log_file, shared_dict)
             for i in range(params.repeats)])
        pool.close()

        precision = 1 - int(math.log(params.epsilon, 10))

        # graceful way to interrupt all processes by Ctrl+C
        min_counter = 0
        while True:
            try:
                multiple_results = pool_map.get(
                    60 * params.time_for_print)
                break
            # catch TimeoutError and get again
            except multiprocessing.TimeoutError as ex:
                print_best_solution_now(
                    start_time, shared_dict, params, log_file, 
                    precision, draw_model=params.matplotlib_available)
            except Exception, e:
                pool.terminate()
                support.error(str(e))
        print_best_solution_now(start_time, shared_dict, params,log_file, 
                precision, draw_model=params.matplotlib_available)
        support.write_log(log_file, '\n--Finish pipeline--\n')
        if params.test:
            support.write_log(log_file, '--Test passed correctly--')
        if params.theta is None:
            support.write_log(
                log_file, "\nYou didn't specify theta at the beginning. If you want change it and rescale parameters, please see tutorial.\n")
        if params.resume_dir is not None and (params.initial_structure != params.final_structure).any():
            support.write_log(
                log_file, '\nYou have resumed from another launch. Please, check best AIC model, as information about it was lost.\n')

        support.write_log(log_file, 'Thank you for using GADMA!')

    except KeyboardInterrupt:
        pool.terminate()
        support.error('KeyboardInterrupt')


if __name__ == '__main__':
    main()
