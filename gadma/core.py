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
    number = 'ID'
    try:
        def write_func(string): return support.write_log(log_file, string,
                                                         write_to_stdout=not options.options_storage.silence)

        number, log_file, shared_dict = params_tuple

        write_func('Run genetic algorithm number ' + str(number))

        ga_instance = GA(options.options_storage, prefix=str(number))
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


def print_best_solution_now(start_time, shared_dict,
                            log_file, precision, draw_model=True):
    """Prints best demographic model by logLL among all processes.

    start_time :    time when equation was started.
    shared_dict :   dictionary to share information between processes.
    log_file :      file to write logs.
    draw_model :    plot model best by logll and best by BIC.
    """

    def write_func(string): return support.write_log(log_file, string,
                                                     write_to_stdout=not options.options_storage.silence)

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
    write_func('GA number\tlogLL\t\tBIC\t\tModel')
    for number, res in all_models:
        write_func(('Model %3s' % number) + '\t' +
                   my_str(-res.get_fitness_func_value()) + '\t' +
                   my_str(res.get_bic_score()) + '\t' +
                   str(res))

    all_bic_models = [(i, all_models_data[i][1]) for i in all_models_data]
    all_bic_models = sorted(all_bic_models, key=lambda x: x[1].get_bic_score())

    if (options.options_storage.final_structure != options.options_storage.initial_structure).any():
        write_func('\nAll best BIC models:')
        write_func('GA number\tlogLL\t\tBIC\t\tModel')
        for number, res in all_bic_models:
            write_func(('Model %3s' % number) + '\t' +
                       my_str(-res.get_fitness_func_value()) + '\t' +
                       my_str(res.get_bic_score()) + '\t' +
                       str(res))

    write_func('\n--Best model by log likelihood--')
    write_func('Log likelihood:\t' +
               my_str(-all_models[0][1].get_fitness_func_value()))
    write_func('with BIC score:\t' +
               my_str(all_models[0][1].get_bic_score()))
    write_func('Model:\t' + str(all_models[0][1]))

    if (options.options_storage.final_structure != options.options_storage.initial_structure).any():
        write_func('\n--Best model by BIC score--')
        write_func('Log likelihood:\t' +
                   my_str(-all_bic_models[0][1].get_fitness_func_value()))
        write_func('with BIC score:\t' +
                   my_str(all_bic_models[0][1].get_bic_score()))
        write_func('Model:\t' + str(all_bic_models[0][1]))

    if draw_model:
        all_models[0][1].draw(
            os.path.join(options.options_storage.output_dir,
                         'best_logLL_model.png'),
            title='logLL: ' +
            support.float_representation(-all_models[0][1].get_fitness_func_value(), precision) +
            ', BIC: ' + support.float_representation(all_models[0][1].get_bic_score(), precision))
        if (options.options_storage.final_structure != options.options_storage.initial_structure).any():
            all_bic_models[0][1].draw(
                os.path.join(options.options_storage.output_dir,
                             'best_bic_model.png'),
                title='logLL: ' +
                support.float_representation(-all_bic_models[0][1].get_fitness_func_value(), precision) +
                ', BIC: ' + support.float_representation(all_bic_models[0][1].get_bic_score(), precision))

    all_models[0][1].dadi_code_to_file(
        os.path.join(options.options_storage.output_dir, 'best_logLL_model_dadi_code.py'))
    all_models[0][1].moments_code_to_file(
        os.path.join(options.options_storage.output_dir, 'best_logLL_model_moments_code.py'))
    if (options.options_storage.final_structure != options.options_storage.initial_structure).any():
        all_bic_models[0][1].dadi_code_to_file(
            os.path.join(options.options_storage.output_dir, 'best_bic_model_dadi_code.py'))
        all_bic_models[0][1].moments_code_to_file(
            os.path.join(options.options_storage.output_dir, 'best_bic_model_moments_code.py'))

    if not options.options_storage.draw_iter == 0 and not options.options_storage.test:
        write_func(
            '\nYou can find its picture and python code in output directory')

    elif not options.options_storage.test:
        write_func('\nYou can find its python code in output directory')


def main():
    options.parse_args()

    log_file = os.path.join(
        options.options_storage.output_dir, 'GADMA.log')
    open(log_file, 'w').close()

    support.write_log(log_file, "--Successful arguments' parsing--\n")
    params_filepath = os.path.join(
        options.options_storage.output_dir, 'params')
    options.options_storage.save(options.options_storage.output_dir)
    if not options.options_storage.test:
        support.write_log(
            log_file, 'You can find all parameters of this run in:\t\t' + params_filepath + '\n')
        support.write_log(log_file, 'All output is saved (without warnings and errors) in:\t' +
                          os.path.join(options.options_storage.output_dir, 'GADMA.log\n'))

    support.write_log(log_file, '--Start pipeline--\n')

    # For debug
#        run_genetic_algorithm((1, log_file, None))

    # Create shared dictionary
    m = Manager()
    shared_dict = m.dict()

    # Start pool of processes
    start_time = datetime.now()
    
    pool = Pool(processes=options.options_storage.processes,
                initializer=worker_init)
    try:
        pool_map = pool.map_async(
            run_genetic_algorithm,
            [(i + 1, log_file, shared_dict)
             for i in range(options.options_storage.repeats)])
        pool.close()

        precision = 1 - int(math.log(options.options_storage.epsilon, 10))

        # graceful way to interrupt all processes by Ctrl+C
        min_counter = 0
        while True:
            try:
                multiple_results = pool_map.get(
                    60 * options.options_storage.time_for_print)
                break
            # catch TimeoutError and get again
            except multiprocessing.TimeoutError as ex:
                print_best_solution_now(
                    start_time, shared_dict, log_file, precision)
            except Exception, e:
                pool.terminate()
                support.error(str(e))
        print_best_solution_now(start_time, shared_dict,
                                log_file, precision)
        support.write_log(log_file, '\n--Finish pipeline--\n')
        if options.options_storage.test:
            support.write_log(log_file, '--Test passed correctly--')
        if options.options_storage.theta is None:
            support.write_log(
                log_file, "\nYou didn't specify theta at the beginning. If you want change it and rescale parameters, please see tutorial.\n")
        if options.options_storage.resume_dir is not None and (options.options_storage.initial_structure != options.options_storage.final_structure).any():
            support.write_log(
                log_file, '\nYou have resumed from another launch. Please, check best BIC model, as information about it was lost.\n')

        support.write_log(log_file, 'Thank you for using GADMA!')

    except KeyboardInterrupt:
        pool.terminate()
        support.error('KeyboardInterrupt')


if __name__ == '__main__':
    main()
