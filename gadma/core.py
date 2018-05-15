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


def draw_bootstrap(logLL_models, output_dir, log_file):
    """Calculate means, maxs, mins and so on for demographic models, print it
    to file and draw boxplots.

    logLL_models :  best models by log likelihood, which we want to compare.
    output_dir :    directory to save files.
    log_file :      file to write logs.
    """
    def write_func(string): return support.write_log(log_file, string,
                                                     write_to_stdout=not options.options_storage.silence)

    write_func('\n--Start bootstrap comparison of result demographic models--')

    logLL_data = defaultdict(list)
    for _ in xrange(100):
        data_sample = options.options_storage.input_data.sample()
        for i, logLL_model in logLL_models.iteritems():
            logLL_data[i].append(
                -logLL_model.get_fitness_func_value(data_sample=data_sample))
    for i in logLL_data:
        logLL_data[i] = np.array(logLL_data[i])
    logLL_means_of_logLL_data = {}

    if not options.options_storage.test:
        write_func('Write statistics to file')
    with open(os.path.join(output_dir, 'some_statistics'), 'w') as f:
        def print_to_file(name_of_model, parameter, minimum,
                          perc_25, median, mean, perc_75, maximum):
            print(
                name_of_model,
                parameter,
                minimum,
                perc_25,
                median,
                mean,
                perc_75,
                maximum,
                file=f,
                sep='\t')
        print_to_file(
            'Number of model',
            'Best by',
            'Minimum',
            '25th percentile',
            'Median',
            'Mean',
            '75th percentile',
            'Maximum')
        for i in logLL_data:
            logLL_means_of_logLL_data[i] = np.mean(logLL_data[i])
            print_to_file(
                'Model ' + i,
                'logLL',
                np.min(logLL_data[i]),
                np.percentile(logLL_data[i], 25),
                np.median(logLL_data[i]),
                logLL_means_of_logLL_data[i],
                np.percentile(logLL_data[i], 75),
                np.max(logLL_data[i]))

    if not options.options_storage.draw_iter == 0:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({'font.size': 10})
        write_func('Draw boxplots')

        fig = plt.figure()
        plt.title('Boxplots of result demographic models by logLL')

        def draw_boxplots_from_data(data, means, start_pos):
            positions = xrange(start_pos, start_pos + len(data))
            plt.boxplot(data.values(), positions=positions)
            plt.plot(positions, means.values(), 'ro')

        draw_boxplots_from_data(logLL_data, logLL_means_of_logLL_data, 1)
        ticks = ['Model ' + str(i) for i in logLL_data]
        plt.gca().set_xlim(xmin=0.5, xmax=len(ticks) + 1)
        positions = xrange(1, len(logLL_data) + 1)
        plt.xticks(
            positions,
            ticks,
            rotation=25,
            ha='right')

        plt.ylabel('Log likelihood')
        fig.savefig(os.path.join(output_dir, 'boxplots.png'))

    best_logLL_mean = None
    best_logLL_mean_model = None
    for i in logLL_means_of_logLL_data:
        if best_logLL_mean is None or best_logLL_mean <= logLL_means_of_logLL_data[i]:
            best_logLL_mean = logLL_means_of_logLL_data[i]
            best_logLL_mean_model = logLL_models[i]

    write_func('\n--Best model by mean logLL:--')
    write_func('Mean log likelihood:\t' + str(best_logLL_mean))
    write_func('Model:\t' + str(best_logLL_mean_model))

    write_func('\n--Finish bootstrap comparison of result demographic models--')
    if not options.options_storage.test:
        write_func(
            'You can find bootstrap comparison of models in output directory.\n')


def print_best_solution_now(start_time, shared_dict,
                            log_file, precision, bootstrap=False):
    """Prints best demographic model by logLL among all processes.

    start_time :    time when equation was started.
    shared_dict :   dictionary to share information between processes.
    log_file :      file to write logs.
    bootstrap :     bool, if True bootstrap analysis will be made.
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

    if not options.options_storage.draw_iter == 0:
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

    if bootstrap:
        draw_bootstrap({i: all_models_data[i][0] for i in all_models_data},
                       options.options_storage.output_dir, log_file)
    if not options.options_storage.draw_iter == 0 and not options.options_storage.test:
        write_func(
            '\nYou can find its picture and python code in output directory')

    elif not options.options_storage.test:
        write_func('\nYou can find its python code in output directory')


def main():
    try:
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
        from contextlib import closing

        pool = Pool(processes=options.options_storage.processes,
                    initializer=worker_init)

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
                min_counter += 1
                if min_counter < options.options_storage.time_for_bootstrap:
                    print_best_solution_now(
                        start_time, shared_dict, log_file, precision)
                else:
                    min_counter = 0
                    print_best_solution_now(
                        start_time, shared_dict, log_file, precision, bootstrap=True)
            except Exception, e:
                pool.terminate()
                support.error(str(e))
        print_best_solution_now(start_time, shared_dict,
                                log_file, precision, bootstrap=True)
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
