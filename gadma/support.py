#!/usr/bin/env python3

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

from __future__ import print_function
import sys
import os
import shutil
from os.path import abspath, expanduser
import numpy as np
from threading import Thread
import functools
import importlib
import math
import traceback

ERROR_PREFIX = "Error:"
WARNING_PREFIX = "Warning:"
SUPPORT_STRING = "\nIn case of any questions or problems, please contact: ekaterina.e.noskova@gmail.com\n"


def error(error_str, prefix=ERROR_PREFIX, error_instance=None,exit=True):
    if error_instance is not None:
        sys.stderr.write(traceback.format_exc())
    sys.stderr.write("\n\033[1;31m" + prefix + "\033[0;0m " + error_str + "\n")
    sys.stderr.write(SUPPORT_STRING)
    sys.stderr.flush()
    if exit:
        os._exit(1)


def warning(warning_str, prefix=WARNING_PREFIX):
    sys.stdout.write("\n\033[1;35m" + prefix +
                     "\033[0;0m " + warning_str + "\n\n")
    sys.stdout.flush()


def write_to_file(filename=None, *args):
    if filename is None:
        print(*args, sep='\t')
        return
    with open(filename, 'a') as f:
        print(*args, file=f, sep='\t')


def write_log(log_file, string, write_to_stdout=True):
    if write_to_stdout:
        print(string)
    write_to_file(log_file, string)


def check_file_existence(input_filename):
    filename = abspath(expanduser(input_filename))
    if not os.path.isfile(filename):
        return error("file " + filename + " doesn't exist")
    return filename


def check_dir_existence(input_dirname):
    dirname = abspath(expanduser(input_dirname))
    if not os.path.isdir(dirname):
        return error("directory " + input_dirname + " doesn't exist")
    return dirname


def check_comma_sep_list(l_str, is_int=True, is_float=False):
    try:
        splited_list = l_str.split(',')
        if splited_list[0].startswith('['):
            splited_list[0] = splited_list[0][1:]
        if splited_list[-1].endswith(']'):
            splited_list[-1] = splited_list[-1][:1]
        for i in range(len(splited_list)):
            if splited_list[i].startswith("'") and splited_list[i].endswith("'"):
                splited_list[i] = splited_list[i][1:-1]
        if is_int:
            return np.array([int(x) for x in splited_list])
        elif is_float:
            return np.array([float(x) for x in splited_list])
        else:
            return np.array([x.strip().lower() for x in splited_list])
    except Exception as e:
        if l_str.strip().lower() == 'none':
            return None
        if is_int:
            er_str = "can't read comma-separated list of ints: "
        if is_float:
            er_str = "can't read comma-separated list of floats: "
        else:
            er_str = "can't read comma-separated list: "
        er_str += str(l_str) + ' (' + str(e) + ')' 
        error(er_str)



def ask_user(question):
    return raw_input(question + " [Y/n]: ").lower().strip()[0] == "y"


def ensure_dir_existence(dirname, check_emptiness=False):
    dirname = abspath(expanduser(dirname))
    if os.path.isfile(dirname):
        error(dirname + " is not a directory, but file.")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if os.listdir(dirname) != [] and check_emptiness:
        error(
            "directory " +
            dirname +
            " is not empty\nYou can write:  rm -rf " +
            dirname +
            "\t to remove directory")
    return dirname


def clear_dir(dirname):
    for the_file in os.listdir(dirname):
        file_path = os.path.join(dirname, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            error("Can't clean directory " + dirname + ": " + e)


def get_home_dir():
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def float_representation(value, precision, min_precision=2):
    return ("%10." + str(max(min_precision, precision)) + "f") % value


def sample_from_truncated_normal(mean, std=None, from_=0.0, to_=1.0):
    from scipy.stats import truncnorm
    myclip_a = from_
    myclip_b = to_
    my_mean = mean
    my_std = mean if std is None else std

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    return truncnorm.rvs(a, b, loc=my_mean, scale=my_std)


def get_dadi_or_moments():
    try:
        return importlib.import_module('dadi')
    except ImportError:
        try:
            return importlib.import_module('moments')
        except:
            error("None of the dadi or the moments are installed")

# read input file
def read_fs_file(filename, proj, pop_labels):
    dadi_or_moments = get_dadi_or_moments()
    data = dadi_or_moments.Spectrum.from_file(filename)
    ns = np.array(data.shape) - 1
    if data.pop_ids is None:
        if pop_labels is not None:
            warning(
                "Spectrum file %s is in old format - without population labels,"\
                        " so they will be taken from corresponding parameter: " % filename +
                ', '.join(pop_labels) +
                ".")
            data.pop_ids = pop_labels
        else:
            data.pop_ids = ['pop ' + str(i) for i in range(len(data.shape))]

    if proj is not None and not list(ns) == list(proj):
        try:
            data = data.project(proj)
        except Exception as e:
            error("Wrong Projections parameter: " + str(e))

    if pop_labels is not None and data.pop_ids != pop_labels:
        d = {x: i for i, x in enumerate(data.pop_ids)}
        try:
            d = [d[x] for x in pop_labels]
        except:
            error(
                "Wrong Population labels parameter, list of population labels is: " +
                ', '.join(
                    data.pop_ids))
        data = np.transpose(data, d)
        data.pop_ids = pop_labels
    return data, np.array(data.shape) - 1, data.pop_ids


def read_dadi_file(filename, proj, pop_labels):
    dadi_or_moments = get_dadi_or_moments()
    try:
        dd = dadi_or_moments.Misc.make_data_dict(filename)
    except Exception as e:
        error("Wrong input file: " + str(e))
    polarized = True
    with open(filename) as f:
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        splited_line = line.split()
        if (len(splited_line) - 6) % 2 != 0:
            error(
                "Cannot calculate number of populations in dadi's input file. Maybe it's wrong?"
            )
        number_of_pops = (len(splited_line) - 6) / 2
        if pop_labels is None:
            if not splited_line[3].isdigit():
                pop_ids = splited_line[3:3 + number_of_pops]
                splited_line = f.readline().split()
            else:
                pop_ids = ["Pop_" + str(x) for x in range(number_of_pops)]
        else:
            pop_ids = pop_labels
        if proj is None:
            proj = [
                int(splited_line[3 + x]) +
                int(splited_line[4 + number_of_pops + x])
                for x in range(number_of_pops)
            ]
        for line in f:
            splited_line = line.split()
            if splited_line[1][1].lower() not in ['a', 't', 'c', 'g']:
                polarized = False
            if proj is None:
                for x in range(number_of_pops):
                    sum_of_ind = int(
                        splited_line[3 + x]) + int(splited_line[4 + number_of_pops + x])
                    if sum_of_ind > proj[x]:
                        proj[x] = sum_of_ind
    data = dadi_or_moments.Spectrum.from_data_dict(
        dd, pop_ids=pop_ids, projections=proj, polarized=polarized)
    return data, proj, pop_ids


READ_ALLOWED_EXTENSIONS = {
    '.txt': read_dadi_file,
    '.fs': read_fs_file,
    '.sfs': read_fs_file
}


def load_spectrum(filename, proj, pop_labels):
    ext = "." + os.path.splitext(filename)[1][1:]
    global READ_ALLOWED_EXTENSIONS
    if ext not in READ_ALLOWED_EXTENSIONS.keys():
        error(
            "File " +
            filename +
            " doesn't end with one of {}".format(READ_ALLOWED_EXTENSIONS.keys()))
    return READ_ALLOWED_EXTENSIONS[ext](filename, proj, pop_labels)


# for interrupting models drawing
class TimeoutError(Exception):
    pass


# for interrupting models drawing
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [
                TimeoutError('function [%s] timeout [%s seconds] exceeded!' %
                             (func.__name__, timeout))
            ]

            def new_func():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=new_func)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                error('Error starting thread to timeout model drawing')
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


# Very specific help functions for main part and genetic algorithm

def get_ll_precision(params):
    return 1 - int(math.log(params.epsilon, 10))

def print_set_of_models(log_file, set_of_enum_models, params, first_col='N', heading='', silence=True):
    '''
    log_file : log file to write.
    set_of_enum_models : set of enumerated models - (num, model).
    params : params of run.
    first_col : name of first column.
    heading : heading before print.
    silence : if True then stdout is empty.
    '''
    ll_precision = get_ll_precision(params)
    write_log(log_file, heading, write_to_stdout=not silence)
    has_claic = True
    for i, model in set_of_enum_models:
        if model.claic_score is None:
            has_claic = False
    if params.linked_snp:
        if has_claic:
            write_log(log_file, first_col + '\tlogLL\t\t\tCLAIC\t\t\t\tmodel', write_to_stdout=not silence)
        else:
            write_log(log_file, first_col + '\tlogLL\t\t\t\tmodel', write_to_stdout=not silence)
    else:
        write_log(log_file, first_col + '\tlogLL\t\t\tAIC\t\t\t\tmodel', write_to_stdout=not silence)
    
    for i, model in set_of_enum_models:
        ll_str = float_representation(
                -model.get_fitness_func_value(),
                ll_precision)
        aic_str = float_representation(
                model.get_aic_score(),
                ll_precision)
        if params.linked_snp:
            if has_claic:
                claic_str = 'None' if (model.claic_score is None) else (float_representation(
                        model.claic_score,
                        ll_precision) + ' (eps=%.1e)' % model.claic_eps)
                write_log(log_file, '\t'.join([('Model %3s' % i), ll_str, claic_str, str(model)]), write_to_stdout=not silence)
            else:
                write_log(log_file, '\t'.join([('Model %3s' % i), ll_str, str(model)]), write_to_stdout=not silence)
        else:
            write_log(log_file, '\t'.join([('Model %3s' % i), ll_str, aic_str, str(model)]), write_to_stdout=not silence)

def print_one_model_long(log_file, model, params, heading='', silence=True):
    '''
    log_file : log file to write.
    model : model that should be printed.
    params : params of run.
    heading : heading before print.
    silence : if True then stdout is empty.
    '''
    ll_precision = get_ll_precision(params)
    write_log(log_file, heading, write_to_stdout=not silence)
    write_log(log_file,
                          'Log likelihood:\t' + float_representation(-model.get_fitness_func_value(),
                                                       ll_precision), write_to_stdout=not silence)
    if not params.linked_snp:
        write_log(
            log_file,
            'with AIC score:\t\t' + float_representation(
                model.get_aic_score(),
                ll_precision), write_to_stdout=not silence)
    elif model.claic_score is not None:
        claic, eps = model.get_claic_score(get_eps=True)
        write_log(
            log_file,
            'with CLAIC score:\t\t' + float_representation(
                claic, ll_precision) + ' (eps=%.1e)' % eps, write_to_stdout=not silence)

    write_log(log_file, 'Model:\t' + str(model), write_to_stdout=not silence)

def print_best_logll_model_long(log_file, model, params, silence=True):
    print_one_model_long(log_file, model, params, heading='\n--Best model by log likelihood--', silence=silence)

def print_best_aic_model_long(log_file, model, params, silence=True):
    print_one_model_long(log_file, model, params, heading='\n--Best model by AIC--', silence=silence)

def print_one_model_short(log_file, model, params, heading='', first_col='', silence=True):
    ll_precision = get_ll_precision(params)
    write_log(log_file, heading, write_to_stdout=not silence)
    write_log(
        log_file, '\t'.join([str(first_col),
        float_representation(
            -model.get_fitness_func_value(),
            ll_precision), str(model)]), write_to_stdout=not silence)

def print_final_model(log_file, model, params, silence=True):
    print_one_model_short(log_file, model, params, heading='BEST:', silence=silence)

def print_model_code(out_dir, model, params, prefix, suffix='', sub_folders=False):
    if not params.model_func is not None or not params.moments_scenario:
        if sub_folders:
            dadi_out_dir = os.path.join(out_dir, 'dadi')
        else:
            dadi_out_dir = out_dir
        model.dadi_code_to_file(
            os.path.join(dadi_out_dir, prefix + '_dadi_code' + suffix + '.py'))
    if not params.model_func is not None or params.moments_scenario:
        if sub_folders:
            moments_out_dir = os.path.join(out_dir, 'moments')
        else:
            moments_out_dir = out_dir

        model.moments_code_to_file(
            os.path.join(moments_out_dir, prefix + '_moments_code' + suffix + '.py'))

def save_model_plot(out_file, model, params, title):
    if params.matplotlib_available:
        ll_precision = get_ll_precision(params)
        plot_title = title + ', logLL: '\
                + float_representation(-model.get_fitness_func_value(), ll_precision)
        if not params.linked_snp:
            plot_title += ', AIC: '\
                    + float_representation(model.get_aic_score(), ll_precision)
             
        model.draw(out_file, plot_title)

