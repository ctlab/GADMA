#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

import argparse
import sys
import support
import os
import numpy as np
import re
from operator import itemgetter
import ast
from gadma.version import __version__
import imp 
import pkgutil
import gadma



# default options
START_MODEL_NUMBER_OF_PERIODS = 1
FINAL_MODEL_NUMBER_OF_PERIODS = 2


class Options_storage:
    '''
    Class to handle great amount of options for
    genetic algorithm and Demographic model.
    '''

    def __init__(self):
        # All parameters with default values

        # Main options. Output and input.
        self.output_dir = None
        self.input_file = None
        self.input_data = None
        self.number_of_populations = None
        self.pop_labels = None
        self.ns = None
        self.linked_snp = True
        self.boot_dir = None
        self.boots = None

        # Pipeline
        self.theta = None
        self.gen_time = None
        self.multinom = None
        self.only_sudden = False
        self.dadi_pts = None
        self.moments_scenario = True
        self.relative_params = False
        self.no_mig = False

        #Custom model
        self.model_func_file = None
        self.model_func = None
        self.lower_bound = None
        self.upper_bound = None
        self.p_ids = None

        # Structure of models
        self.initial_structure = None
        self.final_structure = None

        # Time bounds
        self.split_1_lim = None
        self.split_2_lim = None

        # GA options
        self.size_of_generation = 10
        self.fracs = "0.2, 0.3, 0.3"
        self.frac_of_old_models = None
        self.frac_of_mutated_models = None
        self.frac_of_crossed_models = None

        self.mutation_strength = 0.2
        self.const_for_mut_strength = 1.01

        self.mutation_rate = 0.2
        self.const_for_mut_rate = 1.02

        self.stop_iter = 100
        self.epsilon = 1e-2

        # Local search
        self.optimize_name = 'optimize_powell'

        # Printing and drawing
        self.code_iter = 0
        self.silence = False
        self.draw_iter = 0
        self.gen_time_units = 1.0

        self.repeats = 1
        self.processes = 1
        self.test = False
        self.resume_dir = None
        self.only_models = None

        # Extra parameters

        # Bounds on models parameters. They are relative to N_A (!)
        self.min_N = 0.01
        self.max_N = 100.0
        self.min_T = 0.0
        self.max_T = 5.0
        self.min_M = 0.0
        self.max_M = 10.0

        # Parameters for local search alg
        self.ls_verbose = None
        self.ls_flush_delay = 0.5
        self.ls_epsilon = 1e-3
        self.ls_gtol = 1e-05
        self.ls_maxiter = None
        # for hill climbing
        self.hc_mutation_rate = None
        self.hc_const_for_mutation_rate = None
        self.hc_stop_iter = None

        # Options of mutation, crossing and random generating
        self.random_N_A = True
        self.multinom_cross = False
        self.multinom_mutate = False

        # Options of printing summary information about repeats
        self.time_for_print = 1  # min

        # Options of distributions
        self.distribution = 'normal'  # can be 'uniform'
        self.std = None  # std for normal dist

        # Some options about drawing plots:
        self.matplotlib_available = False
        self.pil_available = False
        self.moments_available = False

    def from_file(self, input_filename):
        with open(input_filename) as f:
            line_number = 0
            for line in f:
                line = line.strip()
                pos = line.find('#')
                if pos != -1:
                    line = line[:pos]
                if line == '':
                    continue

                if len(line.split(':')) > 2:
                    support.warning(
                        "In parameters file in line number " +
                        str(line_number) +
                        " two ':'. May be error? Everything after second ':' was ignored.")

                identity = line.split(':')[0].strip().lower()
                value = line.split(':')[1].strip()
                if identity == 'output directory':
                    self.output_dir = value
                elif identity == 'resume from':
                    self.resume_dir = value if value.lower() != 'none' else None
                elif identity == 'only models':
                    if value.lower() == 'true':
                        self.only_models = True
                    elif  value.lower() == 'false':
                        self.only_models = False
                    else:
                        self.only_models = None
                elif identity == 'input file':
                    self.input_file = value
                elif identity == 'population labels':
                    self.pop_labels = value if value.lower() != 'none' else None
                elif identity == 'projections':
                    self.ns = value if value.lower() != 'none' else None
                elif identity == 'theta0':
                    self.theta = float(
                        value) if value.lower() != 'none' else None
                elif identity == 'time for generation':
                    self.gen_time = float(
                        value) if value.lower() != 'none' else None
                elif identity == 'multinom':
                    self.multinom = True if value.lower() == 'true' else False
                elif identity == 'initial structure':
                    self.initial_structure = value
                elif identity == 'final structure':
                    self.final_structure = value
                elif identity == 'relative parameters':
                    self.relative_params = value.lower() == 'true'
                elif identity == 'no migrations':
                    self.no_mig = value.lower() == 'true'
                elif identity == 'size of population in ga':
                    self.size_of_generation = int(value)
                elif identity == 'fractions in ga':
                    self.fracs = value
                elif identity == 'mean mutation strength':
                    self.mutation_strength = float(value)
                elif identity == 'mean mutation rate':
                    self.mutation_rate = float(value)
                elif identity == 'const for mutation rate':
                    self.const_for_mut_rate = float(value)
                elif identity == 'const for mutation strength':
                    self.const_for_mut_strength = float(value)
                elif identity == 'epsilon':
                    self.epsilon = float(value)
                elif identity == 'stop iteration':
                    self.stop_iter = int(value)
                elif identity == 'pts':
                    self.dadi_pts = value if value.lower() != 'none' else None
                elif identity == 'use moments or dadi':
                    if value == 'moments':
                        self.moments_scenario = True
                    else:
                        self.moments_scenario = False
                elif identity == 'draw models every n iteration':
                    self.draw_iter = int(value)
                elif identity == "print models' code every n iteration":
                    self.code_iter = int(value)
                elif identity == 'units of time in drawing':
                    if value.lower() == 'years':
                        self.gen_time_units = 1
                    elif value.lower() == 'kya' or value.lower() == 'thousand years':
                        self.gen_time_units = 1000
                    else:
                        support.warning(
                        'Cannot understand units of time in line ' +\
                                str(line_number) + ' in parameters file. Years were taken.')
                elif identity == 'silence':
                    self.silence = value.lower() == 'true'
                elif identity == 'number of repeats':
                    self.repeats = int(value)
                elif identity == 'number of processes':
                    self.processes = int(value)
                elif identity == 'upper bound of first split':
                    self.split_1_lim = float(
                        value) if value.lower() != 'none' else None
                elif identity == 'upper bound of second split':
                    self.split_2_lim = float(
                        value) if value.lower() != 'none' else None
                elif identity == 'name of local optimization':
                    self.optimize_name = value if value.lower() != 'none' else None
                    names = [
                        'optimize',
                        'optimize_log',
                        'optimize_powell',
                        'optimize_lbfgsb',
                        'optimize_log_lbfgsb',
                        'optimize_log_fmin',
                        'hill_climbing']
                    if value not in names:
                        support.error(
                            "Can't parse name of local search. Acceptable names are: " +
                            ', '.join(names))
                    else:
                        pass
                # now extra args
                elif identity == 'min_n':
                    self.min_N = float(value)
                elif identity == 'max_n':
                    self.max_N = float(value)
                elif identity == 'min_t':
                    self.min_T = float(value)
                elif identity == 'max_t':
                    self.max_T = float(value)
                elif identity == 'min_m':
                    self.min_M = float(value)
                elif identity == 'max_m':
                    self.max_M = float(value)
                elif identity == 'verbose':
                    self.ls_verbose = None if value.lower() == 'none' else int(value)
                elif identity == 'flush delay':
                    self.ls_flush_delay = float(value)
                elif identity == 'epsilon for ls':
                    self.ls_epsilon = float(value)
                elif identity == 'gtol':
                    self.ls_gtol = float(value)
                elif identity == 'maxiter':
                    self.ls_maxiter = None if value.lower() == 'none' else int(value)
                elif identity == 'mean mutation rate for hc':
                    self.hc_mutation_rate = None if value.lower() == 'none' else float(value)
                elif identity == 'const for mutation rate for hc':
                    self.hc_const_for_mutation_rate = None if value.lower() == 'none' else float(value)
                elif identity == 'stop iteration for hc':
                    self.hc_stop_iter = None if value.lower() == 'none' else float(value)
                elif identity == 'multinomial mutation':
                    self.multinom_mutation = value.lower() == 'true'
                elif identity == 'multinomial crossing':
                    self.multinom_mutation = value.lower() == 'true'
                elif identity == 'random n_a':
                    self.random_N_A = value.lower == 'true'
                elif identity == 'time to print summary':
                    self.time_for_print = float(value)
                elif identity == 'distribution':
                    self.distribution = value.lower()
                elif identity == 'std':
                    self.std = None if value.lower() == 'none' else float(value)
                elif identity == 'only sudden':
                    self.only_sudden = value.lower() == 'true'
                elif identity == 'custom filename':
                    self.model_func_file = value if value.lower() != 'none' else None
                elif identity == 'lower bounds':
                    self.lower_bound = value if value.lower() != 'none' else None
                elif identity == 'upper bounds':
                    self.upper_bound = value if value.lower() != 'none' else None
                elif identity == 'parameter identifiers':
                    self.p_ids = value if value.lower() != 'none' else None
                elif identity == "linked snp's" or identity == "linked snp":
                    self.linked_snp = value.lower() == 'true'
                elif identity == "unlinked snp's" or identity == "unlinked snp":
                    self.linked_snp = value.lower() == 'false'
                elif identity == 'directory with bootstrap' or identity == 'directory of bootstrap':
                    self.boot_dir = value if value.lower() != 'none' else None
                else:
                    support.error(
                        'Cannot recognize identifier: ' +
                        str(line.split(':')[0].strip()))

                line_number += 1

    def to_file(self, output_filename):
        def comma_sep_repr(l):
            if l is None:
                return 'None'
            if len(l) == 1:
                return str(l[0])
            return ', '.join([str(x) for x in l])

        home_dir = support.get_home_dir()
        with open(os.path.join(home_dir, "params_template")) as t:
            string = t.readlines()
        string = ''.join(string)

        with open(output_filename, 'w') as out:
            out.write(
                string.format(
                    self.output_dir,
                    self.resume_dir,
                    self.only_models,
                    self.input_file,
                    comma_sep_repr(self.pop_labels),
                    comma_sep_repr(self.ns),
                    str(self.theta),
                    str(self.gen_time),
                    'moments' if self.moments_scenario else 'dadi',
                    str(self.multinom),
                    comma_sep_repr(self.dadi_pts),
                    comma_sep_repr(self.initial_structure),
                    comma_sep_repr(self.final_structure),
                    str(self.split_1_lim),
                    str(self.split_2_lim),
                    str(self.relative_params),
                    str(self.no_mig),
                    str(self.size_of_generation),
                    comma_sep_repr(self.fracs),
                    str(self.mutation_strength),
                    str(self.const_for_mut_strength),
                    str(self.mutation_rate),
                    str(self.const_for_mut_rate),
                    str(self.epsilon),
                    str(self.stop_iter),
                    str(self.optimize_name),
                    str(self.draw_iter),
                    str(self.code_iter),
                    'Thousand Years' if self.gen_time_units == 1000 else 'Years',
                    str(self.silence),
                    str(self.repeats),
                    str(self.processes),
                    str(self.only_sudden),
                    str(self.model_func_file),
                    comma_sep_repr(self.lower_bound),
                    comma_sep_repr(self.upper_bound),
                    comma_sep_repr(self.p_ids),
                    str(self.linked_snp),
                    str(self.boot_dir)
                ))

    def to_file_extra(self, output_filename):
        home_dir = support.get_home_dir()
        with open(os.path.join(home_dir, 'extra_params_template')) as t:
            string = t.readlines()
        string = ''.join(string)

        with open(output_filename, 'w') as out:
            out.write(
                string.format(
                    str(self.min_N),
                    str(self.max_N),
                    str(self.min_T),
                    str(self.max_T),
                    str(self.min_M),
                    str(self.max_M),
                    str(self.ls_verbose),
                    str(self.ls_flush_delay),
                    str(self.ls_epsilon),
                    str(self.ls_gtol),
                    str(self.ls_maxiter),
                    str(self.hc_mutation_rate),
                    str(self.hc_const_for_mutation_rate),
                    str(self.hc_stop_iter),
                    str(self.random_N_A),
                    str(self.multinom_cross),
                    str(self.multinom_mutate),
                    str(self.time_for_print),
                    str(self.distribution),
                    str(self.std)
                ))

    def save(self, out_dir):
        self.to_file(os.path.join(out_dir, 'params'))
        self.to_file_extra(os.path.join(out_dir, 'extra_params'))

    def restore(self, resume_dir):
        self.from_file(os.path.join(resume_dir, 'params'))
        self.from_file(os.path.join(resume_dir, 'extra_params'))

    def put_default_structures(self):
        '''
        If some parameters aren't defined, put default values.
        '''
        # if not custom model then fill structures
        if self.initial_structure is not None:
            self.initial_structure = support.check_comma_sep_list(
                self.initial_structure)
        if self.final_structure is not None:
            self.final_structure = support.check_comma_sep_list(
                self.final_structure)

        if self.model_func_file is None:
            if self.initial_structure is None:
                self.initial_structure = [
                    START_MODEL_NUMBER_OF_PERIODS
                    for _ in xrange(self.number_of_populations)]
            if self.final_structure is None:
                self.final_structure = np.array(self.initial_structure)
        if self.initial_structure is not None:
            self.initial_structure = np.array(
                self.initial_structure)
        if self.final_structure is not None:
            self.final_structure = np.array(
                self.final_structure)
        else:
            if self.lower_bound is None:
                self.lower_bound = []
                for p_id in self.p_ids:
                    if p_id == 'n':
                        self.lower_bound.append(self.min_N)
                    elif p_id == 't':
                        self.lower_bound.append(self.min_T)
                    elif p_id == 'm':
                        self.lower_bound.append(self.min_M)
                    elif p_id == 's':
                        self.lower_bound.append(0.0 + self.min_N)
            else:
                self.lower_bound = [float(x) for x in support.check_comma_sep_list(self.lower_bound, is_int=False)]

            if self.upper_bound is None:
                self.upper_bound = []
                for p_id in self.p_ids:
                    if p_id == 'n':
                        self.upper_bound.append(self.max_N)
                    elif p_id == 't':
                        self.upper_bound.append(self.max_T)
                    elif p_id == 'm':
                        self.upper_bound.append(self.max_M)
                    elif p_id == 's':
                        self.upper_bound.append(1.0 - self.min_N)
            else:
                self.upper_bound = [float(x) for x in support.check_comma_sep_list(self.upper_bound, is_int=False)]


    def check(self):
        '''
        Check correctness of parameters. Unless throws error.
        '''
        if self.multinom is None:
            if self.model_func_file is None:
                self.multinom = False
            else:
                self.multinom = True

        if self.pop_labels is not None:
            self.pop_labels = [x.strip() for x in self.pop_labels.split(',')]
        if self.ns is not None:
            self.ns = support.check_comma_sep_list(self.ns)

        self.input_file = support.check_file_existence(self.input_file)
        
        if self.resume_dir is not None:
            self.resume_dir = support.check_dir_existence(self.resume_dir)
        if self.resume_dir is not None and self.output_dir is None:
            self.output_dir = support.ensure_dir_existence(
                self.resume_dir + "_resumed", check_emptiness=True)
        elif self.output_dir is None:
            support.error("Parameter `Output directory` is required")
        else:
            self.output_dir = support.ensure_dir_existence(
                self.output_dir, check_emptiness=True)

        if self.input_file is None:
            support.error(
                "Parameter `Input file` is required")
        if self.theta is None:
            support.warning(
                "`Theta0` is not specified. It would be 1.0.")
        if self.gen_time is None:
            support.warning(
                "`Time for one generation` is not specified. Time will be in genetic units.")

        self.input_data, self.ns, self.pop_labels = support.load_spectrum(
                self.input_file, self.ns, self.pop_labels)
        self.ns = np.array(self.ns)
        self.number_of_populations = len(self.ns)

        # Linked or unlinked data
        if not self.linked_snp and self.boot_dir is not None:
            support.warning(
                    "SNP's are marked as unlinked, so the directory with bootstrap will be ignored.")
        elif self.linked_snp:
            if self.boot_dir is not None:
                self.boot_dir = support.check_dir_existence(self.boot_dir)
                self.boots = gadma.Inference.load_bootstrap_data_from_dir(self.boot_dir, self.ns, self.pop_labels)

        # Custom model
        if self.model_func_file is not None:
            self.model_func_file = support.check_file_existence(self.model_func_file)
            file_with_model_func = imp.load_source('module', self.model_func_file)
            try:
                self.model_func = file_with_model_func.model_func  
            except:
                support.error(
                    "File " + self.model_func_file + ' does not contain function named `model_func`.')

        
        if self.model_func_file is not None:
            if self.p_ids is not None:
                self.p_ids = support.check_comma_sep_list(self.p_ids, is_int=False)
                
        self.fracs = [float(x) for x in self.fracs.split(",")]
        if len(self.fracs) != 3:
            support.error(
                "length of `Fractions` (Parameters of genetic algorithm) must be 3")
        self.frac_of_old_models = self.fracs[0]
        self.frac_of_mutated_models = self.fracs[1]
        self.frac_of_crossed_models = self.fracs[2]

        if self.moments_scenario and self.dadi_pts is not None:
            support.warning(
                "Moments doesn't use --pts argument, so it would be ignored")
        if self.dadi_pts is None:
            max_n = max(self.ns)
            self.dadi_pts = [max_n, max_n + 10, max_n + 20]
        else:
            self.dadi_pts = support.check_comma_sep_list(self.dadi_pts)

        self.put_default_structures()

        self.final_check()

    def final_check(self):
        if self.model_func_file is not None and self.model_func_file is None:
            if self.p_ids is None and (self.lower_bound is None or self.upper_bound is None):
                support.error(
                        "Either parameter identifiers or lower and upper bounds should be specified.")

        if self.model_func_file is not None and self.initial_structure is not None:
            support.warning(
                    "Both structure and custom model are specified. Custom model will be optimized, structure will be ignored.")
        if self.model_func_file is not None and self.only_sudden:
            support.warning(
                    "Both custom model and `Only sudden: True` are specified. `Only sudden` will be ignored.")

        if (self.frac_of_old_models +
                self.frac_of_crossed_models +
                self.frac_of_mutated_models) > 1:
            support.error(
                "Sum of Fractions (Parameters of genetic algorithm) must be less than or equal to 1")
        if (self.frac_of_old_models +
                self.frac_of_crossed_models +
                self.frac_of_mutated_models) == 1:
            support.warning("Faction of random models is 0")


        # check lengths of bounds and p_ids
        if self.model_func_file is not None:
            if len(self.lower_bound) != len(self.upper_bound):
                support.error(
                        "Lengths of lower and upper bounds should be equal.")
            if self.p_ids is not None:
                if len(self.p_ids) != len(self.lower_bound):
                    print self.p_ids
                    print self.lower_bound
                    support.error(
                        "Lengths of lower, upper bounds and parameters identificators should be equal.")


        if self.initial_structure is not None:
            if len(self.initial_structure
                   ) != self.number_of_populations:
                support.error("wrong length of initial model structure: must be " +
                              str(self.number_of_populations))
            for n in self.initial_structure:
                if n < 0:
                    support.error('elements in comma-separated list ' + ','.join(
                        str(x) for x in self.initial_structure) +
                        ' must be positive (`Initial structure` parameter)')
        if self.final_structure is not None:
            if len(self.final_structure
                   ) != self.number_of_populations:
                support.error("Wrong length of final model structure: must be " +
                              str(self.number_of_populations))
            for n in self.final_structure:
                if n < 0:
                    support.error('Elements in comma-separated list ' + ','.join(
                        self.final_structure) +
                        ' must be positive (`Final structure` parameter)')
            if not (self.final_structure >=
                    self.initial_structure).all():
                support.error(
                    "Final structure must be greater than initial structure")
        if self.split_1_lim is not None and self.split_2_lim is not None and not self.split_1_lim > self.split_2_lim:
            support.error(
                "Upper bound of first split must be greater than upper bound of second split")
        if self.size_of_generation <= 0:
            support.error(
                "Size of population (Parameters of genetic algorithm) must be positive"
            )
        if self.mutation_strength > 1 or self.mutation_strength < 0:
            support.error(
                "Mutation strength (Paramters of genetic algorithm) must be between 0 and 1"
            )
        if self.mutation_rate > 1 or self.mutation_rate < 0:
            support.error(
                "Mutation rate (Parameters of genetic algorithm) must be between 0 and 1"
            )
        if self.const_for_mut_rate < 1 or self.const_for_mut_rate > 2:
            support.error(
                "Const for adaptive mutation rate (Parameters of genetic algorithm) must be between 1 and 2"
            )
        if self.const_for_mut_strength < 1 or self.const_for_mut_rate > 2:
            support.error(
                "Const for adaptive mutation strength (Parameters of genetic algorithm) must be between 1 and 2"
            )
        if self.dadi_pts is not None:
            for n in self.dadi_pts:
                if n < 0:
                    support.error('elements in comma-separated list ' +
                                  ','.join(str(x) for x in self.dadi_pts) +
                                  ' must be positive (Pts parameter)')
        if self.repeats <= 0:
            support.error("Repeats (Parameters of pipeline) must be positive")
        if self.processes <= 0:
            support.error(
                "Processes (Parameters of pipeline) must be positive")

        if self.number_of_populations < 3 and self.split_2_lim is not None:
            support.warning("There is no second split in case of " +
                            str(self.number_of_populations) +
                            " populations. Upper bound for it will be ignored.")
            self.split_2_lim = None
        if self.number_of_populations < 2 and self.split_1_lim is not None:
            support.warning(
                "There is no first split in case of 1 populations. Upper bound for it will be ignored.")
            self.split_1_lim = None

        if self.moments_scenario:
            if pkgutil.find_loader('moments') is None:
                if self.model_func_file is not None:
                    support.error("moments is not installed. You tried to use custom model and moments.")
                if pkgutil.find_loader('dadi') is not None:
                    options_storage.moments_scenario = False
                    support.warning("moments is not installed, dadi with " + str(self.dadi_pts) +"grid size will be used instead.")
                else:
                    support.error("None of the dadi or the moments are installed.")
        else:
            if pkgutil.find_loader('dadi') is None:
                if self.model_func_file is not None:
                    support.error("dadi is not installed. You tried to use custom model and moments.")
                if pkgutil.find_loader('moments') is not None:
                    options_storage.moments_scenario = True
                    support.warning("dadi is not installed, moments will be used instead.")
                else:
                    support.error("None of the dadi or the moments are installed.")

        packages = []
        self.matplotlib_available = pkgutil.find_loader('matplotlib') is not None
        if not self.matplotlib_available:
            packages.append('matplotlib')
        
        # If custom model and dadi is used we can ignore PIL absence
        if self.model_func_file is None or self.moments_scenario:
            self.pil_available = pkgutil.find_loader('PIL') is not None
            if not self.pil_available:
                packages.append('Pillow')
        
        self.moments_available = pkgutil.find_loader('moments') is not None
        if not self.moments_available:
            packages.append('moments')
            
        if not self.matplotlib_available:
            support.warning(
                "To draw models and SFS plots you should install: " +
                ', '.join(packages))
        elif not self.pil_available and self.moments_available:
            support.warning(
                "To draw concatenated plots you should install: Pillow")
        elif not self.moments_available:
            support.warning(
                "To draw models plots you should install: " +
                ', '.join(packages))

        if self.optimize_name == 'optimize_powell' and not self.moments_scenario:
            if not self.moments_available:
                support.warning(
                    "To use Powell optimization one need moments installed. BFGS (optimize_log) will be used instead.")
                self.optimize_name = 'optimize_log'


        if self.distribution != 'normal' and self.distribution != 'uniform':
            support.error(
                "Distribution in extra parameters must be `normal` or `uniform`.")
        if self.distribution == 'uniform' and self.std is not None:
            support.warning(
                'Std in extra parameters will be ignored as uniform distribution was chosen.')

# options
options_storage = Options_storage()


def version():
    '''
    Returns string with current version.
    '''
    return "GADMA version " + str(
        __version__
    ) + "\tby Ekaterina Noskova (ekaterina.e.noskova@gmail.com)" + "\n"


def usage():
    '''
    Returns usage of tool.
    '''
    return version() + "" \
        "Usage: \n\tgadma -p/--params <params_file> -e/--extra <extra_params_file>\n"\
        "\n\n"\
        "Instead/With -p/--params and -e/--extra option you can set:\n"\
        "\t-o/--output <output_dir>\toutput directory.\n"\
        "\t-i/--input <in.fs>/<in.txt>\tinput file with AFS or in dadi format.\n"\
        "\t--resume <resume_dir>\t\tresume another launch from <resume_dir>.\n"\
        "\t--only_models\t\t\tflag to take models only from another launch (--resume option).\n"\
        "\n\n"\
        "\t-h/--help\t\tshow this help message and exit.\n"\
        "\t-v/--version\t\tshow version and exit.\n"\
        "\t--test\t\t\trun test case.\n" + support.SUPPORT_STRING


class ArgParser(argparse.ArgumentParser):

    def format_help(self):
        return usage()

    def error(self, message):
        support.error(message)


def test_args():
    '''
    Put default args for test case.
    '''
    global options_storage

    import tempfile
    options_storage.output_dir = tempfile.mkdtemp("gadma_test_dir")
    options_storage.output_dir = support.ensure_dir_existence(
        options_storage.output_dir)

    options_storage.input_file = os.path.join(
        support.get_home_dir(), "..", "fs_examples", "test.fs")
    options_storage.input_data, options_storage.ns, options_storage.pop_labels = support.load_spectrum(
            options_storage.input_file, None, None)
    options_storage.ns = np.array(options_storage.ns)
    options_storage.number_of_populations = 1
    options_storage.linked_snp = False
    options_storage.theta = 0.37976
    options_storage.gen_time = 25
    options_storage.initial_structure = np.array([1])
    options_storage.final_structure = np.array([2])
    options_storage.size_of_generation = 5
    options_storage.fracs = [float(x)
                             for x in options_storage.fracs.split(",")]
    options_storage.frac_of_old_models = options_storage.fracs[0]
    options_storage.frac_of_mutated_models = options_storage.fracs[1]
    options_storage.frac_of_crossed_models = options_storage.fracs[2]
    options_storage.optimize_name = 'hill_climbing'
    options_storage.moments_scenario = True

    options_storage.relative_params = False
    options_storage.dadi_pts = [20, 30, 40]
    options_storage.repeats = 2
    options_storage.processes = 2
    options_storage.epsilon = 1
    options_storage.test = True
    options_storage.multinom = True

    options_storage.final_check()

    return options_storage


def parse_args():
    '''
    Parse args from command line and store them in options_storage.
    '''

    global options_storage

    usage = str(sys.argv[0]) + " -p <params_file>"
    parser = ArgParser(add_help=False)

    parser.add_argument(
        '-p',
        '--params',
        metavar="<params_file>",
        required=False,
        default=None)
    parser.add_argument(
        '-e',
        '--extra',
        metavar="<extra_params_file>",
        required=False,
        default=None)

    parser.add_argument(
        '--resume', metavar="<dir>", required=False, default=None)
    parser.add_argument(
        '--only_models', action='store_true')

    parser.add_argument(
        '-o', '--output', metavar="<dir>", required=False, default=None)
    parser.add_argument(
        '-i',
        '--input',
        metavar="<filename.fs>/<filename.txt>",
        required=False,
        default=None)

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=version(),
        default=argparse.SUPPRESS)
    parser.add_argument(
        '--test', action='store_true')
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.test:
        print("--Running test case--\n")
        return test_args()
        
    if args.only_models and args.resume is None:
        support.error("Option --only_models  must be used with --resume option.")

    if args.output is not None:
        options_storage.output_dir = args.output
    if args.input is not None:
        options_storage.input_file = args.input
    if args.resume is not None:
        options_storage.resume_dir = args.resume
    if args.only_models:
        options_storage.only_models = True

    if args.params is not None:
        options_storage.from_file(args.params)
    elif args.output is None and args.input is None and args.resume is None:
        support.error(
            "-p/--params or --resume or -o/output and -i/--input options are requered.")

    if args.extra is not None:
        options_storage.from_file(args.extra)

    if args.output is not None and options_storage.output_dir != args.output:
        support.error(
            "Output directory in parameters file doesn't match to one from -o/--output option")
    if args.input is not None and options_storage.input_file != args.input:
        support.error(
            "Input file in parameters file doesn't match to one from -i/--input option")

    if options_storage.resume_dir is not None:
        resume_dir = options_storage.resume_dir
        options_storage = Options_storage()
        options_storage.restore(resume_dir)
        options_storage.output_dir = None
        if args.params is not None:
            options_storage.from_file(args.params)
        if args.extra is not None:
            options_storage.from_file(args.extra)
        if args.output is not None:
            options_storage.output_dir = args.output
        if args.input is not None:
            options_storage.input_file = args.input
        if args.resume is not None:
            options_storage.resume_dir = args.resume
        if args.only_models:
            options_storage.only_models = True

    if args.resume is not None and os.path.abspath(
        os.path.expanduser(
            options_storage.resume_dir)) != os.path.abspath(
            os.path.expanduser(
                args.resume)):
        support.error(
            "Resume directory in parameters file doesn't match to one from --resume option")

    options_storage.check()

    return options_storage
