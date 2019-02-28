#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

import time
import copy
import numpy as np
import os
import sys
import math

from gadma.demographic_model import Demographic_model
from gadma import options
from gadma import support


class GA(object):
    max_mutation_rate = 1.0

    def __init__(
            self,
            params,
            prefix=None,
            one_initial_model=None):
        """Genetic algorithm class.

        params :    an object with parameters to work with
            parameters for genetic algorithm:
            size_of_generation :    size of the generation of demographic models.
            frac_of_old_models :    the fraction of models from the previous
                                    population in a new one.
            frac_of_mutated_models :    the fraction of mutated models in a new
                                        population.
            frac_of_crossed_models :    the fraction of crossed models in a
                                        new population.
            mutation_rate : the rate to change one parameter of model in order
                            to get mutatetd model from it
            epsilon :   constant to stop genetic algorithm, its "presition"
            out_dir :   output directory
            final_structure :   structure of final model
        (Other fields in params are for Demographic model class)
        prefix :    prefix for output folder.
        """
        # all parameters
        self.params = params

        # some constants on number of iterations:
        # number of iterations, during which fitness function doesn't change, when we stop ga
        # look for is_stoped function
        self.it_without_changes_to_stop_ga = self.params.stop_iter

        self.prefix = prefix
        self.is_custom_model = self.params.initial_structure is None
        self.one_initial_model = one_initial_model
        self.ll_precision = 1 - int(math.log(self.params.epsilon, 10))

        # our adaptive mutation_rate
        self.cur_mutation_rate = self.params.mutation_rate
        self.cur_mutation_strength = self.params.mutation_strength

        # helpful parameters
        self.size_of_generation = self.params.size_of_generation
        self.number_of_old_models = int(
            params.size_of_generation * params.frac_of_old_models)
        self.number_of_mutated_models = int(
            params.size_of_generation * params.frac_of_mutated_models)
        self.number_of_crossed_models = int(params.size_of_generation * params.
                                            frac_of_crossed_models)
        self.number_of_random_models = params.size_of_generation - \
            self.number_of_old_models - self.number_of_mutated_models - \
            self.number_of_crossed_models

        # basic parameters
        self.cur_iteration = 0
        self.first_iteration = 0  # is not 0 if we restore ga
        self.work_time = 0
        self.models = []
        self.final_models = []

        # options that can be False after restore
        self.run_before_ls = True
        self.run_ls = True

        # variables for stops
        self.without_changes = 0

        # connected files and directories
        self.out_dir = None
        self.log_file = None
        self.best_model_by_aic = None

    def restore(self):
        def restore_from_cur_pop_of_models(list_of_str):
            for restore_str in list_of_str:
                self.models.append(
                    Demographic_model(
                        self.params,
                        restore_string=restore_str.strip().split('\t')[3]))
            if not (self.models[0].get_structure() <=
                    self.params.final_structure).all():
                raise RuntimeError(
                    'New final structure is less than current structure.')

        def restore_iteration_state(iter_out, size):
            if not self.params.only_models:
                self.cur_iteration = int(iter_out[0].split('#')[-1][:-1])
                self.first_iteration = self.cur_iteration

            start_ind = 3
            end_ind = 3 + size
            restore_from_cur_pop_of_models(iter_out[start_ind: end_ind])
            
            if self.params.only_models:
                return

            if not iter_out[end_ind].startswith('Current mean mutation rate:'):
                support.warning(
                    "Expect 'Current mean mutation rate:' after current population during restoring.")
            else:
                self.cur_mutation_rate = float(iter_out[end_ind].split(':')[-1])

            if not iter_out[
                    end_ind +
                    1].startswith('Current mean number of params to change:'):
                support.warning(
                    "Expect 'Current mean number of params to change:' after current population during restoring.")
            else:
                self.cur_mutation_strength = float(
                    iter_out[end_ind + 1].split(':')[-1]) / self.models[0].get_number_of_params()

        def restore_from_ls_string(ls_string, is_best=False):
            if is_best:
                index = 1
            else:
                index = 2
            self.models[0] = Demographic_model(
                self.params, restore_string=ls_string.strip().split('\t')[index])

        def read_values_properly():
            dadi_code_file = os.path.join(self.params.resume_dir, self.prefix, 'current_best_logLL_model_dadi_code.py')
            moments_code_file = os.path.join(self.params.resume_dir, self.prefix, 'current_best_logLL_model_moments_code.py')
            par_values = None
            for code_file in [dadi_code_file, moments_code_file]:
                if os.path.isfile(code_file):
                    with open(code_file) as f:
                        key_start_1 = '#current best params = '
                        key_start_2 = 'popt = '
                        for key in [key_start_1, key_start_2]:
                            for line in f:
                                if line.startswith(key):
                                    par_values = [float(x) for x in line.strip()[len(key) + 1: -1].split(',')]
                                    break
                    break
            if par_values is not None:
                support.write_log(
                        self.log_file, 
                        'GA number ' +
                        self.prefix +
                        ' find good file to restore')
                self.models[0].construct_from_vector(par_values)


        if not os.path.isfile(
            os.path.join(
                self.params.resume_dir,
                self.prefix,
                'GADMA_GA.log')):
            return
        support.write_log(
            self.log_file,
            'GA number ' +
            self.prefix +
            ' find dir to restore')
        iter_out = []
        prev_iter_out = []
        self.models = []
        with open(os.path.join(self.params.resume_dir, self.prefix, 'GADMA_GA.log')) as log_file:
            for line in log_file:
                if line.startswith('Iteration #'):
                    prev_iter_out = iter_out
                    iter_out = []
                iter_out.append(line.strip())

        if len(iter_out) == 0 or not iter_out[0].startswith('Iteration #'):
            support.write_log(
                self.log_file,
                'GA number ' +
                self.prefix +
                ' find empty dir to restore. It would be run from random models.')
            return

        pos_of_empty_str = 0
        for line in iter_out:
            if line == '':
                break
            pos_of_empty_str += 1
        
        # if there is no empty line then we need prev iteration
        if len(iter_out) == pos_of_empty_str:
            iter_out = prev_iter_out
            pos_of_empty_str = 0
            for line in iter_out:
                if line == '':
                    break
                pos_of_empty_str += 1
        size = pos_of_empty_str - 5
        restore_iteration_state(iter_out, size)

        pos_of_last_empty_str = len(iter_out)
        for line in reversed(iter_out):
                if line != '':
                    break
                pos_of_last_empty_str -= 1
                
        # try to find file with all parameters
        read_values_properly()
        
        if pos_of_last_empty_str - 11 > size:
            if iter_out[-1].startswith(
                    'BEST') and iter_out[-2].startswith('Try to improve'):
                # when we have not print final result
                self.run_before_ls = False
                self.run_ls = False
                self.select(size)
                return
            if iter_out[-1].startswith('Try to improve'):
                # when we have not print final result too
                self.run_before_ls = False
                self.select(size)
                return
            if iter_out[-1].startswith('BEST'):
                # remove string with BEST
                iter_out.pop()
            # if hill climbing there can be extra strings (in reverse order):
            if iter_out[-1].startswith(
                    'Current mean number of params to change:'):
                # remove string
                iter_out.pop()
            if iter_out[-1].startswith('Current mean mutation rate:'):
                # remove string
                iter_out.pop()

            # now we want to restore model from last string
            self.run_before_ls = False
            restore_from_ls_string(
                iter_out[-1], iter_out[-2].startswith('BEST'))
            if iter_out[-2].startswith('BEST'):
                self.run_ls = False
        read_values_properly()
        self.select(size)

    def init_first_population_of_models(self):
        """Get the first population of models to run genetic algorithm.

        If self.params.restore_dir is not None, then population will be
        restored from previous run.
        """
        # we need to restore if we resume
        if self.prefix is not None and self.params.resume_dir is not None:
            self.restore()

        if len(self.models) > 0:
            for i in xrange(self.params.size_of_generation - len(self.models)):
                self.models.append(self.get_random_model())
            if self.params.size_of_generation > len(self.models):
                self.select(self.params.size_of_generation)
            return
        # generate random models 5 times more than the population's size
        for i in xrange(5 * self.size_of_generation):
            self.models.append(self.get_random_model())

        # if there was initial model
        if self.one_initial_model is not None:
            self.models.append(self.one_initial_model)
        # sort by fintess function and select first size_of_generation models
        self.select(self.params.size_of_generation)
        self.print_mutation_rate_and_strength()

        support.print_best_logll_model_long(self.log_file, self.models[0], self.params)

    def get_mutated_model(
            self,
            mutation_strength=None,
            mutation_rate=None,
            p=None):
        """Get the mutated model from population.

        mutation_rate :  the rate to change one parameter of the selected model
        (if None then mutation_rate from self.params will be chosen)
        p :  probabilities to choose a model for mutation
        """
        # choose a model to mutate
        model = np.random.choice(self.models, p=p)

        # choose a parameter of the model to mutate and get two new models
        # increasing and decreasing this parameter
        new_model_1 = copy.deepcopy(model)
        inds_and_signs = new_model_1.mutate(mutation_strength, mutation_rate)

        # mark changed indices in parent model
        for i, sign in inds_and_signs:
            model.number_of_changes[i] += 1

        new_model_2 = copy.deepcopy(model)
        new_model_2.mutate(
            mutation_strength, mutation_rate, [
                (ind, -sgn) for ind, sgn in inds_and_signs])

        # if change is good then we increase prob to choose it again
        if new_model_1.get_fitness_func_value(
        ) < new_model_2.get_fitness_func_value():
            if model.get_fitness_func_value(
            ) > new_model_1.get_fitness_func_value():
                for index, sign in inds_and_signs:
                    new_model_1.number_of_changes[index] -= 1
            return new_model_1
        else:
            if model.get_fitness_func_value(
            ) > new_model_2.get_fitness_func_value():
                for index, sign in inds_and_signs:
                    new_model_2.number_of_changes[index] -= 1
            return new_model_2

    def get_random_model(self, structure=None):
        return Demographic_model(self.params, structure=structure)

    def select(self, size, print_iter=True):
        """Selection in population of models.

        size :   size of result population

        If size of current population is bigger than size, then we discard the worst.
        """
        self.models = sorted(
            self.models, key=lambda x: x.get_fitness_func_value())[:size]
        if print_iter:
            support.write_to_file(self.log_file,
                                  '\n\nIteration #' + str(self.cur_iteration) + '.')

        support.print_set_of_models(self.log_file, enumerate(self.models), 
                self.params, first_col='N\t', heading='Current population of models:')

    def upgrade_model(self, model, mutation_rate=None):
        """Step for hill climbing.

        model :  the model to be improved
        mutation_rate :  the rate to mutate one parameter in the model

        Returns pair:
        True/False if model became better
        and the best model among all observed models: new mutated and old one.
        """
        new_model = copy.deepcopy(model)
        # mutate one parameter of model and take its index and sign
        index, sign = new_model.mutate_one(mutation_rate=mutation_rate)
        new_model.info = new_model.info[:-1]

        # first try to improve by sign, then if bad try to improve with -sign
        for i in xrange(2):
            if i == 1:
                new_model = copy.deepcopy(model)
                index, sign = new_model.mutate_one(
                    mutation_rate, index_and_sign=(index, -sign))
                new_model.info = new_model.info[:-1]

            if new_model.get_fitness_func_value(
            ) < model.get_fitness_func_value():
                new_new_model = copy.deepcopy(new_model)
                index, sign = new_new_model.mutate_one(
                    mutation_rate, (index, sign))
                new_new_model.info = new_new_model.info[:-1]
                counter = 0
                # if improvement is good then try again
                while (new_new_model.get_fitness_func_value() <
                       new_model.get_fitness_func_value()) and counter < 50:
                    new_model = new_new_model
                    new_model.number_of_changes[index] -= 1
                    new_new_model = copy.deepcopy(new_new_model)
                    index, sign = new_new_model.mutate_one(mutation_rate,
                                                           (index, sign))
                    new_new_model.info = new_new_model.info[:-1]
                    counter += 1
                return (True, new_model)
        return (False, model)

    def mean_time(self):
        """Get mean time for one iteration."""
        return self.work_time / (self.cur_iteration + 1 - self.first_iteration)

    def best_model(self):
        """Get best model in current population."""
        return self.models[0]

    def best_fitness_value(self):
        """Get best fitness value of current population of models."""
        return self.best_model().get_fitness_func_value()

    def is_stoped(self):
        """Check if we need to stop."""
        return self.without_changes >= self.it_without_changes_to_stop_ga or (
            self.models[0].get_number_of_params() - int(not self.params.multinom) == 0)

    def check_best_aic(self, final=True):
        """Check if we have best by AIC model on current iteration.

        If so, we print it to file.
        """
        if self.params.linked_snp or self.is_custom_model or (self.params.initial_structure == self.params.final_structure).all():
            return 
        if self.best_model_by_aic is None or self.best_model_by_aic.get_aic_score() - \
                self.best_model().get_aic_score() > 1e-8:
            self.best_model_by_aic = copy.deepcopy(self.best_model())

            self.print_and_draw_best_model(best_by='AIC', final=final)

    def check_claic(self):
        if self.params.linked_snp:
            self.best_model().get_claic() 
            self.print_and_draw_best_model(best_by='CLAIC', final=final)
            supprt.print_one_model_long(log_file, self.best_model(), params, heading='\n--Calculate CLAIC of the current best model--')


    def print_and_draw_best_model(self, suffix='', best_by='logLL', final=False):
        # print currrent best model
        if self.out_dir is None:
            return
        if final:
            prefix = 'result_'
        else:
            prefix = 'current_'
            
        if best_by == 'logLL' or best_by == 'CLAIC':
            model = self.models[0]
        elif best_by == 'AIC':
            model = self.best_model_by_aic

        support.print_model_code(self.out_dir, model, self.params, prefix=prefix + 'best_' + best_by.lower() + '_model')

        if final:
            support.save_model_plot(
                    os.path.join(self.out_dir, prefix + 'best_' + best_by.lower() + '_model.png'), 
                    model,
                    self.params, 
                    title='Iteration ' + str(self.cur_iteration) + suffix)

        if best_by != 'logLL':
            return
        if (not self.params.code_iter ==
                    0) and self.cur_iteration % self.params.code_iter == 0:
            # print best model's code
            code_dir = os.path.join(self.out_dir, 'python_code')
            support.print_model_code(code_dir, model, self.params, 
                    prefix='iteration_' + str(self.cur_iteration), suffix=suffix, sub_folders=True)

        # draw its picture every self.params.draw_iter iteration
        if not self.params.draw_iter == 0 and self.cur_iteration % self.params.draw_iter == 0:
            support.save_model_plot(
                    os.path.join(self.out_dir, 'pictures', 'iteration_' + str(self.cur_iteration) + '.png'), 
                    model, 
                    self.params, 
                    title='Iteration ' + str(self.cur_iteration) + suffix)


    def run_one_iteration(self):
        """Iteration step for genetic algorithm."""
        start = time.time()

        # take  self.number_of_old_models best models from previous population
        new_models = self.models[:self.number_of_old_models]

        # create probabilities to choose models for crossing and mutation
        p = []
        for m in self.models:
            p.append(m.get_fitness_func_value())
        p = np.array(p)
        p -= max(p) + 1
        p = -p
        p /= sum(p)

        # add mutated models to our new population
        for i in xrange(self.number_of_mutated_models):
            model = self.get_mutated_model(
                mutation_strength=self.cur_mutation_strength,
                mutation_rate=self.cur_mutation_rate,
                p=p)
            new_models.append(model)

        # add crossed models to our new population
        for i in xrange(self.number_of_crossed_models):
            first_model_index = np.random.choice(xrange(len(self.models)), p=p)
            first_model = self.models[first_model_index]

            p[first_model_index] = 0
            p /= sum(p)
            second_model = np.random.choice(self.models, p=p)

            new_models.append(first_model.cross_with_other(second_model))

        # add random models to our new population
        for i in xrange(self.number_of_random_models):
            new_models.append(
                self.get_random_model(
                    structure=self.models[0].get_structure()))

        # remember prev best value of fitness function
        prev_value_of_fit = self.models[0].get_fitness_func_value()

        # new population become current population
        self.models = new_models

        # sort population by fitness function
        self.select(self.params.size_of_generation)

        self.print_and_draw_best_model() 

        # try to make better mutated and crossed models, random are so by
        # definition
        if self.without_changes == self.it_without_changes_to_stop_ga / 2 or self.is_stoped():
            prev_value_of_fit = self.best_fitness_value()
            support.write_to_file(
                self.log_file,
                '\nTime to normalize models with optmal_sfs_scaling:')
            for model in new_models:
                if model.info != 'r':
                    model.normalize_by_Nref()
            self.select(self.size_of_generation, print_iter=False)

        # update our adaptive mutation_rate and print it to log file
        self.update_mutation_rate_and_strength(prev_value_of_fit)
        self.print_mutation_rate_and_strength()
        self.check_best_aic()


        # check if we get stuck
        if abs(
                prev_value_of_fit -
                self.best_fitness_value()) < self.params.epsilon:
            self.without_changes += 1
        else:
            self.without_changes = 0
        self.cur_iteration += 1

        # print results to log file
        support.print_best_logll_model_long(self.log_file, self.best_model(), self.params)

        stop = time.time()
        self.work_time += stop - start
        support.write_to_file(self.log_file, '\nMean time: ', self.mean_time())

    def run_hill_climbing_of_best(self):
        local_without_changes = 0
        it = 0
        old_cur_mut_rate = self.cur_mutation_rate
        self.cur_mutation_rate = self.params.hc_mutation_rate
        while local_without_changes < self.params.hc_stop_iter:
            prev_value_of_fit = self.models[0].get_fitness_func_value()

            flag, self.models[0] = self.upgrade_model(
                self.models[0], self.cur_mutation_rate)

            mut_rate = self.update_mutation_rate(
                self.params.hc_const_for_mut_rate,
                prev_value_of_fit < self.models[0])

            if flag:
                local_without_changes = 0
                support.print_one_model_short(self.log_file, self.models[0], self.params, first_col=str(it))
                
                print_and_draw_best_model(self, suffix='')
                self.check_best_aic()
                if shared_dict is not None:
                    shared_dict[self.prefix] = (
                        self.models[0], self.final_models)
            else:
                local_without_changes += 1
            self.update_mutation_rate(flag)
            it += 1

            # print adaptive mutation_rate
            self.print_mutation_rate_and_strength()

            # normalize model sometime
            if local_without_changes == self.params.hc_stop_iter / 2 or local_without_changes == self.params.hc_stop_iter:
                self.models[0].normalize_by_Nref()
        self.cur_mutation_rate = old_cur_mut_rate

    def run(self, shared_dict=None):
        """Main function to run genetic algorithm.

        shared_dict :   dictionary to share information among processes.
        """
        # checking dirs
        if self.prefix is not None:
            self.out_dir = os.path.join(self.params.output_dir, self.prefix)
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
            if not self.params.draw_iter == 0:
                support.ensure_dir_existence(
                    os.path.join(self.out_dir, 'pictures'))
            if not self.params.code_iter == 0:
                support.ensure_dir_existence(
                    os.path.join(self.out_dir, 'python_code'))
                support.ensure_dir_existence(os.path.join(
                    self.out_dir, 'python_code', 'dadi'))
                support.ensure_dir_existence(os.path.join(
                    self.out_dir, 'python_code', 'moments'))
            self.log_file = os.path.join(self.out_dir, 'GADMA_GA.log')
            open(self.log_file, 'a').close()
        else:
            self.log_file = None

        # help functions
        def run_one_ga_and_one_ls():
            while (not self.is_stoped() and self.run_before_ls):
                self.run_one_iteration()
                if shared_dict is not None:
                    shared_dict[self.prefix] = (
                        copy.deepcopy(self.models[0]), self.final_models)
            if not self.run_before_ls:
                self.run_before_ls = True
            if self.run_ls:
                best_model = copy.deepcopy(self.models[0])
                support.write_to_file(
                    self.log_file,
                    '\nTry to improve best model (' +
                    self.params.optimize_name +
                    ')')
                try: # catch error of `Factor is exactly singular`
                    if self.params.optimize_name != 'hill_climbing':
                        if self.out_dir is not None:
                            self.models[0].run_local_search(self.params.optimize_name, os.path.join(
                                self.out_dir, self.params.optimize_name + '_' + str(self.cur_iteration) + '_out'))
                        else:
                            self.models[0].run_local_search(self.params.optimize_name, None)
                        self.check_best_aic()
                    else:
                        self.run_hill_climbing_of_best()
                    self.print_and_draw_best_model(suffix='_ls')
                except RuntimeError as e:
                    if e.message == 'Factor is exactly singular':
                        support.write_log(self.log_file, 
                                'Local search failed of the following error: Factor is exactly singular.')
                    self.models[0] = best_model
                        
            if not self.run_ls:
                self.run_ls = True

            if shared_dict is not None:
                shared_dict[self.prefix] = (
                    copy.deepcopy(self.models[0]), self.final_models)
            self.check_best_aic()
            self.check_claic()
            self.final_models.append(copy.deepcopy(self.best_model()))

            support.print_final_model(self.log_file, self.models[0], self.params)

            self.cur_iteration += 1


        def increase_models_complexity():
            support.write_to_file(self.log_file,
                                  "\nIncrease models' complexities:")
            index = None
            for model in self.models:
                index = model.increase_complexity(index)
            self.without_changes = 0
            self.select(self.size_of_generation, print_iter=False)
            self.cur_mutation_rate = self.params.mutation_rate
            self.cur_mutation_strength = self.params.mutation_strength

        # begin
        support.write_to_file(self.log_file,
                              '--Start genetic algorithm pipeline--')

        # initialization of first population
        self.init_first_population_of_models()        
        if shared_dict is not None:
            shared_dict[
                self.prefix] = (
                copy.deepcopy(
                    self.models[0]),
                self.final_models)
        self.print_and_draw_best_model()
 

        self.cur_iteration += 1

        # main part
        run_one_ga_and_one_ls()
        
        if not self.is_custom_model:
            while not (self.models[0].get_structure() ==
                       self.params.final_structure).all():
                increase_models_complexity()
                run_one_ga_and_one_ls()

        for model in self.models:
            model.info = 'f'
        self.check_best_aic()

        if shared_dict is not None:
            shared_dict[self.prefix] = (
                copy.deepcopy(self.models[0]), self.final_models)

        # final part
        if self.out_dir is not None:
            self.print_and_draw_best_model(final=True)

        return self.best_model()

    def update_mutation_rate(self, flag):
        if not flag:
            # nothing changed
            self.cur_mutation_rate /= (
                self.params.const_for_mut_rate) ** (0.25)
        else:
            # became better
            self.cur_mutation_rate *= self.params.const_for_mut_rate
            self.cur_mutation_rate = min(
                self.cur_mutation_rate, self.max_mutation_rate)

    def update_mutation_strength(self, flag):
        if not flag or not self.models[0].info.endswith('m'):
            # nothing changed
            self.cur_mutation_strength /= (
                self.params.const_for_mut_strength) ** (0.25)
            self.cur_mutation_strength = max(
                self.cur_mutation_strength,
                1.0 / self.models[0].get_number_of_params())
        else:
            # became better
            self.cur_mutation_strength *= self.params.const_for_mut_strength
            self.cur_mutation_strength = min(self.cur_mutation_strength, 1.0)

    def update_mutation_rate_and_strength(self, prev_value_of_fit):
        flag = (
            prev_value_of_fit -
            self.models[0].get_fitness_func_value()) > 1e-8
        self.update_mutation_rate(flag)
        self.update_mutation_strength(flag)

    def print_mutation_rate_and_strength(self):
        support.write_to_file(
            self.log_file,
            'Current mean mutation rate: ' +
            support.float_representation(
                self.cur_mutation_rate,
                5))
        n_params = self.models[0].get_number_of_params()
        support.write_to_file(self.log_file,
                              'Current mean number of params to change: ' +
                              str(max(1, int(n_params * self.cur_mutation_strength))))
