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
            chromosomes_only=False,
            one_initial_model=None):
        """Genetic algorithm class.

        params :    an object with parameters to work with
            parameters for genetic algorithm:
            size_of_population :    size of the population of demographic models.
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
        prefix :    prefix for output folder and so on.
        chromosomes_only :  if we want to consider dem models as lists.
        """
        # all parameters
        self.params = params

        # some constants on number of iterations:
        # number of iterations, during which fitness function doesn't change, when we stop ga
        # look for is_stoped function
        self.it_without_changes_to_stop_ga = self.params.stop_iter

        self.prefix = prefix
        self.chromosomes_only = chromosomes_only
        self.one_initial_model = one_initial_model
        self.ll_precision = 1 - int(math.log(self.params.epsilon, 10))

        # our adaptive mutation_rate
        self.cur_mutation_rate = self.params.mutation_rate
        self.cur_mutation_strength = self.params.mutation_strength

        # helpful parameters
        self.size_of_population = self.params.size_of_population
        self.number_of_old_models = int(
            params.size_of_population * params.frac_of_old_models)
        self.number_of_mutated_models = int(
            params.size_of_population * params.frac_of_mutated_models)
        self.number_of_crossed_models = int(params.size_of_population * params.
                                            frac_of_crossed_models)
        self.number_of_random_models = params.size_of_population - \
            self.number_of_old_models - self.number_of_mutated_models - \
            self.number_of_crossed_models

        # basic parameters
        self.cur_iteration = 0
        self.first_iteration = 0  # is not 0 if we restore ga
        self.work_time = 0
        self.models = []

        # variables for stops
        self.without_changes = 0

        # connected files and directories
        self.out_dir = None
        self.log_file = None
        self.best_model_by_bic = None

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
            self.cur_iteration = int(iter_out[0].split('#')[-1][:-1])
            self.first_iteration = self.cur_iteration

            start_ind = 3
            end_ind = 3 + size
            restore_from_cur_pop_of_models(iter_out[start_ind: end_ind])

            if not iter_out[end_ind].startswith('Current mean mutation rate:'):
                raise RuntimeError(
                    "Expect 'Current mean mutation rate:' after current population during restoring.")
            self.cur_mutation_rate = float(iter_out[end_ind].split(':')[-1])

            if not iter_out[
                    end_ind +
                    1].startswith('Current mean number of params to change:'):
                raise RuntimeError(
                    "Expect 'Current mean number of params to change:' after current population during restoring.")
            self.cur_mutation_strength = float(
                iter_out[end_ind + 1].split(':')[-1]) / self.models[0].get_number_of_params()

        def restore_from_ls_string(ls_string, is_best=False):
            if is_best:
                index = 1
            else:
                index = 2
            self.models[0] = Demographic_model(
                self.params, restore_string=ls_string.strip().split('\t')[index])

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

        if pos_of_last_empty_str - 11 > size:
            if iter_out[-1].startswith(
                    'BEST') and iter_out[-2].startswith('Try to improve'):
                # when we don't print final result
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
            restore_from_ls_string(
                iter_out[-1], iter_out[-2].startswith('BEST'))
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
            for i in xrange(self.params.size_of_population - len(self.models)):
                self.models.append(self.get_random_model())
            if self.params.size_of_population > len(self.models):
                self.select(self.params.size_of_population)
            return
        # generate random models 5 times more than the population's size
        for i in xrange(5 * self.size_of_population):
            self.models.append(self.get_random_model())

        # if there was initial model
        if self.one_initial_model is not None:
            self.models.append(self.one_initial_model)
        # sort by fintess function and select first size_of_population models
        self.select(self.params.size_of_population)
        self.print_mutation_rate_and_strength()

        support.write_to_file(self.log_file,
                              'Best log likelihood:',
                              support.float_representation(-self.best_fitness_value(),
                                                           self.ll_precision))
        support.write_to_file(
            self.log_file,
            'with BIC score:\t',
            support.float_representation(
                self.best_model().get_bic_score(),
                self.ll_precision))
        support.write_to_file(self.log_file, 'Model:', self.models[0])

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
        if self.chromosomes_only:
            from Inference import Demographic_model
        else:
            from demographic_model import Demographic_model
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
        support.write_to_file(
            self.log_file,
            'Current population of models:\nN\tlogLL\t\t\tBIC\t\t\t\tmodel')
        for i, model in enumerate(self.models):
            support.write_to_file(
                self.log_file,
                i,
                support.float_representation(
                    -model.get_fitness_func_value(),
                    self.ll_precision),
                support.float_representation(
                    model.get_bic_score(),
                    self.ll_precision),
                model)
        self.check_best_bic()

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

    def check_best_bic(self):
        """Check if we have best by BIC model on current iteration.

        If so, we print it to file.
        """
        if self.best_model_by_bic is None or self.best_model_by_bic.get_bic_score() - \
                self.best_model().get_bic_score() > 1e-8:
            self.best_model_by_bic = copy.deepcopy(self.best_model())
            if not self.chromosomes_only:
                self.best_model_by_bic.dadi_code_to_file(
                    os.path.join(self.out_dir,
                                 'current_best_bic_model.py'))
                self.best_model_by_bic.moments_code_to_file(
                    os.path.join(self.out_dir,
                                 'current_best_bic_model_moments.py'))

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
                    structure=self.models[0].cur_structure))

        # remember prev best value of fitness function
        prev_value_of_fit = self.models[0].get_fitness_func_value()

        # new population become current population
        self.models = new_models

        # sort population by fitness function
        self.select(self.params.size_of_population)

        if not self.chromosomes_only:
            # print currrent best model

            self.models[0].dadi_code_to_file(
                os.path.join(self.out_dir,
                             'current_best_logLL_model.py'))
            self.models[0].moments_code_to_file(
                os.path.join(self.out_dir,
                             'current_best_logLL_model_moments.py'))
            if (not self.params.code_iter ==
                    0) and self.cur_iteration % self.params.code_iter == 0:
                # print best model's code
                self.models[0].dadi_code_to_file(
                    os.path.join(self.out_dir, 'python_code', 'dadi',
                                 'iteration_' + str(self.cur_iteration) + '.py'))
                self.models[0].moments_code_to_file(
                    os.path.join(self.out_dir, 'python_code', 'moments',
                                 'iteration_' + str(self.cur_iteration) + '.py'))

        # draw its picture every self.params.draw_iter iteration
        if not self.params.draw_iter == 0 and self.cur_iteration % self.params.draw_iter == 0:
            if not self.params.draw_iter == 0 and self.cur_iteration % self.params.draw_iter == 0:
                self.best_model().draw(
                    os.path.join(self.out_dir, 'pictures',
                                 'iteration_' + str(self.cur_iteration) + '.png'),
                    'Iteration ' + str(self.cur_iteration) + ', logLL: ' +
                    support.float_representation(-self.best_fitness_value(), self.ll_precision) + ', BIC: ' +
                    support.float_representation(self.best_model().get_bic_score(), self.ll_precision))

        # try to make better mutated and crossed models, random are so by
        # definition
        if not self.chromosomes_only and (
                self.without_changes == self.it_without_changes_to_stop_ga /
                2 or self.is_stoped()):
            prev_value_of_fit = self.best_fitness_value()
            support.write_to_file(
                self.log_file,
                '\nTime to normalize models with optmal_sfs_scaling:')
            for model in new_models:
                if model.info != 'r':
                    model.normalize_by_Nref()
            self.select(self.size_of_population, print_iter=False)

        # update our adaptive mutation_rate and print it to log file
        self.update_mutation_rate_and_strength(prev_value_of_fit)
        self.print_mutation_rate_and_strength()

        # check if we get stuck
        if abs(
                prev_value_of_fit -
                self.best_fitness_value()) < self.params.epsilon:
            self.without_changes += 1
        else:
            self.without_changes = 0
        self.cur_iteration += 1

        # print results and statistics to log file
        support.write_to_file(self.log_file,
                              '\nBest log likelihood:',
                              support.float_representation(-self.best_fitness_value(),
                                                           self.ll_precision))
        support.write_to_file(
            self.log_file,
            'with BIC score:\t',
            support.float_representation(
                self.best_model().get_bic_score(),
                self.ll_precision))
        support.write_to_file(self.log_file, 'Model:', self.models[0])
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
                support.write_to_file(
                    self.log_file,
                    it,
                    support.float_representation(
                        -self.models[0].get_fitness_func_value(),
                        self.ll_precision),
                    self.models[0])
                if not self.params.code_iter == 0 and self.cur_iteration % self.params.code_iter == 0:
                    self.models[0].dadi_code_to_file(
                        os.path.join(self.out_dir, 'python_code', 'dadi',
                                     'iteration_' + str(self.cur_iteration) + '_after_hc.py'))
                    self.models[0].moments_code_to_file(
                        os.path.join(self.out_dir, 'python_code', 'moments',
                                     'iteration_' + str(self.cur_iteration) + '_after_hc.py'))
                self.check_best_bic()
                if shared_dict is not None:
                    shared_dict[self.prefix] = (
                        self.models[0], self.best_model_by_bic)
            else:
                local_without_changes += 1
            self.update_mutation_rate(flag)
            it += 1

            # print adaptive mutation_rate
            self.print_mutation_rate_and_strength()

            # normalize model sometime
            if not self.chromosomes_only and (
                    local_without_changes == self.params.hc_stop_iter /
                    2 or local_without_changes == self.params.hc_stop_iter):
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
            while (not self.is_stoped()):
                self.run_one_iteration()
                if shared_dict is not None:
                    shared_dict[self.prefix] = (
                        copy.deepcopy(self.models[0]), self.best_model_by_bic)
            if self.chromosomes_only:
                return
            support.write_to_file(
                self.log_file,
                '\nTry to improve best model (' +
                self.params.optimize_name +
                ')')
            if self.params.optimize_name != 'hill_climbing':
                self.models[0].run_local_search(self.params.optimize_name, os.path.join(
                    self.out_dir, self.params.optimize_name + '_' + str(self.cur_iteration) + '_out'))
                self.check_best_bic()
            else:
                self.run_hill_climbing_of_best()

            if shared_dict is not None:
                shared_dict[self.prefix] = (
                    copy.deepcopy(self.models[0]), self.best_model_by_bic)

            if self.params.code_iter != 0 and self.cur_iteration % self.params.code_iter == 0:
                self.models[0].dadi_code_to_file(
                    os.path.join(self.out_dir, 'python_code', 'dadi',
                                 'iteration_' + str(self.cur_iteration) +
                                 '_after_local_search.py'))
                self.models[0].moments_code_to_file(
                    os.path.join(self.out_dir, 'python_code', 'moments',
                                 'iteration_' + str(self.cur_iteration) +
                                 '_after_local_search.py'))
            self.check_best_bic()

            support.write_to_file(self.log_file, 'BEST:')
            support.write_to_file(
                self.log_file,
                support.float_representation(
                    -self.best_fitness_value(),
                    self.ll_precision),
                self.models[0])

            if not self.params.draw_iter == 0 and self.cur_iteration % self.params.draw_iter == 0:
                self.best_model().draw(
                    os.path.join(self.out_dir, 'pictures',
                                 'iteration_' + str(self.cur_iteration) + '_after_local_search.png'),
                    'Iteration ' + str(self.cur_iteration) + '(LS), logLL: ' +
                    support.float_representation(-self.best_model().get_fitness_func_value(),
                                                 self.ll_precision) + ', BIC: ' +
                    support.float_representation(self.best_model().get_bic_score(),
                                                 self.ll_precision))

        def increase_models_complexity():
            support.write_to_file(self.log_file,
                                  "\nIncrease models' complexities:")
            index = None
            for model in self.models:
                index = model.increase_complexity(index)
            self.without_changes = 0
            self.select(self.size_of_population, print_iter=False)
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
                self.best_model_by_bic)

        self.cur_iteration += 1

        # main part
        run_one_ga_and_one_ls()
        if not self.chromosomes_only:
            while not (self.models[0].cur_structure ==
                       self.params.final_structure).all():
                increase_models_complexity()
                run_one_ga_and_one_ls()

        for model in self.models:
            model.info = 'f'
        self.check_best_bic()

        if shared_dict is not None:
            shared_dict[self.prefix] = (
                copy.deepcopy(self.models[0]), self.best_model_by_bic)

        # final part
        if not self.params.draw_iter == 0 and self.cur_iteration % self.params.draw_iter == 0:
            self.best_model().draw(
                os.path.join(self.out_dir, 'result_model' + '.png'),
                'Iteration ' + str(self.cur_iteration) + ', logLL: ' +
                support.float_representation(-self.models[0].get_fitness_func_value(), self.ll_precision))
        if not self.chromosomes_only:
            self.models[0].dadi_code_to_file(
                os.path.join(self.out_dir,
                             'result_model_dadi_code.py'))
            self.models[0].moments_code_to_file(
                os.path.join(self.out_dir,
                             'result_model_moments_code.py'))
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
