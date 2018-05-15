import os,sys

import numpy
from numpy import logical_and, logical_not
import support
from genetic_algorithm import GA

import numpy as np
import math
import random
import copy

try:
    import dadi as sim_sfs_lib
    import moments as sim_sfs_lib
except ImportError:
    try:
        import moments as sim_sfs_lib
    except ImportError:
        try:
            import dadi
        except ImportError:
            support.error("None of the dadi or the moments are installed")


class Demographic_model:
    def __init__(self, global_params, structure=None, initial_params=None):
        self.global_params = global_params
        self.number_of_params = global_params.number_of_params
        self.lower_bound = global_params.lower_bound
        self.upper_bound = global_params.upper_bound

        self.info = ''
        
        if initial_params is None:
            self.params = [np.random.uniform(low=self.lower_bound[i], high=self.upper_bound[i]) for i in xrange(self.number_of_params)]
            self.info = 'r'
        else:
            self.params = initial_params

        self.fitness_func_value = None
        self.bic_score = None
        
        self.number_of_changes = np.array(
            [0 for _ in xrange(self.number_of_params)], dtype=float)

        self.cur_structure = structure

    def get_number_of_params(self):
        return self.number_of_params
        
    def has_changed(self):
        self.sfs = None
        self.fitness_func_value = None
        self.bic_score = None

    def mutate_one(self, mutation_rate=None, index_and_sign=None):
        '''
        Mutate by one parameter.

        mutation_rate : percent to change chosen parameter, default is in self.params
        index_and_sign :    pair (index, sign). if not None then index is number of parameter to change by sign

        If index_and_sign is None then parameter is chosen randomly according to probability, which defines by number of changes of each parameters.
        Returns index and sign of parameter that was changed.
        '''
        if mutation_rate is None:
            mutation_rate = self.global_mutation_rate
            
        self.number_of_changes = np.array(self.number_of_changes)
        if min(self.number_of_changes) > 1:
            self.number_of_changes /= min(self.number_of_changes)
        if index_and_sign is None:
            # calculate probabilities and choose parameter
            p = max(self.number_of_changes) + 1 - self.number_of_changes
            p /= sum(p)
            i = np.random.choice(xrange(len(p)), p=p)
            sign = random.choice([-1, 1])
        else:
            i, sign = index_and_sign

        # mutate by parameter
        change = support.sample_from_truncated_normal(mutation_rate)
        while change == 0:
            change = support.sample_from_truncated_normal(mutation_rate)
        
        self.params[i] *= 1 + sign * change
        self.params[i] = max(self.lower_bound[i], self.params[i])
        self.params[i] = min(self.upper_bound[i], self.params[i])
        
        # remember changes
        self.number_of_changes[i] += 1

        return i, sign

    def mutate(self, mutation_strength=None, mutation_rate=None, inds_and_signs=None):
        '''
        Mutate by one parameter.

        mutation_strength : mean fraction of parameters to change
        mutation_rate :     mean rate to change chosen parameter
        inds_and_signs :    if None they will be chosed randomly

        Choose max(mutation_strength*number of params, 1) and mutate them by mutation_rate.
        Returns index and sign of parameters that were changed.
        '''

        if mutation_strength is None:
            mutation_strength = self.global_params.mutation_strength
        
        number_of_params_to_change = max(1, int(self.number_of_params * mutation_strength))

        if inds_and_signs is None:
            # calculate probabilities and choose parameter
            p = max(self.number_of_changes) + 1 - self.number_of_changes
            p /= sum(p)

            #choose parameters
            inds = np.random.choice(xrange(len(p)), size=number_of_params_to_change, replace=False, p=p)
            signs = np.random.choice([-1, 1], size=number_of_params_to_change)

            inds_and_signs = [(inds[i], signs[i]) for i in xrange(number_of_params_to_change)]
             
        for i in xrange(number_of_params_to_change):
            self.mutate_one(mutation_rate, inds_and_signs[i])
        
        self.info += 'm'
        self.has_changed()

        return inds_and_signs

    def cross_with_other(self, other):
        '''
        Crossover with other model for genetic algorithm. 
        '''
        take_from_self = np.array([random.random() for _ in xrange(self.number_of_params)]) > 0.5
        child = copy.deepcopy(other)
        child.params = [self.params[i] if take_from_self[i] else other.params[i] for i in xrange(self.number_of_params)]
        child.number_of_changes = [self.number_of_changes[i] if take_from_self[i] else other.number_of_changes[i] for i in xrange(self.number_of_params)]

        child.info = 'c'
        child.has_changed()
        return child

    def get_sfs(self):
        '''
        Get SFS from model.
        '''
        ns = self.global_params.data.sample_sizes
        if self.global_params.pts is not None:
            self.sfs = self.global_params.model_func(self.params, ns, self.global_params.pts)
        else:
            self.sfs = self.global_params.model_func(self.params, ns)
        return self.sfs

    def get_fitness_func_value(self):
        '''
        Calculate fitness function value for the model.
        '''
        if self.fitness_func_value is None:
            sfs = self.get_sfs()
            data = self.global_params.data
            if self.global_params.multinom:
                log_likelihood = sim_sfs_lib.Inference.ll_multinom(self.sfs, data)
            else:
                log_likelihood = dadi.Inference.ll(self.sfs, data)
            self.fitness_func_value = - log_likelihood
            ns = self.global_params.data.sample_sizes
            self.bic_score = math.log(np.prod(
                np.array(ns))) * self.number_of_params - 2 * log_likelihood

        return_value = self.fitness_func_value
        return return_value

    def get_bic_score(self):
        '''
        Calculate BIC score for the model.
        '''
        if self.bic_score is None:
            self.get_fitness_func_value()
        return self.bic_score

    def __str__(self):
        return '[' +', '.join(["%10.3f" % x for x in self.params]) + ']\t' + self.info


class params_storage:
    def __init__(self, number_of_params,  data, model_func, pts, lower_bound, upper_bound, p0, multinom,
            mutation_strength, const_for_mut_strength, mutation_rate, const, epsilon, size_of_population_in_ga, frac_of_old_models,
            frac_of_mutated_models, frac_of_crossed_models, stop_iter):
        self.number_of_params = number_of_params
        self.data = data
        self.model_func = model_func
        self.pts = pts
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p0 = p0
        self.multinom = multinom
        self.size_of_population = size_of_population_in_ga
        self.frac_of_old_models = frac_of_old_models
        self.frac_of_mutated_models = frac_of_mutated_models
        self.frac_of_crossed_models = frac_of_crossed_models
        self.mutation_strength = mutation_strength
        self.const_for_mut_strength = const_for_mut_strength
        self.mutation_rate = mutation_rate
        self.const_for_mut_rate = const
        self.epsilon = epsilon
        self.draw_iter = 0
        self.code_iter = 0
        self.stop_iter = stop_iter


def optimize_ga(number_of_params, data, model_func, pts=None, lower_bound=None, upper_bound=None, p0=None,
                 multinom=True, mutation_strength=0.3, const_for_mut_strength=1.0, mutation_rate=0.1, const_for_mut_rate=1.1,
                 epsilon=1e-2, stop_iter=100, size_of_population_in_ga=10, frac_of_old_models=0.2, frac_of_mutated_models=0.3, 
                 frac_of_crossed_models=0.3):
    """
    Find optimized params to fit model to data using Genetic Algorithm.
    
    Note: dadi and moments are choosing from value of pts argument!
    If pts is None then moments will be used.

    This optimization method is method for global search. It starts from 
    random parameters and improve them to get best.

    number_of_params :  Number of parameters to find.

    data :              Spectrum with data.
    
    model_func :        Function to evaluate model spectrum. Should take arguments
                        parameters, (n1,n2...) (and pts in case of dadi!).
    
    pts :               Number of grid points for dadi. If you use moments, 
                        don't set it or set to None.
    
    lower_bound :       Lower bound on parameter values. If not None, must be of
                        length equal to number_of_params.

    upper_bound :       Upper bound on parameter values. If not None, must be of
                        length equal to number_of_params.

    p0 :                Initial parameters. You can start from some known parameters.
                        However it is not proper global search.

    multinom :          If True, do a multinomial fit where model is optimially scaled to
                        data at each step. If False, assume theta is a parameter and do
                        no scaling.


    mutation_strength :     Mean fraction of parameters to mutate during global mutation
                            process.

    const_for_mut_strength: Const for adaptive mutation strength. Must be between 1 and 2.

    mutation_rate :         Mean rate to change the parameter during its mutation.

    const_for_mut_rate :    Const for adaptive mutation rate. Must be between 1 and 2.

    epsilon :               Const for model's log likelihood compare.
                            Model is better if its log likelihood is greater than 
                            log likelihood of another model by epsilon.

    stop_iter :             Number of iterations for GA stopping: GA stops when 
                            it can't improve model during max_iter iterations.

    size_of_population_in_ga: Number of models in GA.

    frac_of_old_models :    Fraction of models from previous ga population that are taken 
                            to new population.

    frac_of_mutated_models: Fraction of mutated models in new population.

    frac_of_crossed_models: Fraction of crossed models in new population.
    
    """
    global_params = params_storage(number_of_params, data, model_func, pts, lower_bound, upper_bound, p0, multinom,
            mutation_strength, const_for_mut_strength, mutation_rate, const_for_mut_rate, epsilon, size_of_population_in_ga, frac_of_old_models,
            frac_of_mutated_models, frac_of_crossed_models, stop_iter)
    ga_instance = GA(global_params, chromosomes_only=True, one_initial_model=Demographic_model(global_params, initial_params=p0))
    best_model = ga_instance.run()
    return best_model.params

