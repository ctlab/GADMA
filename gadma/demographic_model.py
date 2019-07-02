#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

import random
import numpy as np
import copy
import os

import io
import signal

import support
import ast
import math
import sys
import gadma

try:
    import dadi
    import moments
except ImportError:
    try:
        import moments
    except ImportError:
        try:
            import dadi
        except ImportError:
            support.error('None of the dadi or the moments are installed')

if 'matplotlib' in sys.modules:
    # First we make matplotlib backend as Agg
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')



class Period(object):
    """Class Period.

    Base class of Demographic_model.
    """

    def __init__(self,
                 time,
                 sizes_of_populations,
                 growth_types=None,
                 migration_rates=None,
                 is_first_period=False,
                 is_split_of_population=False):
        '''
        time :  time of the period
        sizes_of_populations :  the list of sizes of population AT THE END of
                                the period
        growth_types :  the list of growth types of populations during the
                        period (0 is for sudden change, 1 for linear growth
                        and 2 for exponential)
        migration_rates :   migration rates between populations during the period
        is_first_period :   is True for the first "const" period
        is_split_of_population :    is True for periods that are splits
        '''
        self.time = time
        self.sizes_of_populations = copy.deepcopy(sizes_of_populations)
        self.number_of_populations = len(sizes_of_populations)
        self.only_sudden = growth_types is None
        if growth_types is None:
            if not is_first_period and not is_split_of_population:
                self.growth_types = [0] * self.number_of_populations
            else:
                self.growth_types = None
        else:
            self.growth_types = growth_types
        self.migration_rates = migration_rates
        self.is_first_period = is_first_period
        self.is_split_of_population = is_split_of_population

        # array to remember changes of parameters during mutations
        if self.is_first_period:
            self.number_of_parameters = 1
        elif self.is_split_of_population:
            self.number_of_parameters = 1
        else:
            self.number_of_parameters = 1 + self.number_of_populations * (2 - self.only_sudden) + (
                0 if self.migration_rates is None else
                (2 if self.number_of_populations == 2 else 6))

    def check_params(self, bounds, N_A):
        if self.is_first_period:
            return
        if self.is_split_of_population:
            self.check_prop(bounds.min_N, N_A)
            return
        self.time = max(self.time, bounds.min_T * N_A)
        self.time = min(self.time, bounds.max_T * N_A)

        for i in xrange(self.number_of_populations):
            self.get_sizes_of_populations()[i] = max(
                self.get_sizes_of_populations()[i],
                bounds.min_N * N_A)
            self.get_sizes_of_populations()[i] = min(
                self.get_sizes_of_populations()[i],
                bounds.max_N * N_A)

            if self.migration_rates is not None:
                for j in xrange(self.number_of_populations):
                    if i != j:
                        self.migration_rates[i][j] = max(
                            self.migration_rates[i][j], bounds.min_M / N_A)
                        self.migration_rates[i][j] = min(
                            self.migration_rates[i][j], bounds.max_M / N_A)

    def populations(self):
        """Iterator over populations."""
        return xrange(self.number_of_populations)

    def get_sizes_of_populations(self):
        return self.sizes_of_populations

    def mutate_by_parameter(self, param_index, change, sign, N_A, bounds):
        '''
        Change the parameter number param_index by (1 + mutation_rate) * sign.

        N_A is for right scaling and limitations.
        '''
        # if it is split then there is only one parameter: split percent
        if self.is_split_of_population:
            self.mutate(change, sign, N_A, bounds.min_N)

        # if it is first population then there is only one parameter: size of
        # ancestral population
        elif self.is_first_period:
            self.get_sizes_of_populations()[0] *= 1 + sign * change

        # otherwise change corresponding parameter
        elif param_index == 0:
            self.time *= 1 + sign * change
            self.time = max(self.time, bounds.min_T * N_A)
            self.time = min(self.time, bounds.max_T * N_A)
        elif param_index <= self.number_of_populations:
            self.get_sizes_of_populations()[
                param_index - 1] *= 1 + sign * change
            self.get_sizes_of_populations()[param_index - 1] = max(
                self.get_sizes_of_populations()[param_index - 1],
                bounds.min_N * N_A)
            self.get_sizes_of_populations()[param_index - 1] = min(
                self.get_sizes_of_populations()[param_index - 1],
                bounds.max_N * N_A)

        elif param_index <= 2 * self.number_of_populations and not self.only_sudden:
            self.growth_types[param_index -
                              self.number_of_populations - 1] += sign
            self.growth_types[param_index -
                              self.number_of_populations - 1] %= 3
        else:
            i = param_index - (1 + int(not self.only_sudden)) * self.number_of_populations - 1

            def find_pos(i):
                x = 0
                y = 0
                for x in xrange(self.number_of_populations):
                    for y in xrange(self.number_of_populations):
                        if i == 0 and x != y:
                            return x, y
                        if x != y:
                            i -= 1

            x, y = find_pos(i)
            size_of_pop = self.get_sizes_of_populations()[x]
            if self.migration_rates[x][y] == 0 and sign == 1:
                self.migration_rates[x][y] = np.random.uniform(
                    bounds.min_M / N_A, bounds.max_M / size_of_pop)
            else:
                self.migration_rates[x][y]
                self.migration_rates[x][
                    y] *= 1 + sign * change
                self.migration_rates[x][y] = max(
                    self.migration_rates[x][y], bounds.min_M / N_A)
                self.migration_rates[x][y] = min(
                    self.migration_rates[x][y], bounds.max_M / N_A)
            if self.migration_rates[x][y] * size_of_pop < 1e-3 and sign == -1:
                self.migration_rates[x][y] *= random.choice([0, 1])

    def __str__(self):
        """String representation of period.

        If period is first:
        [ [size of ancestral pop] ]

        Else:
        if one population:
        [ time, [size of population], [growth type] ]
        else:
        [ time, [sizes of populations], [growth types], [migration rates] ]
        """
        if self.is_first_period:
            return '[ ' + list_float_representation(
                [self.get_sizes_of_populations()[0]]) + ' ]'
        if self.is_split_of_population:
            if self.split_prop is None:
                return '[ Split ]'
            else:
                return '[ ' + float_representation(None if self.split_prop is None else 100 * self.split_prop) +\
                    '%, ' + list_float_representation(
                        self.get_sizes_of_populations()) + ' ]'
        if self.migration_rates is None:
            mig_str = ''
        else:
            mig_str = ', ' + migr_float_representation(self.migration_rates)
        return '[ ' + float_representation(self.time) + ', ' + \
            list_float_representation(self.get_sizes_of_populations()) + ', ' + \
            str(self.growth_types)  + mig_str + ' ]'


class Split(Period):
    """Class Split.

    is derived from Period.
    """

    def __init__(self, split_prop, population_to_split,
                 sizes_of_populations_before_split):
        '''
        split_prop : proportion to split population
        population_to_split :   population that is splitted
        sizes_of_populations_before_split : field sizes_of_populations from previous period
        '''
        if population_to_split >= 0:
            self.population_to_split = population_to_split
        else:
            self.population_to_split = len(
                sizes_of_populations_before_split) + population_to_split

        self.sizes_of_pops_before = copy.deepcopy(
            sizes_of_populations_before_split)
        self.split_prop = split_prop
        super(Split, self).__init__(
            time=0,
            sizes_of_populations=self.get_sizes_of_populations(),
            is_split_of_population=True)
        if split_prop is None:
            self.number_of_parameters = 0
        else:
            self.number_of_parameters = 1
        self.number_of_changes = [0.0]

    def get_sizes_of_populations(self):
        sizes_of_pops = copy.deepcopy(self.sizes_of_pops_before)
        sizes_of_pops.append(sizes_of_pops[-1])
        if self.split_prop is None:
            sizes_of_pops[-1] = None
            sizes_of_pops[-2] = None
        else:
            sizes_of_pops[-1] *= (1 - self.split_prop)
            sizes_of_pops[-2] *= self.split_prop
        return sizes_of_pops

    def check_prop(self, min_N, N_A):
        if self.split_prop is None:
            return True
        min_split_prop = min_N * N_A / \
            self.sizes_of_pops_before[self.population_to_split]
        return_flag = False
        if self.split_prop > 1 - min_split_prop:
            self.split_prop = 1 - min_split_prop
            return_flag = True
        if self.split_prop < min_split_prop:
            self.split_prop = min_split_prop
            return_flag = True
        return return_flag

    def mutate(self, change, sign, min_N, N_A):
        """Change split_prop by mutation_rate.

        N_A is for right scaling and limitations.
        """
        if self.split_prop is None:
            return
        self.split_prop *= 1 + sign * change
        self.check_prop(min_N, N_A)
        
    def update(self, new_size_of_population_before_split, min_N=None, N_A=None):
        """Update sizes of populations of previous period."""
        if self.split_prop is None:
            return 
        self.sizes_of_pops_before = copy.deepcopy(
            new_size_of_population_before_split)
        if min_N is not None and N_A is not None:
            self.check_prop(min_N, N_A)


class Demographic_model:
    """Main class Demographic_model.

    Corresponds to the demographic history with all parameters.
    """

    def __init__(
            self,
            params,
            structure=None,
            random=True,
            restore_string=None,
            initial_vector=None):
        '''
        params :    an object with parameters for working
            input_data :    given SFS
            input_file :    filename of given SFS
            initial_structure : structure of default models
                                (if structure is None)
            final_structure :   structure of final model
                                (for increase_complexity function)
            ns :    number of chromosomes of each population
            dadi_pts :  grid sizes to run DaDi
            theta : total mutation flux
            gen_time :  time for one generation
            pop_labels :    list of population labels
            moments_scenario :   if True, then moments will be run instead DaDi
        structure : the structure of the model. If it is None, then the
                    structure will be params.initial_structure
        random :    if False, then create empty model, else create random
                    model with required structure

        if params.structure is None we consider custom model
        '''

        self.params = params

        if structure is None and self.params.model_func_file is not None:
            self.popt = []
            self.is_custom_model = True
            self.number_of_populations = len(self.params.ns)
            self.popt_len = 0
            self.lower_bound = self.params.lower_bound
            self.upper_bound = self.params.upper_bound
            self.params.multinom = self.params.multinom or self.params.p_ids is None
            self.cur_structure = None
        else:
            self.lower_bound = None
            self.upper_bound = None
            self.is_custom_model = False
            self.number_of_periods = 0
            self.periods = []
            
            self.cur_structure = None
            self.get_structure()

        self.info = ''

        self.split_1_pos = None
        self.split_2_pos = None


        self.dt_fac = 0.1  # for moments
        self.sfs = None
        self.fitness_func_value = None
        self.aic_score = None
        self.claic_score = None
        self.claic_eps = None

        self.param_ids = None
        self.number_of_changes = np.array(
            [], dtype=float)
        self.get_param_ids()

        if restore_string is not None:
            self.from_string(restore_string)
            if not self.is_custom_model:
                for i in xrange(self.number_of_periods):
                    self.periods[i].check_params(self.params, self.get_N_A())
            else:
                lower_bound, upper_bound = self.get_lower_and_upper_bounds()
                for i in xrange(self.get_number_of_params()):
                    self.popt[i] = min(self.popt[i], upper_bound[i])
                    self.popt[i] = max(self.popt[i], lower_bound[i])

        if random and restore_string is None and initial_vector is None:
            self.init_random_model(structure)
        if initial_vector is not None:
            self.construct_from_vector(initial_vector)

    def generate_random_value(self, low_bound, upp_bound, identificator=None):
        """Generate random value for different parameters of models"""
        if identificator == 't' or identificator == 's':
            return np.random.uniform(low_bound, upp_bound)
        # if identificator == 'm' or identificator == 'n' or None
        log = True
        if low_bound <= 0:
            low_bound = 1e-15
        normal = True

        mode = 1.0

        # remember bounds and mean
        l = low_bound # left bound
        u = upp_bound # right bound
        # mean
        if low_bound >= mode:
            m = low_bound
        elif upp_bound <= mode:
            m = upp_bound
        else:
            m = mode
        # determine random function and transform to log if need
        if log:
            l = np.log(l)
            u = np.log(u)
            m = np.log(m)
        if normal:
            random_generator = lambda a,b,c:  support.sample_from_truncated_normal(b, max(b-a, c-b) / 3, a, c)
        else:
            random_generator = np.random.triangular
        # generate sample
        sample = random_generator(l, m, u)

        if log:
            sample = np.exp(sample)
        return sample

    def init_random_model(self, structure):
        """Generate random model of a given structure.
        This method is MAGIC, real magic. It is strange just because it is the best way to do it."""
        if structure is None:
            if self.is_custom_model:
                for i, (low_bound, upp_bound) in enumerate(zip(self.lower_bound, self.upper_bound)):
                    if self.params.p_ids is not None:
                        p_id = self.params.p_ids[i]
                    else:
                        p_id = None
                    self.popt.append(self.generate_random_value(low_bound, upp_bound, p_id))
                    self.popt_len += 1
                    self.number_of_changes = np.append(self.number_of_changes, 0.0)
#                if not self.params.multinom or self.params.p_ids is not None:
#                    Nref = self.generate_random_value(self.params.min_N, self.params.max_N, 'n')
                if not self.params.multinom:
                    self.popt.append(1.0)
                    self.popt_len += 1
                    self.number_of_changes = np.append(self.number_of_changes, 0.0)
#                if self.params.p_ids is not None:
#                    self.normalize_by_Nref(1 / Nref)
#                    for i, (low_bound, upp_bound) in enumerate(zip(self.lower_bound, self.upper_bound)):
#                        self.popt[i] = max(low_bound, self.popt[i])
#                        self.popt[i] = min(upp_bound, self.popt[i])
            else:
                structure = self.params.initial_structure
        
        if not self.is_custom_model:
            # add first "const" period
            if self.params.random_N_A:
                N_A = self.generate_random_value(self.params.min_N, self.params.max_N, 'n')
            else:
                N_A = 1.0
            self.add_period(
                Period(
                    time=0,
                    sizes_of_populations=[N_A],
                    is_first_period=True))
            # add other periods
            for i in xrange(structure[0] - 1):
                self.add_period(
                    Period(
                        time=self.generate_random_value(
                            self.params.min_T, self.params.max_T, 't'),
                        sizes_of_populations=[self.generate_random_value(
                            self.params.min_N, self.params.max_N, 'n')],
                        growth_types= None if self.params.only_sudden else [random.choice([0, 1, 2])]))
            for num_of_pops, number_of_periods in enumerate(structure[1:]):
                num_of_pops += 2
                self.add_period(
                    Split(
                        split_prop=None if self.params.only_sudden else self.generate_random_value(
                            self.params.min_N, 1 - self.params.min_N, 's'),
                        population_to_split=num_of_pops - 2,
                        sizes_of_populations_before_split=self.periods[-1]
                        .get_sizes_of_populations()))
                for i in xrange(number_of_periods):
                    sizes_of_pops = [
                        self.generate_random_value(self.params.min_N, self.params.max_N, 'n')
                        for x in xrange(num_of_pops)
                    ]
                    self.add_period(
                        Period(
                            time=self.generate_random_value(
                                self.params.min_T, self.params.max_T, 't'),
                            sizes_of_populations=sizes_of_pops,
                            growth_types=None if self.params.only_sudden else [
                                random.choice([0, 1, 2])
                                for x in xrange(num_of_pops)
                            ],
                            migration_rates=None if self.params.no_mig else self.generate_migration_rates(
                                sizes_of_pops)))
            for i, period in enumerate(self.periods):
                period.time = max(period.time, N_A * self.params.min_T)
                period.time = min(period.time, N_A * self.params.max_T)
                for p in xrange(period.number_of_populations):
                    not_last = (i + 1 < self.number_of_periods)
                    if not_last:
                        next_is_split = self.periods[i + 1].is_split_of_population
                        if next_is_split and p == self.periods[i +
                                                               1].population_to_split:
                            period.sizes_of_populations[p] = max(
                                period.sizes_of_populations[p], 2 * N_A * self.params.min_N)
                    period.sizes_of_populations[p] = max(
                        period.sizes_of_populations[p], N_A * self.params.min_N)
                    period.sizes_of_populations[p] = min(
                        period.sizes_of_populations[p], N_A * self.params.max_N)

                    if period.migration_rates is None:
                        continue
                    for p2 in period.populations():
                        if p == p2:
                            continue
                        period.migration_rates[p][p2] = max(
                            period.migration_rates[p][p2], self.params.min_M / N_A)
                        period.migration_rates[p][p2] = min(
                            period.migration_rates[p][p2], self.params.max_M / N_A)

        self.normalize_by_Nref()

        self.info = 'r'

    def from_string(self, string):
        """read the model from string, out of __str__ function."""
        string = string.strip()
        if string.endswith(')'):
            last_occ = string.rfind('(')
            if string[last_occ + 1:].startswith('N_A'):
                N_str = 'N_A = '
            else:
                N_str = 'Nref = '
            real_Nref = float(string[last_occ + len(N_str): -1])
            string = string[:last_occ - 1]

        if self.is_custom_model:
            self.popt = [float(x) for x in string[1:-1].split(', ')]
            if not self.params.multinom:
                self.popt.append(real_Nref)
            self.popt_len = len(self.popt)
            self.number_of_changes = np.array([0 for _ in xrange(self.popt_len)])
            self.normalize_by_Nref(real_Nref)
        else:
            periods = string.split(' ][ ')
            periods[0] = periods[0][2:]
            periods[-1] = periods[-1][:-2]

            for p in periods:
                params = p.split(', [')

                if len(self.periods) == 0:
                    NA = float(params[0][1:-1])

                if len(params) == 1:
                    self.add_period(
                        Period(
                            0.0,
                            [float(params[0][1:-1])],
                            is_first_period=True))
                    continue
                if params[0].endswith('%'):
                    sizes = params[1][:-1].split(',')
                    proportion = float(sizes[-2]) / self.periods[-1].get_sizes_of_populations()[-1]
                    if proportion == 1.0:
                        proportion = 1.0 - float(sizes[-1]) / self.periods[-1].get_sizes_of_populations()[-1]
                    if float(sizes[-1]) == 0:
                        proportion = 1 - 1 / NA
                    if float(sizes[-2]) == 0:
                        proportion = 1 / NA
                    self.add_period(
                        Split(
                            None if self.params.only_sudden else proportion,
                            -1,
                            self.periods[-1].get_sizes_of_populations()))
                    self.periods[-1].update(self.periods[-2].get_sizes_of_populations(), self.params.min_N, N_A=NA)
                    continue
                sizes = ast.literal_eval('[' + params[1])
                self.add_period(
                    Period(
                        time=float(params[0]),
                        sizes_of_populations=sizes,
                        growth_types=None if params[2] == "None]" or self.params.only_sudden else ast.literal_eval(
                            '[' +
                            params[2]),
                        migration_rates= None if len(params) < 4 or self.params.no_mig else ast.literal_eval(
                            ('[' +
                             params[3]).replace(
                                '][',
                                '], ['))))
            if self.params.relative_params:
                self.normalize_by_Nref(real_Nref)

            self.check_time_and_correct_it()

    def construct_from_vector(self, vector, short=False):
        """The model can change its parameters to parameters from vector.

        N_A is missed. It is equal to 1.0. Structure of model doesn't
        change.
        """
        if short and not self.is_custom_model:
            vector = np.insert(vector, 0, 1.0)

        cur_index = 0
        vector = list(vector)
        if self.is_custom_model:
            self.popt = vector
            if not self.params.multinom:
                self.popt.append(1.0)
                self.normalize_by_Nref()
            self.has_changed()
            return

        for i, period in enumerate(self.periods):
            if period.is_first_period:
                    period.sizes_of_populations = [vector[cur_index]]
                    cur_index += 1
            elif period.is_split_of_population:
                all_sudden = (np.array(self.periods[i + 1].growth_types) == 0).all()
                if not all_sudden:
                    period.split_prop = vector[cur_index]
                    period.update(
                        self.periods[
                            i - 1].get_sizes_of_populations(),
                        self.params.min_N)
                    cur_index += 1
            else:
                period.sizes_of_populations = copy.deepcopy(vector[
                    cur_index:cur_index + period.number_of_populations])
                cur_index += period.number_of_populations
        for period in self.periods:
            if not period.is_split_of_population and not period.is_first_period:
                period.time = vector[cur_index]
                cur_index += 1
        for period in self.periods:
            if period.migration_rates is not None:
                if period.number_of_populations == 2:
                    period.migration_rates[0][1] = vector[cur_index]
                    period.migration_rates[1][0] = vector[cur_index + 1]
                    cur_index += 2
                else:
                    period.migration_rates[0][1] = vector[cur_index]
                    period.migration_rates[0][2] = vector[cur_index + 1]
                    period.migration_rates[1][0] = vector[cur_index + 2]
                    period.migration_rates[1][2] = vector[cur_index + 3]
                    period.migration_rates[2][0] = vector[cur_index + 4]
                    period.migration_rates[2][1] = vector[cur_index + 5]
                    cur_index += 6

        for i in xrange(self.number_of_periods):
            self.periods[i].check_params(self.params, self.get_N_A())
        self.has_changed()
        self.normalize_by_Nref()
        self.check_time_and_correct_it()

    def get_lower_and_upper_bounds(self):
        if self.lower_bound is not None:
            return self.lower_bound, self.upper_bound
        lower_bound = []
        upper_bound = []
        for i, period in enumerate(self.periods):
            n_pop = period.number_of_populations
            if period.is_first_period:
                continue
            if period.is_split_of_population:
                all_sudden = (np.array(self.periods[i + 1].growth_types) == 0).all()
                if not all_sudden:
                    bias = self.params.min_N
                    lower_bound.append(0.0 + bias)
                    upper_bound.append(1.0 - bias)
            else:
                lower_bound.extend(
                    [self.params.min_N for _ in xrange(n_pop)])
                upper_bound.extend(
                    [self.params.max_N for _ in xrange(n_pop)])
        for period in self.periods:
            if not period.is_first_period and not period.is_split_of_population:
                lower_bound.append(self.params.min_T)
                upper_bound.append(self.params.max_T)
        for period in self.periods:
            if period.migration_rates is not None:
                n_pop = period.number_of_populations
                for p1 in xrange(n_pop):
                    for p2 in xrange(n_pop):
                        if p1 == p2:
                            continue
                        lower_bound.append(self.params.min_M)
                        upper_bound.append(self.params.max_M)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        return self.lower_bound, self.upper_bound

    def run_local_search(self, name_of_search, filename, data_sample=None):
        if data_sample is None:
            data = self.params.input_data
        else:
            data = data_sample
        if self.get_number_of_params() == int(not self.params.multinom):
            return
        old_func_value = self.get_fitness_func_value(data_sample)
        self.normalize_by_Nref(1 / self.get_N_A())
        lower_bound, upper_bound = self.get_lower_and_upper_bounds()
        if self.is_custom_model:
            if self.params.multinom:
                p0 = self.popt
            else:
                p0 = self.popt[:-1]
        else:
            p0 = self.as_short_vector()
        if self.params.ls_verbose is not None:
            verbose = self.params.ls_verbose
        else:
            verbose = len(p0)
        func_kwargs = {
            'multinom': False,
            'output_file': filename,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'verbose': verbose,
            'flush_delay': self.params.ls_flush_delay,
            'epsilon': self.params.ls_epsilon,
            'gtol': self.params.ls_gtol,
            'maxiter': self.params.ls_maxiter}

        if name_of_search == 'optimize':
            if self.params.moments_scenario:
                optimize_func = moments.Inference.optimize
            else:
                optimize_func = dadi.Inference.optimize
        elif name_of_search == 'optimize_log':
            if self.params.moments_scenario:
                optimize_func = moments.Inference.optimize_log
            else:
                optimize_func = dadi.Inference.optimize_log
        elif name_of_search == 'optimize_powell':
            del func_kwargs['epsilon']
            del func_kwargs['gtol']
            # we have checked in options that moments is avaliable
            optimize_func = moments.Inference.optimize_powell
        elif name_of_search == 'optimize_lbfgsb':
            func_kwargs['pgtol'] = func_kwargs['gtol']
            del func_kwargs['gtol']
            if self.params.moments_scenario:
                optimize_func = moments.Inference.optimize_lbfgsbg
            else:
                optimize_func = dadi.Inference.optimize_lbfgsb
        elif name_of_search == 'optimize_log_lbfgsb':
            func_kwargs['pgtol'] = func_kwargs['gtol']
            del func_kwargs['gtol']
            if self.params.moments_scenario:
                optimize_func = moments.Inference.optimize_log_lbfgsbg
            else:
                optimize_func = dadi.Inference.optimize_log_lbfgsb
        else:  # name_of_search == 'optimize_log_fmin'
            del func_kwargs['epsilon']
            del func_kwargs['gtol']
            if self.params.moments_scenario:
                optimize_func = moments.Inference.optimize_log_fmin
            else:
                optimize_func = dadi.Inference.optimize_log_fmin

        if self.is_custom_model:
            func_kwargs['multinom'] = True
            if self.params.moments_scenario:
                func = self.params.model_func
            else:
                func = func_ex = dadi.Numerics.make_extrap_log_func(self.params.model_func)
        else:
            if self.params.moments_scenario:
                func = self.moments_code
            else:
                func = func_ex = dadi.Numerics.make_extrap_log_func(self.dadi_code)

        if self.params.moments_scenario:
            p_opt = optimize_func(p0, data,
                                  func, **func_kwargs)
        elif name_of_search == 'optimize_powell':
            # we need to run moments function with dadi's one, so we create
            # extra function abd put pts inside
            def my_func(*args, **kwargs):
                kwargs['pts'] = self.params.dadi_pts
                return func(*args, **kwargs)
            p_opt = optimize_func(
                p0, data, my_func, **func_kwargs)
        else:
            p_opt = optimize_func(p0, data,
                                  func, self.params.dadi_pts, **func_kwargs)
#        if not self.is_custom_model and not self.params.multinom:
#            x = [1.0]
#            x.extend(p_opt)
#            p_opt = np.array(x)
#            x = [1.0]
#            x.extend(p0)
#            p0 = np.array(x)

        if not np.isnan(p_opt).any() and not (p_opt < 0).any():
            self.construct_from_vector(p_opt, short=True)
            if self.get_fitness_func_value(data_sample) > old_func_value:
                self.construct_from_vector(p0, short=True)
        else:
            self.construct_from_vector(p0, short=True)

    def has_changed(self):
        """If the model has been changed, then old spectra and fitness function
        value aren't value."""
        self.sfs = None
        self.fitness_func_value = None
        self.aic_score = None
        self.claic_score = None

    def add_period(self, period):
        """Add one period to the list of periods."""
        self.number_of_periods += 1
        self.periods.append(period)

        # change current structure
        if period.is_split_of_population:
            self.cur_structure = np.append(self.cur_structure, 0)
            self.number_of_populations += 1

            # remember split positions
            if self.split_1_pos is None:
                self.split_1_pos = len(self.periods) - 1
            elif self.split_2_pos is None:
                self.split_2_pos = len(self.periods) - 1

        else:
            self.cur_structure[-1] += 1

        if not (period.is_first_period and self.params.multinom):
            for i in xrange(period.number_of_parameters):
                self.param_ids.append((self.number_of_periods - 1, i))
                self.number_of_changes = np.append(self.number_of_changes, 0.0)
        self.has_changed()

    def add_list_of_periods(self, list_of_periods):
        """Add several periods to the list of periods."""
        for period in list_of_periods:
            self.add_period(period)

    def get_structure(self):
        """Get a current structure from the list of periods."""
        if self.is_custom_model:
            return None
        self.cur_structure = [0]
        for p in self.periods:
            if p.is_split_of_population:
                self.cur_structure.append(0)
            else:
                self.cur_structure[-1] += 1
        self.number_of_populations = len(self.cur_structure)
        return self.cur_structure

    def get_total_time(self):
        """Calculate total time of all periods except the first period."""
        self.total_time = sum([x.time for x in self.periods])
        return self.total_time

    def get_number_of_params(self):
        if self.is_custom_model:
            return self.popt_len
        else:
            return sum([p.number_of_parameters for p in self.periods]
                   ) - self.params.multinom

    def get_param_ids(self):
        """Returns a list of pairs: (number of period, number of parameter of
        this period)."""
        if self.param_ids is not None:
            return self.param_ids
        if self.is_custom_model:
            self.param_ids = xrange(self.popt_len)
        else:
            self.param_ids = []
            for i in xrange(self.number_of_periods):
                for j in xrange(self.periods[i].number_of_parameters):
                    # ignore N_A if we are multinom
                    if self.periods[i].is_first_period and self.params.multinom:
                        continue
                    self.param_ids.append((i, j))
            self.number_of_changes = np.array(
                [0] * len(self.param_ids), dtype=float)
        return self.param_ids

    def mutate_one(self, mutation_rate, index_and_sign=None):
        """Mutate by one parameter.

        mutation_rate :  mean rate to change chosen parameter, default is in self.params
        index_and_sign :    pair (index, sign). if not None then index is number
                            of parameter to change by sign

        If index_and_sign is None then parameter is chosen randomly according to probability,
        which defines by number of changes of each parameters.
        Returns index and sign of parameter that was changed.
        """
        if mutation_rate is None:
            mutation_rate = self.params.mutation_rate

        self.number_of_changes = np.array(self.number_of_changes)
        if index_and_sign is None:
            # calculate probabilities and choose parameter
            p = max(self.number_of_changes) + 1 - self.number_of_changes
            p /= sum(p)
            i = np.random.choice(xrange(len(p)), p=p)
            sign = random.choice([-1, 1])
        else:
            i, sign = index_and_sign

        # generate change
        change = 0
        if self.params.distribution == 'uniform':
            while change == 0:
                change = np.random.uniform(0, 2 * mutation_rate)
        else:
            while change == 0:
                change = support.sample_from_truncated_normal(
                    mutation_rate, self.params.std, 0.0, 1.0)

        # mutate by parameter
        if self.is_custom_model:
            self.popt[i] *= 1 + sign * change
            if i != self.popt_len - 1:
                low_bound = self.lower_bound[i]
                upp_bound = self.upper_bound[i]
                if not self.params.multinom:
                    Nref = self.get_N_A()
                    low_bound = self.normalize_param_by_Nref(low_bound, Nref, self.params.p_ids[i])
                    upp_bound = self.normalize_param_by_Nref(upp_bound, Nref, self.params.p_ids[i])
                self.popt[i] = max(low_bound, self.popt[i])
                self.popt[i] = min(upp_bound, self.popt[i])
            
        else:
            period_index, param_index = self.get_param_ids()[i]
            self.periods[period_index].mutate_by_parameter(
                param_index, change, sign=sign, N_A=self.get_N_A(), bounds=self.params)

            # if we change time we need to check bounds of splits
            if not self.periods[period_index].is_first_period and not self.periods[
                    period_index].is_split_of_population and param_index == 0:
                self.check_time_and_correct_it()

            # if split we need to update sizes
            if self.periods[period_index].is_split_of_population:
                self.periods[period_index].update(
                    self.periods[
                        period_index -
                        1].get_sizes_of_populations(),
                    self.params.min_N,
                    N_A=self.get_N_A())

            # if we change sizes before split it should be not less than 2*min_N,
            # and we need to update split
            if period_index + 1 != self.number_of_periods and self.periods[
                    period_index + 1].is_split_of_population and param_index == self.periods[
                    period_index + 1].population_to_split + 1:
                self.periods[period_index].sizes_of_populations[
                    param_index -
                    1] = max(
                    self.periods[period_index].sizes_of_populations[
                        param_index -
                        1],
                    2 *
                    self.params.min_N *
                    self.get_N_A())
                self.periods[
                    period_index +
                    1].update(
                    self.periods[period_index].get_sizes_of_populations(),
                    self.params.min_N,
                    N_A=self.get_N_A())

            # if we change N_A we need to update split percent
            if not self.params.multinom and param_index == 0:
                N_A = self.get_N_A()
                k = 0
                for i, j in enumerate(self.get_structure()[:-1]):
                    k += j
                    self.periods[
                        i +
                        k].update(
                        self.periods[
                            i +
                            k -
                            1].get_sizes_of_populations(),
                        self.params.min_N,
                        N_A=N_A)

            # we could change sizes of populations in previous to split period, we
            # should update it in split
            if period_index != (self.number_of_periods - 1):
                if self.periods[period_index + 1].is_split_of_population:
                    self.periods[
                        period_index +
                        1].update(
                        self.periods[period_index].get_sizes_of_populations(),
                        self.params.min_N)

        # remember changes
        self.number_of_changes[i] += 1

        return i, sign

    def mutate(
            self,
            mutation_strength=None,
            mutation_rate=None,
            inds_and_signs=None):
        """Mutate by one parameter.

        mutation_strength : mean fraction of parameters to change
        mutation_rate :     mean rate to change chosen parameter
        inds_and_signs :    if None they will be chosed randomly

        Choose max(mutation_strength*number of params, 1) and mutate them by
        mutation_rate.
        Returns index and sign of parameters that were changed.
        """

        if mutation_strength is None:
            mutation_strength = self.params.mutation_strength

        if inds_and_signs is None:
            number_of_params_to_change = max(1, np.random.binomial(
                n=self.get_number_of_params(), p=mutation_strength))
        else:
            number_of_params_to_change = len(inds_and_signs)

        if inds_and_signs is None:
            # calculate probabilities and choose parameter
            p = (max(self.number_of_changes) + 1.0) - self.number_of_changes
            p /= sum(p)

            # choose parameters
            inds = np.random.choice(
                xrange(
                    len(p)),
                size=number_of_params_to_change,
                replace=False,
                p=p)
            signs = np.random.choice([-1, 1], size=number_of_params_to_change)

            inds_and_signs = [(inds[i], signs[i])
                              for i in xrange(number_of_params_to_change)]

        for i in xrange(number_of_params_to_change):
            self.mutate_one(mutation_rate, inds_and_signs[i])

        self.info += 'm'
        self.has_changed()

        if self.params.multinom or self.params.multinom_mutate:
            self.normalize_by_Nref()

        return inds_and_signs

    def increase_complexity(self, index=None):
        """Increase current structure's complexity.

        The structure can't be greater than params.final_structure.
        """
        # Find place to add new period
        final_structure = np.array(self.params.final_structure)
        diff = np.array(final_structure - self.cur_structure, dtype=float)
        if (final_structure <= self.cur_structure).all():
            return
        if index is not None:
            period_index_to_divide = index
            structure_index = 0
            while sum(self.cur_structure[:structure_index + 1]
                      ) + structure_index < period_index_to_divide:
                structure_index += 1
        else:
            structure_index = np.random.choice(
                xrange(len(self.cur_structure)), p=diff / sum(diff))
            number_of_periods_to_divide = self.cur_structure[structure_index]

            start_index_to_divide = sum(
                self.cur_structure[:structure_index]) + structure_index
            final_index_to_divide = start_index_to_divide + \
                self.cur_structure[structure_index]
            p = []
            
            for i in xrange(start_index_to_divide, final_index_to_divide):
                p.append(self.periods[i].time)

            if self.periods[
                    start_index_to_divide].is_first_period:
                if len(p) > 1:
                    p[0] = min(p[1:])
                else:
                    p[0] = 1.0

            p = np.array(p)
            p /= sum(p)

            period_index_to_divide = np.random.choice(
                range(start_index_to_divide, final_index_to_divide), p=p)

        # add new period by spliting chosen period to two periods
        total_time = self.get_total_time()
        if period_index_to_divide == 0:
            if total_time == 0:
                time = np.random.uniform(
                    self.params.min_T, self.params.max_T) * self.get_N_A()
            else:
                time = total_time / self.number_of_periods
            self.periods.insert(
                1,
                Period(
                    time=time,
                    sizes_of_populations=copy.deepcopy(
                        self.periods[0].get_sizes_of_populations()),
                    growth_types=None if self.params.only_sudden else [0]))
            period_index_to_divide = 0
        else:
            period_to_divide = self.periods[period_index_to_divide]
            period_to_divide.time /= 2.0
            pops_sizes = []
            pops_exp = []
            for p in xrange(period_to_divide.number_of_populations):
                if period_to_divide.growth_types[p] == 0:
                    pops_sizes.append(
                        period_to_divide.get_sizes_of_populations()[p])
                    pops_exp.append(0)
                elif period_to_divide.growth_types[p] == 1:
                    pops_sizes.append(
                        (self.periods[
                            period_index_to_divide -
                            1].get_sizes_of_populations()[p] +
                            period_to_divide.get_sizes_of_populations()[p]) /
                        2)
                    pops_exp.append(1)
                else:
                    pops_sizes.append(
                        (self.periods[
                            period_index_to_divide -
                            1].get_sizes_of_populations()[p] *
                            period_to_divide.get_sizes_of_populations()[p]) ** (0.5))
                    pops_exp.append(2)
            self.periods.insert(
                period_index_to_divide,
                Period(
                    time=period_to_divide.time,
                    sizes_of_populations=pops_sizes,
                    migration_rates=copy.deepcopy(
                        period_to_divide.migration_rates),
                    growth_types=None if self.params.only_sudden else pops_exp))
        self.number_of_periods += 1
        if self.split_1_pos is not None and period_index_to_divide < self.split_1_pos:
            if self.split_2_pos is not None:
                self.split_2_pos += 1
            self.split_1_pos += 1
        elif self.split_2_pos is not None and period_index_to_divide < self.split_2_pos:
            self.split_2_pos += 1

        self.cur_structure[structure_index] += 1

        ind = period_index_to_divide
        if ind == 0:
            ind = 1
        number_of_changes = list(self.number_of_changes)
        index_in_num_of_ch = sum([p.number_of_parameters for p in self.periods[
                                 :ind]]) - int(self.params.multinom)
        for i in xrange(self.periods[ind].number_of_parameters):
            number_of_changes.insert(index_in_num_of_ch + i, 0)

        self.lower_bound, self.upper_bound = None, None
        self.get_lower_and_upper_bounds()
        
        self.param_ids = None
        self.get_param_ids()
        self.number_of_changes = np.array(number_of_changes)

        self.get_aic_score()
        self.claic_score = None
        return period_index_to_divide

    def normalize_param_by_Nref(self, value, N_ref, identificator):
        if identificator == 'n' or identificator == 't':
            return value * N_ref
        elif identificator == 'm':
            return value / N_ref
        return value
    
    def normalize_by_Nref(self, N_ref=None, remove_fitness_func_value=True):
        """Change all parameter accordingly to new N_ref."""
        if self.is_custom_model and (self.params.multinom or self.params.p_ids is None):
            return
        
        if N_ref is None:
            if self.params.moments_scenario:
                opt_scale = moments.Inference.optimal_sfs_scaling(
                    self.get_sfs(), self.params.input_data)
            else:
                opt_scale = dadi.Inference.optimal_sfs_scaling(
                    self.get_sfs(), self.params.input_data)
            # checks for splits upper bounds:
            if self.params.split_1_lim is not None:
                opt_scale = min(opt_scale, self.params.split_1_lim /
                                sum([p.time for p in self.periods[self.split_1_pos:]]))
            if self.params.split_2_lim is not None:
                opt_scale = min(opt_scale, self.params.split_2_lim /
                                sum([p.time for p in self.periods[self.split_2_pos:]]))
            N_ref = opt_scale
        if self.is_custom_model:
            for i in xrange(self.popt_len - int(not self.params.multinom)):
                self.popt[i] = self.normalize_param_by_Nref(self.popt[i], 
                        N_ref, self.params.p_ids[i])
        else:
            if N_ref == 1.0:
                return
            for period_number, period in enumerate(self.periods):
                if period.is_split_of_population:
                    period.update(
                        self.periods[
                            period_number -
                            1].get_sizes_of_populations(),
                        self.params.min_N)
                    continue
                for i in xrange(period.number_of_populations):
                    period.sizes_of_populations[i] = self.normalize_param_by_Nref(
                            period.sizes_of_populations[i], N_ref, 'n')
                period.time = self.normalize_param_by_Nref(period.time, N_ref, 't')
                if period.migration_rates is not None:
                    for x in xrange(len(period.migration_rates)):
                        for y in xrange(len(period.migration_rates)):
                            if period.migration_rates[x][y] is not None:
                                period.migration_rates[x][y] = self.normalize_param_by_Nref(
                                        period.migration_rates[x][y], N_ref, 'm')
        if remove_fitness_func_value:
            if self.sfs is not None:
                self.sfs *= N_ref
            self.fitness_func_value = None

    def check_time_and_correct_it(self):
        if self.params.split_2_lim is not None:
            sum_of_time = sum(
                [p.time for p in self.periods[self.split_2_pos:]])
            if sum_of_time > self.params.split_2_lim:
                for p in self.periods[self.split_2_pos:]:
                    p.time *= self.params.split_2_lim / sum_of_time
                self.has_changed()
        if self.params.split_1_lim is not None:
            sum_of_time = sum(
                [p.time for p in self.periods[self.split_1_pos:]])
            if sum_of_time > self.params.split_1_lim:
                for p in self.periods[self.split_1_pos:]:
                    p.time *= self.params.split_1_lim / sum_of_time
                self.has_changed()
        #check split props
        def check_split_props(split_pos):
            if split_pos is not None:
                size_of_pop = self.periods[split_pos-1].sizes_of_populations[-1]
                min_size = 2 * self.params.min_N * self.get_N_A()
                if size_of_pop < min_size:
                    self.periods[split_pos-1].sizes_of_populations[-1] = min_size
                    self.periods[split_pos].update(self.periods[split_pos-1].get_sizes_of_populations(), self.params.min_N, self.get_N_A())
                    self.has_changed()
                elif size_of_pop > self.params.max_N * self.get_N_A() - min_size:
                    self.periods[split_pos-1].sizes_of_populations[-1] = self.params.max_N * self.get_N_A() - min_size
                    self.periods[split_pos].update(self.periods[split_pos-1].get_sizes_of_populations(), self.params.min_N, self.get_N_A())
                    self.has_changed()
                else:
                    if self.periods[split_pos].check_prop(self.params.min_N, 
                            self.get_N_A()):
                        self.has_changed()
        check_split_props(self.split_1_pos)
        check_split_props(self.split_2_pos)

    def generate_migration_rates(self, sizes_of_pops):
        """Generate migration rates from number_of_populations.

        sizes_of_pops regulate mig rates. Migration rate must be of
        order 1/N.
        """
        N_A = self.get_N_A()

        number_of_populations = len(sizes_of_pops)
        if number_of_populations == 1:
            return None
        if number_of_populations == 2:
            return [[None,
                     random.choice([0,
                                    1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[0]) / N_A, 'm')],
                    [random.choice([0,
                                    1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[1]) / N_A, 'm'),
                     None]]
        if number_of_populations == 3:
            return [[
                None,
                random.choice([0, 1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[0]) / N_A, 'm'),
                random.choice([0, 1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[0]) / N_A, 'm')
            ], [
                random.choice([0, 1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[1]) / N_A, 'm'), None,
                random.choice([0, 1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[1]) / N_A, 'm')
            ], [
                random.choice([0, 1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[2]) / N_A, 'm'),
                random.choice([0, 1]) * self.generate_random_value(self.params.min_M / N_A,
                                                            min(self.params.max_M,
                                                                self.params.max_M * sizes_of_pops[2]) / N_A, 'm'), None
            ]]

    def __str__(self, end='\n'):
        """String representation."""
        if self.params.relative_params:
            N_ref = self.get_N_A()
            self.normalize_by_Nref(1.0 / N_ref, remove_fitness_func_value=False)
        s = ''
        if self.is_custom_model:
            if self.params.multinom:
                s = '[' +', '.join(['%10s' % float_representation(x, 3) for x in self.popt]) + ']'
            else:
                s = '[' +', '.join(['%10s' % float_representation(x, 3) for x in self.popt[:-1]]) + ']'
        else:
            for period in self.periods:
                s += str(period)
        if self.params.relative_params:
            if self.is_custom_model:
                s += ' (Nref = ' + float_representation(N_ref) + ')'
            else:
                s += ' (N_A = ' + float_representation(N_ref) + ')'
            self.normalize_by_Nref(N_ref, remove_fitness_func_value=False)
        s += '\t' + self.info
        return s

    def get_N_A(self):
        if self.is_custom_model:
            if not self.params.multinom:
                return self.popt[-1]
            else:
                return 1.0
        return self.periods[0].get_sizes_of_populations()[0]

    def get_sfs(self):
        """Get SFS from model."""
        if self.sfs is None:
            if self.is_custom_model:
                if not self.params.multinom:
                    popt = self.popt[:-1]
                else:
                    popt = self.popt
                if not self.params.moments_scenario:
                    func = lambda ns, pts: self.params.model_func(popt, ns, pts)
                else:
                    func = lambda ns: self.params.model_func(popt, ns)
            else:
                if not self.params.moments_scenario:
                    func = self.dadi_code
                else:
                    func = self.moments_code

            if not self.params.moments_scenario:
                func_ex = dadi.Numerics.make_extrap_log_func(func)
                self.sfs = func_ex(self.params.ns, self.params.dadi_pts)

            else:
                for i in xrange(10):
                    try:
                        self.sfs = func(self.params.ns)
                        break
                    except RuntimeError as e:
                        if e.message == 'Factor is exactly singular':
                            if i == 9:
                                raise RuntimeError(
                                    'Factor is exactly singular')
                            self.dt_fac /= 2

            if self.is_custom_model and self.params.multinom:
                # self.optimal_theta is used on drawing with moments and only there!
                if self.params.moments_scenario:
                    self.optimal_theta = moments.Inference.optimal_sfs_scaling(self.sfs, self.params.input_data)           
                else:
                    self.optimal_theta = dadi.Inference.optimal_sfs_scaling(self.sfs, self.params.input_data)
                self.sfs *= self.optimal_theta

        return self.sfs

    def get_fitness_func_value(self, data_sample=None):
        """Calculate fitness function value for the model."""
        if data_sample is not None:
            self.has_changed()
        if self.fitness_func_value is None:
            sfs = self.get_sfs()
            if self.params.moments_scenario:
                ll_func = moments.Inference.ll
            else:
                ll_func = dadi.Inference.ll

            if data_sample is None:
                if (self.params.multinom and not self.is_custom_model):
                    self.normalize_by_Nref()

            if data_sample is None:
                data = self.params.input_data
            else:
                data = data_sample
            log_likelihood = ll_func(self.sfs, data)
            self.fitness_func_value = - log_likelihood
            self.aic_score = self.get_number_of_params() * 2 - 2 * log_likelihood

        return_value = self.fitness_func_value
        if data_sample is not None:
            self.has_changed()
        return return_value

    def get_aic_score(self, data_sample=None):
        """Calculate AIC score for the model."""
        if data_sample is not None:
            self.has_changed()
        if self.aic_score is None:
            self.get_fitness_func_value(data_sample)
        return_value = self.aic_score
        if data_sample is not None:
            self.has_changed()
        return return_value

    def get_claic_score(self, get_eps=False):
        """Calculate CLAIC score for the model."""
        if self.claic_score is None:
            if self.number_of_populations < 3:
                eps = 1e-14
            else:
                eps = 1e-10
            n_attemts = 0
            while eps <= 1e-2:
                try:
                    if self.params.moments_available and not self.is_custom_model:
                        self.claic_score = 2 * gadma.Inference.get_claic_component(
                                self.moments_code, self.as_short_vector(), 
                                self.params.input_data, pts=None, 
                                all_boot=None, eps=eps) + 2 * self.get_fitness_func_value()
                        break
                    else:
                        if self.is_custom_model:
                            self.claic_score = 2 *  gadma.Inference.get_claic_component(
                                    self.params.model_func, self.as_short_vector(), 
                                    self.params.input_data, 
                                    pts=None if self.params.moments_scenario else self.params.dadi_pts, 
                                    all_boot=None, eps=eps) + 2 * self.get_fitness_func_value()
                            break
                        else:
                            self.claic_score = 2 * gadma.Inference.get_claic_component(
                                    self.moments_code, self.as_short_vector(), 
                                    self.params.input_data, pts=self.params.dadi_pts, 
                                    all_boot=None, eps=eps) + 2 * self.get_fitness_func_value()
                            break
                except np.linalg.linalg.LinAlgError as e:
                    if e.message == 'Singular matrix':
                        eps *= 10
            self.claic_eps = eps
        if get_eps:
            return self.claic_score, self.claic_eps
        else:
            return self.claic_score

    def cross_with_other(self, other):
        """Crossover with other model for genetic algorithm."""
        if self.params.multinom or self.params.multinom_cross:
            self_N_A = self.get_N_A()
            self.normalize_by_Nref(1 / self_N_A, remove_fitness_func_value=False)
            other_N_A = other.get_N_A()
            other.normalize_by_Nref(1.0 / other_N_A, remove_fitness_func_value=False)

        if self.get_number_of_params() == 1:
            child = copy.deepcopy(self)
            child.info = 'c'
            return child
        # generate from whom parameters will be taken
        take_from_other = np.array(
            [random.random() for _ in xrange(self.get_number_of_params())]) > 0.5
        while not take_from_other.any():
            take_from_other = np.array(
                [random.random() for _ in xrange(self.get_number_of_params())]) > 0.5

        # create child
        child = copy.deepcopy(self)
        if self.is_custom_model:
            for i, x in enumerate(other.popt):
                if take_from_other[i]:
                    child.popt[i] = x
                if i != self.popt_len - 1:
                    low_bound = self.lower_bound[i]
                    upp_bound = self.upper_bound[i]
                    if not self.params.multinom:
                        Nref = self.get_N_A()
                        low_bound = self.normalize_param_by_Nref(low_bound, Nref, self.params.p_ids[i])
                        upp_bound = self.normalize_param_by_Nref(upp_bound, Nref, self.params.p_ids[i])
                    child.popt[i] = max(low_bound, child.popt[i])
                    child.popt[i] = min(upp_bound, child.popt[i])
        else:
            cur_ind = 0
            N_A = child.get_N_A()
            for i, period in enumerate(child.periods):
                if period.is_first_period:
                    if self.params.multinom:
                        continue
                    if take_from_other[cur_ind]:
                        period.get_sizes_of_populations()[0] = other.periods[
                            i].get_sizes_of_populations()[0]
                    cur_ind += 1
                    N_A = child.get_N_A()
                    continue

                if period.is_split_of_population:
                    if not self.params.only_sudden:
                        if take_from_other[cur_ind]:
                            period.split_prop = other.periods[i].split_prop
                        period.update(
                            child.periods[
                                i - 1].get_sizes_of_populations(),
                            self.params.min_N,
                            N_A=N_A)
                        cur_ind += 1
                    continue

                if take_from_other[cur_ind]:
                    period.time = other.periods[i].time
                period.time = min(period.time, self.params.max_T * N_A)
                period.time = max(period.time, self.params.min_T * N_A)
                cur_ind += 1

                for p in xrange(period.number_of_populations):
                    if take_from_other[cur_ind]:
                        period.sizes_of_populations[p] = other.periods[
                            i].sizes_of_populations[p]
                    if not self.params.only_sudden and take_from_other[cur_ind + period.number_of_populations]:
                        period.growth_types[p] = other.periods[i].growth_types[p]
                    period.sizes_of_populations[p] = min(
                        period.sizes_of_populations[p], self.params.max_N * N_A)
                    period.sizes_of_populations[p] = max(
                        period.sizes_of_populations[p], self.params.min_N * N_A)
                    not_last = i + 1 < self.number_of_periods
                    if not_last:
                        next_is_split = self.periods[i + 1].is_split_of_population
                        if next_is_split and p == self.periods[i + 1].population_to_split:
                            period.sizes_of_populations[p] = max(
                                period.sizes_of_populations[p], 2 * self.params.min_N * N_A)
                    cur_ind += 1
                if not self.params.only_sudden:
                    cur_ind += period.number_of_populations
                if period.migration_rates is None:
                    continue
                for p1 in xrange(period.number_of_populations):
                    for p2 in xrange(period.number_of_populations):
                        if p1 == p2:
                            continue
                        if take_from_other[cur_ind]:
                            period.migration_rates[p1][p2] = other.periods[
                                i].migration_rates[p1][p2]
                        period.migration_rates[p1][p2] = min(
                            period.migration_rates[p1][p2], self.params.max_M / N_A)
                        period.migration_rates[p1][p2] = max(
                            period.migration_rates[p1][p2], self.params.min_M / N_A)
                        cur_ind += 1
        child.number_of_changes = np.array(child.number_of_changes)
        child.number_of_changes[
            take_from_other] = other.number_of_changes[take_from_other]

        if self.params.multinom or self.params.multinom_cross:
            child.normalize_by_Nref()
            self.normalize_by_Nref(self_N_A, remove_fitness_func_value=False)
            other.normalize_by_Nref(other_N_A, remove_fitness_func_value=False)

        # now check time of splits
        self.check_time_and_correct_it()

        child.info = 'c'
        child.has_changed()
        return child

    def dadi_code(self, normalized_params, ns, pts=None):
        """Model function for DaDi to get SFS."""
        theta = self.params.theta if self.params.theta is not None else 1
        if pts is None:
            pts = ns
            ns = normalized_params
        else:
            params = normalized_params
            self.construct_from_vector(params, short=True)

        xx = dadi.Numerics.default_grid(pts)
        for pos, period in enumerate(self.periods):
            if period.is_first_period:
                phi = dadi.PhiManip.phi_1D(
                    xx,
                    theta0=theta,
                    nu=period.get_sizes_of_populations()[0])
            elif period.is_split_of_population:
                if period.number_of_populations == 2:
                    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
                else:  # if period.number_of_populations == 3:
                    if period.population_to_split == 0:
                        phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)
                    else:  # if period.population_to_split == 1
                        phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
            else:  # change population size
                growth_funcs = []
                for i in xrange(period.number_of_populations):
                    if period.time == 0 or period.growth_types[i] == 0:
                        growth_funcs.append(
                            period.get_sizes_of_populations()[i])
                    elif period.growth_types[i] == 1:
                        growth_funcs.append(
                            _linear_growth_func(
                                self.periods[
                                    pos - 1].get_sizes_of_populations()[i],
                                period.get_sizes_of_populations()[i],
                                period.time))
                    else:
                        growth_funcs.append(
                            _expon_growth_func(
                                self.periods[
                                    pos - 1].get_sizes_of_populations()[i],
                                period.get_sizes_of_populations()[i],
                                period.time))
                if period.number_of_populations == 1:
                    phi = dadi.Integration.one_pop(
                        phi,
                        xx,
                        nu=growth_funcs[0],
                        T=period.time,
                        theta0=theta)
                elif period.number_of_populations == 2:
                    phi = dadi.Integration.two_pops(
                        phi,
                        xx,
                        T=period.time,
                        nu1=growth_funcs[0],
                        nu2=growth_funcs[1],
                        m12=(0 if period.migration_rates is None else
                             period.migration_rates[0][1]),
                        m21=(0 if period.migration_rates is None else
                             period.migration_rates[1][0]),
                        theta0=theta)
                else:
                    phi = dadi.Integration.three_pops(
                        phi,
                        xx,
                        T=period.time,
                        nu1=growth_funcs[0],
                        nu2=growth_funcs[1],
                        nu3=growth_funcs[2],
                        m12=0 if period.migration_rates is None else
                        period.migration_rates[0][1],
                        m13=0 if period.migration_rates is None else
                        period.migration_rates[0][2],
                        m21=0 if period.migration_rates is None else
                        period.migration_rates[1][0],
                        m23=0 if period.migration_rates is None else
                        period.migration_rates[1][2],
                        m31=0 if period.migration_rates is None else
                        period.migration_rates[2][0],
                        m32=0 if period.migration_rates is None else
                        period.migration_rates[2][1],
                        theta0=theta)
        sfs = dadi.Spectrum.from_phi(
            phi, ns, [xx] * self.number_of_populations)
        return sfs

    def dadi_code_to_file(self, filename):
        """Print a python code to the file for a function to run in DaDi."""
        theta = self.params.theta if self.params.theta is not None else 1
        with open(filename, 'w') as output:
            output.write('#current best params = ' + str(self.as_full_vector()) +
                         '\n')
            output.write(
                'import dadi\nimport numpy as np\n\n')
            if self.is_custom_model:
                output.write('import imp\nfile_with_model_func = imp.load_source("file_with_model_func", "' + 
                        self.params.model_func_file + '")\n')
                output.write('generated_model = file_with_model_func.model_func\n')
            else:
                output.write('def generated_model(params, ns, pts):\n')
                Ns_len = 0
                Ts_len = 0
                for i, period in enumerate(self.periods):
                    if period.is_first_period:
                        if not self.params.multinom:
                            Ns_len += 1
                    elif period.is_split_of_population:
                        all_sudden = (np.array(self.periods[i+1].growth_types) == 0).all()
                        if not all_sudden:
                            Ns_len += 1
                    else:
                        Ns_len += period.number_of_populations
                        Ts_len += 1
                output.write('\tNs = params[:' + str(Ns_len) + ']\n')
                output.write('\tTs = params[' + str(Ns_len) + ':' +
                             str(Ns_len + Ts_len) + ']\n')
                output.write('\tMs = params[' + str(Ns_len + Ts_len) + ':]\n')

                ns_index = 0
                ts_index = 0
                ms_index = 0
                output.write('\ttheta1 = ' + str(theta) + '\n')
                output.write('\txx = dadi.Numerics.default_grid(pts)\n')
                for pos, period in enumerate(self.periods):
                    if self.params.only_sudden:
                        all_sudden_later = True
                    else:
                        all_sudden_later = True
                        if pos == len(self.periods) - 1:
                            all_sudden_later = True
                        else:
                            if self.periods[pos+1].is_split_of_population:
                                all_sudden_later = True
                            else:
                                next_not_split_period = self.periods[pos+1]
                                all_sudden_later = (np.array(next_not_split_period.growth_types) == 0).all()

                    if period.is_first_period:
                        output.write(
                            '\tphi = dadi.PhiManip.phi_1D(xx, theta0=theta1')
                        if self.params.multinom:
                            output.write(')\n')
                            if not all_sudden_later:
                                output.write('\tbefore = [1.0]\n')
                        else:
                            output.write(', nu=Ns[' + str(ns_index) + '])\n')
                            if not all_sudden_later:
                                output.write('\tbefore = [Ns[' + str(ns_index) + ']]\n')
                            ns_index += 1

                    elif period.is_split_of_population:
                        if period.number_of_populations == 2:
                            output.write(
                                '\tphi = dadi.PhiManip.phi_1D_to_2D(xx, phi)\n')
                            if not all_sudden_later:
                                output.write('\tbefore.append((1 - Ns[' +
                                             str(ns_index) + ']) * before[-1])\n')
                                output.write(
                                    '\tbefore[-2] *= Ns[' + str(ns_index) + ']\n')
                                ns_index += 1
                        if period.number_of_populations == 3:
                            if period.population_to_split == 0:
                                output.write(
                                    '\tphi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)\n')
                                if not all_sudden_later:
                                    output.write('\tbefore.append((1 - Ns[' +
                                                 str(ns_index) + ']) * before[-1])\n')
                                    output.write(
                                        '\tbefore[0] *= Ns[' + str(ns_index) + ']\n')
                                    ns_index += 1
                            else:
                                output.write(
                                    '\tphi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)\n')
                                if not all_sudden_later:
                                    output.write('\tbefore.append((1 - Ns[' +
                                                 str(ns_index) + ']) * before[-1])\n')
                                    output.write(
                                        '\tbefore[-2] *= Ns[' + str(ns_index) + ']\n')
                                    ns_index += 1

                    else:  # change population size
                        growth_funcs = []
                        output.write('\tT = Ts[' + str(ts_index) + ']\n')
                        ts_index += 1
                        output.write('\tafter = Ns[' + str(ns_index) + ':'
                                     + str(ns_index + period.number_of_populations) + ']\n')
                        ns_index += period.number_of_populations

                        for i in xrange(period.number_of_populations):
                            if period.time == 0 or period.growth_types[i] == 0:
                                if period.time == 0:
                                    output.write('\t# Time of period is equal to 0, so we ignore dynamics and\n'\
                                            '\t# linear and exponential change are considered to be sudden.\n')
                                growth_funcs.append('after[' + str(i) + ']')
                            elif period.growth_types[i] == 1:
                                growth_funcs.append(
                                    _linear_growth_func_str(
                                        'before[' + str(i) + ']',
                                        'after[' + str(i) + ']', 'T'))
                            else:
                                growth_funcs.append(
                                    _expon_growth_func_str(
                                        'before[' + str(i) + ']',
                                        'after[' + str(i) + ']',
                                        'T'))

                        if period.number_of_populations == 1:
                            output.write('\tgrowth_func = ' + growth_funcs[0] +
                                         '\n')
                            output.write(
                                '\tphi = dadi.Integration.one_pop(phi, xx, nu=growth_func,'
                                'T=T, theta0=theta1)\n'
                            )
                        elif period.number_of_populations == 2:
                            output.write('\tgrowth_func_1 = ' + growth_funcs[0] +
                                         '\n')
                            output.write('\tgrowth_func_2 = ' + growth_funcs[1] +
                                         '\n')
                            output.write(
                                '\tphi = dadi.Integration.two_pops(phi, xx,  T=T,'
                                'nu1=growth_func_1, nu2=growth_func_2, m12='
                                + ('0' if period.migration_rates is None else
                                   'Ms[' + str(ms_index) + ']') + ', m21=' +
                                ('0' if period.migration_rates is None else
                                 'Ms[' + str(ms_index + 1) + ']') +
                                ', theta0=theta1)\n')
                            ms_index += 2
                        else:
                            output.write('\tgrowth_func_1 = ' + growth_funcs[0] +
                                         '\n')
                            output.write('\tgrowth_func_2 = ' + growth_funcs[1] +
                                         '\n')
                            output.write('\tgrowth_func_3 = ' + growth_funcs[2] +
                                         '\n')
                            output.write(
                                '\tphi = dadi.Integration.three_pops(phi, xx,  T=T, '
                                'nu1=growth_func_1, nu2=growth_func_2, nu3=growth_func_3, m12='
                                + ('0' if period.migration_rates is None else
                                   'Ms[' + str(ms_index) + ']') + ', m13=' +
                                ('0' if period.migration_rates is None else
                                 'Ms[' + str(ms_index + 1) + ']') + ', m21=' +
                                ('0' if period.migration_rates is None else
                                 'Ms[' + str(ms_index + 2) + ']') + ', m23=' +
                                ('0' if period.migration_rates is None else
                                 'Ms[' + str(ms_index + 3) + ']') + ', m31=' +
                                ('0' if period.migration_rates is None else
                                 'Ms[' + str(ms_index + 4) + ']') + ', m32=' +
                                ('0' if period.migration_rates is None else
                                 'Ms[' + str(ms_index + 5) + ']') +
                                ', theta0=theta1)\n')
                            ms_index += 6
                        if not all_sudden_later or (pos+1 != len(self.periods) and self.periods[pos+1].is_split_of_population):
                            output.write('\tbefore = after\n')
                output.write('\tsfs = dadi.Spectrum.from_phi(phi, ns, [xx]*' + str(
                    self.params.number_of_populations) + ')\n\treturn sfs\n')

            # main
            # read data
            ext = os.path.splitext(self.params.input_file)[1][1:]
            if ext == 'fs':
                output.write("data = dadi.Spectrum.from_file('" + os.path.
                             abspath(self.params.input_file) + "')\n")
                # we need check if we change spectrum from file:
                real_spectrum, real_ns, real_labels  = support.read_fs_file(os.path.
                             abspath(self.params.input_file), proj=None, pop_labels=None)
                if not (real_ns == self.params.ns).all():
                    output.write("data = data.project(" + str(list(self.params.ns)) + ")\n")
                if not real_labels == self.params.pop_labels:
                    d = {x: i for i, x in enumerate(real_labels)}
                    d = [d[x] for x in self.params.pop_labels]
                    output.write("new_axis = " + str(d) + '\n')
                    output.write("data = np.transpose(data, new_axis)\n")
                    output.write("data.pop_ids = " + str(self.params.pop_labels) + '\n')
            else:
                output.write("dd = dadi.Misc.make_data_dict('" + os.path.
                             abspath(self.params.input_file) + "')\n")
                output.write('data = dadi.Spectrum.from_data_dict(dd, pop_ids=' +
                             str(self.params.pop_labels) +
                             ', projections=' +
                             str(list(self.params.ns)) +
                             ', polarized=' +
                             str(not self.params.input_data.folded) +
                             ')\n')
            output.write('\npopt = ' + str(self.as_full_vector()) + '\n')
            output.write('pts = ' + str(list(self.params.dadi_pts)) + '\n')
            output.write('ns = ' + str(list(self.params.ns)) + '\n')
            output.write(
                'func_ex = dadi.Numerics.make_extrap_log_func(generated_model)\n'
            )
            output.write('model =  func_ex(popt, ns, pts)\n')
            if self.params.multinom:
                output.write(
                    'll_model = dadi.Inference.ll_multinom(model, data)\n')
            else:
                if self.is_custom_model:
                    output.write('Nref = ' + self.popt[-1] + '# It is also optimized parameter by GADMA\n')
                    output.write(
                        'll_model = Nref * dadi.Inference.ll(model, data)\n')
                else:
                    output.write(
                        'll_model = dadi.Inference.ll(model, data)\n')

            output.write(
                "print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))\n")

            if self.params.multinom:
                output.write(
                        "\ntheta = dadi.Inference.optimal_sfs_scaling(model, data)\n")
                output.write(
                        "print('Optimal value of theta: {0}'.format(theta))\n")

    def moments_code(self, normalized_params, ns=None):
        """Model function for moments to get SFS."""
        theta = self.params.theta if self.params.theta is not None else 1
        if ns is None:
            ns = normalized_params
        else:
            if not normalized_params == []:
                params = normalized_params
                self.construct_from_vector(params, short=True)


        import moments
        pop_to_split_3 = None
        if self.number_of_populations == 3:
            for period in reversed(self.periods):
                if period.is_split_of_population:
                    pop_to_split_3 = period.population_to_split
                    break
        for pos, period in enumerate(self.periods):
            if period.is_first_period:
                sts = moments.LinearSystem_1D.steady_state_1D(
                    sum(ns),
                    N=period.get_sizes_of_populations()[0],
                    theta=theta)
                fs = moments.Spectrum(sts)
                cur_ns = [sum(ns)]
            elif period.is_split_of_population:
                if period.number_of_populations == 2:
                    if pop_to_split_3 is None or pop_to_split_3 == 1:
                        fs = moments.Manips.split_1D_to_2D(
                            fs, ns[0], sum(ns[1:]))
                        cur_ns = [ns[0], sum(ns[1:])]
                    else:
                        fs = moments.Manips.split_1D_to_2D(
                            fs, ns[0] + ns[2], ns[1])
                        cur_ns = [ns[0] + ns[2], ns[1]]
                else:
                    if period.population_to_split == 0:
                        fs = moments.Manips.split_2D_to_3D_1(fs, ns[0], ns[2])
                    else:
                        fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])
                    cur_ns = ns
            else:
                growth_funcs = []
                for p in xrange(period.number_of_populations):
                    if period.time == 0 or period.growth_types[p] == 0:
                        growth_funcs.append(
                            _sudden_growth_func(
                                period.get_sizes_of_populations()[p]))
                    elif period.growth_types[p] == 1:
                        growth_funcs.append(
                            _linear_growth_func(
                                self.periods[
                                    pos - 1].get_sizes_of_populations()[p],
                                period.get_sizes_of_populations()[p],
                                period.time))
                    else:
                        growth_funcs.append(
                            _expon_growth_func(
                                self.periods[
                                    pos - 1].get_sizes_of_populations()[p],
                                period.get_sizes_of_populations()[p],
                                period.time))

                def list_growth_funcs(t): return [f(t) for f in growth_funcs]

                if period.number_of_populations > 1 and period.migration_rates is not None:
                    m = np.array(period.migration_rates, dtype=float)
                    where_are_nans = np.isnan(m)
                    m[where_are_nans] = 0
                    fs.integrate(
                        Npop=list_growth_funcs,
                        tf=period.time,
                        m=m,
                        dt_fac=self.dt_fac,
                        theta=theta)
                else:
                    fs.integrate(
                        Npop=list_growth_funcs,
                        tf=period.time,
                        dt_fac=self.dt_fac,
                        theta=theta)
        return fs

    def moments_code_to_file(self, filename):
        """Print a python code to the file for a function to run in moments."""
        with open(filename, 'w') as output:
            output.write('#current best params = ' + str(self.as_full_vector()) +
                         '\n')
            output.write(
                'import matplotlib\nmatplotlib.use("Agg")\nimport moments\n'
                'import numpy as np\n\n')
                
            if self.is_custom_model:
                output.write('import imp\nfile_with_model_func = imp.load_source("file_with_model_func'
                        '= file_with_model_func", "' + self.params.model_func_file + '")\n')
                output.write('generated_model = file_with_model_func.model_func\n')
            else:
                pop_to_split_3 = None
                if self.number_of_populations == 3:
                    for period in reversed(self.periods):
                        if period.is_split_of_population:
                            pop_to_split_3 = period.population_to_split
                            break

                output.write('def generated_model(params, ns):\n')
                Ns_len = 0
                Ts_len = 0
                for i, period in enumerate(self.periods):
                    if period.is_first_period:
                        if not self.params.multinom:
                            Ns_len += 1
                    elif period.is_split_of_population:
                        all_sudden = (np.array(self.periods[i+1].growth_types) == 0).all()
                        if not all_sudden:
                            Ns_len += 1
                    else:
                        Ns_len += period.number_of_populations
                        Ts_len += 1
                output.write('\tNs = params[:' + str(Ns_len) + ']\n')
                output.write('\tTs = params[' + str(Ns_len) + ':' +
                             str(Ns_len + Ts_len) + ']\n')
                output.write('\tMs = params[' + str(Ns_len + Ts_len) + ':]\n')

                ns_index = 0
                ts_index = 0
                ms_index = 0
                output.write(
                    '\ttheta1 = ' + str(self.params.theta if self.params.theta is not None else 1) + '\n')
                for pos, period in enumerate(self.periods):
                    if self.params.only_sudden:
                        all_sudden_later = True
                    else:
                        all_sudden_later = True
                        if pos == len(self.periods) - 1:
                            all_sudden_later = True
                        else:
                            if self.periods[pos+1].is_split_of_population:
                                all_sudden_later = True
                            else:
                                next_not_split_period = self.periods[pos+1]
                                all_sudden_later = (np.array(next_not_split_period.growth_types) == 0).all()

                    if period.is_first_period:
                        output.write(
                            '\tsts = moments.LinearSystem_1D.steady_state_1D(sum(ns), theta=theta1')
                        if self.params.multinom:
                            output.write(')\n')
                            if not all_sudden_later:
                                output.write('\tbefore = [1.0]\n')
                        else:
                            output.write(', N=Ns[' + str(ns_index) + '])\n')
                            if not all_sudden_later:
                                output.write('\tbefore = [Ns[' + str(ns_index) + ']]\n')
                            ns_index += 1
                        output.write('\tfs = moments.Spectrum(sts)\n\n')

                    elif period.is_split_of_population:
                        if period.number_of_populations == 2:
                            if pop_to_split_3 is None or pop_to_split_3 == 1:
                                output.write(
                                    '\tfs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))\n\n')
                            else:
                                output.write(
                                    '\tfs = moments.Manips.split_1D_to_2D(fs, ns[0] + ns[2], ns[1])\n\n')
                            if not all_sudden_later:
                                output.write('\tbefore.append((1 - Ns[' +
                                             str(ns_index) + ']) * before[-1])\n')
                                output.write(
                                    '\tbefore[-2] *= Ns[' + str(ns_index) + ']\n')
                                ns_index += 1
                        if period.number_of_populations == 3:
                            if period.population_to_split == 0:
                                output.write(
                                    '\tfs = moments.Manips.split_2D_to_3D_1(fs, ns[0], ns[2])\n')
                                if not all_sudden_later:
                                    output.write('\tbefore.append((1 - Ns[' +
                                                 str(ns_index) + ']) * before[-1])\n')
                                    output.write(
                                        '\tbefore[0] *= Ns[' + str(ns_index) + ']\n')
                                    ns_index += 1
                            else:
                                output.write(
                                    '\tfs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])\n')
                                if not all_sudden_later:
                                    output.write('\tbefore.append((1 - Ns[' +
                                                 str(ns_index) + ']) * before[-1])\n')
                                    output.write(
                                        '\tbefore[-2] *= Ns[' + str(ns_index) + ']\n')
                                    ns_index += 1

                    else:  # change population size
                        output.write('\tT = Ts[' + str(ts_index) + ']\n')
                        ts_index += 1
                        output.write('\tafter = Ns[' + str(ns_index) + ':'
                                     + str(ns_index + period.number_of_populations) + ']\n')
                        ns_index += period.number_of_populations

                        growth_funcs = '['
                        for i in xrange(period.number_of_populations):
                            if period.time == 0 or period.growth_types[i] == 0:
                                if period.time == 0:
                                    output.write('\t# Time of period is equal to 0, so we ignore dynamics and\n'\
                                            '\t# linear and exponential change are considered to be sudden.\n')
                                growth_funcs += 'lambda t: after[' + str(i) + '], '
                            elif period.growth_types[i] == 1:
                                growth_funcs += _linear_growth_func_str(
                                    'before[' + str(i) + ']',
                                    'after[' + str(i) + ']', 'T') + ', '
                            else:
                                growth_funcs += _expon_growth_func_str(
                                    'before[' + str(i) + ']',
                                    'after[' + str(i) + ']', 'T') + ', '
                        growth_funcs = growth_funcs[:-2]
                        growth_funcs += ']'
                        output.write('\tgrowth_funcs = ' + growth_funcs + '\n')
                        output.write(
                            '\tlist_growth_funcs = lambda t: [ f(t) for f in growth_funcs]\n'
                        )

                        if period.number_of_populations == 1:
                            output.write(
                                '\tfs.integrate(Npop=list_growth_funcs, tf=T, dt_fac=' +
                                str(self.dt_fac) + ', theta=theta1)\n\n'
                            )
                        elif period.number_of_populations == 2:
                            output.write('\tm = np.array([[0, Ms[' +
                                         str(ms_index) + ']],[Ms[' +
                                         str(ms_index + 1) + '], 0]])\n')
                            output.write(
                                '\tfs.integrate(Npop=list_growth_funcs, tf=T, m=m, dt_fac=' +
                                str(self.dt_fac) + ', theta=theta1)\n\n'
                            )
                            ms_index += 2
                        else:
                            output.write('\tm = np.array([[0.0, Ms[' +
                                         str(ms_index) + '], Ms[' +
                                         str(ms_index + 1) + ']],[Ms[' +
                                         str(ms_index + 2) + '], 0.0, Ms[' +
                                         str(ms_index + 3) + ']], [Ms[' +
                                         str(ms_index + 4) + '], Ms[' +
                                         str(ms_index + 5) + '], 0.0]])\n')
                            output.write(
                                '\tfs.integrate(Npop=list_growth_funcs, tf=T, m=m, dt_fac=' +
                                str(self.dt_fac) + ', theta=theta1)\n\n'
                            )
                            ms_index += 6
                        if not all_sudden_later or (pos+1 != len(self.periods) and self.periods[pos+1].is_split_of_population):
                            output.write('\tbefore = after\n')
                output.write('\treturn fs\n')

            # main
            # read data
            ext = os.path.splitext(self.params.input_file)[1][1:]
            if ext == 'fs':
                output.write("data = moments.Spectrum.from_file('" + os.path.
                             abspath(self.params.input_file) + "')\n")
                # we need check if we change spectrum from file:
                real_spectrum, real_ns, real_labels  = support.read_fs_file(os.path.
                             abspath(self.params.input_file), proj=None, pop_labels=None)
                if not (real_ns == self.params.ns).all():
                    output.write("data = data.project(" + str(list(self.params.ns)) + ")\n")
                if not real_labels == self.params.pop_labels:
                    d = {x: i for i, x in enumerate(real_labels)}
                    d = [d[x] for x in self.params.pop_labels]
                    output.write("new_axis = " + str(d) + '\n')
                    output.write("data = np.transpose(data, new_axis)\n")
                    output.write("data.pop_ids = " + str(self.params.pop_labels) + '\n')

            else:
                output.write("dd = moments.Misc.make_data_dict('" + os.path.
                             abspath(self.params.input_file) + "')\n")
                output.write('data = moments.Spectrum.from_data_dict(dd, pop_ids=' +
                             str(self.params.pop_labels) +
                             ', projections=' +
                             str(list(self.params.ns)) +
                             ', polarized=' +
                             str(not self.params.input_data.folded) +
                             ')\n')
            output.write('\npopt = ' + str(self.as_full_vector()) + '\n')
            output.write('ns = ' + str(list(self.params.ns)) + '\n')
            output.write('model = generated_model(popt, ns)\n')
            if self.params.multinom:
                output.write(
                    'll_model = moments.Inference.ll_multinom(model, data)\n')
            else:
                if self.is_custom_model:
                    output.write('Nref = ' + self.popt[-1] + '# It is also optimized parameter by GADMA\n')
                    output.write(
                        'll_model = Nref * moments.Inference.ll(model, data)\n')
                else:
                    output.write(
                        'll_model = moments.Inference.ll(model, data)\n')

            output.write(
                "print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))\n")
            
            if self.params.multinom:
                output.write(
                        "\ntheta = moments.Inference.optimal_sfs_scaling(model, data)\n")
                output.write(
                        "print('Optimal value of theta: {0}'.format(theta))\n")


            size_of_first_pop = self.get_N_A()
            self.normalize_by_Nref(1 / size_of_first_pop, remove_fitness_func_value=False)
            output.write(
                '#now we need to norm vector of params so that N_A is 1:\n'
                'popt_norm = ' + str(
                    self.as_full_vector()) + '\n')
            self.normalize_by_Nref(size_of_first_pop, remove_fitness_func_value=False)

            draw_scale = self.params.theta is not None
            if self.params.multinom and draw_scale:
                output.write("theta0 = " + str(self.params.theta) + '\n')
                nref = 'int(theta / theta0)'
            else:
                if draw_scale:
                    nref = str(size_of_first_pop)
                else:
                    nref = 'None'

            if self.params.gen_time is not None:
                if self.params.gen_time_units == 1:
                    gen_time_units = 'Years'
                elif self.params.gen_time_units == 1000:
                    gen_time_units = 'Thousand years'
                else:
                    gen_time_units = 'Genetic units'
            else:
                gen_time_units = 'Genetic units'
                
            output.write("print('Drawing model to model_from_GADMA.png')\n"
                         'model = moments.ModelPlot.generate_model(generated_model, popt_norm, ns)\n'
                         'moments.ModelPlot.plot_model(model, \n'
                         "\tsave_file='model_from_GADMA.png',\n"
                         "\tfig_title='Demographic model from GADMA',\n"
                         '\tpop_labels=' +
                         str(self.params.pop_labels) +
                         ',\n'
                         '\tnref=' + nref +
                         ',\n'
                         '\tdraw_scale=' + str(draw_scale) + ',\n'
                         '\tgen_time=' + 
                         str(float(self.params.gen_time) /
                             self.params.gen_time_units if self.params.gen_time is not None else 1.0) +
                         ',\n'
                         '\tgen_time_units=' + "'" + gen_time_units + "'" +
                         ',\n'
                         '\treverse_timeline=True)')

    def as_full_vector(self):
        """Vector representaton of the model in order to put it in python code
        for DaDi."""
        if self.params.multinom:
            Nref = self.get_N_A()
            self.normalize_by_Nref(1.0 / Nref, remove_fitness_func_value=False)
        if self.is_custom_model:
            if self.params.multinom:
                return self.popt
            else:
                return self.popt[:-1]

        vector = []
        for i, period in enumerate(self.periods):
            if period.is_first_period:
                if not self.params.multinom:
                    vector.append(period.get_sizes_of_populations()[0])
            elif period.is_split_of_population:
                all_sudden = (np.array(self.periods[i + 1].growth_types) == 0).all()
                if not all_sudden:
                    vector.append(period.split_prop)
            else:
                vector.extend(period.get_sizes_of_populations())
        for period in self.periods:
            if not period.is_first_period and not period.is_split_of_population:
                vector.append(period.time)
        for period in self.periods:
            if period.migration_rates is not None:
                if period.number_of_populations == 2:
                    vector.append(period.migration_rates[0][1])
                    vector.append(period.migration_rates[1][0])
                else:
                    vector.append(period.migration_rates[0][1])
                    vector.append(period.migration_rates[0][2])
                    vector.append(period.migration_rates[1][0])
                    vector.append(period.migration_rates[1][2])
                    vector.append(period.migration_rates[2][0])
                    vector.append(period.migration_rates[2][1])

        if self.params.multinom:
#            vector = vector[1:]
            self.normalize_by_Nref(Nref, remove_fitness_func_value=False)
        return vector

    def as_short_vector(self):
        if self.params.multinom:
            p0 = self.as_full_vector()
        else:
            Nref = self.get_N_A()
            self.normalize_by_Nref(1.0 / Nref, remove_fitness_func_value=False)
            p0 = self.as_full_vector()[1:]
            self.normalize_by_Nref(Nref, remove_fitness_func_value=False)
        return p0

    def get_bounds_to_dadi(self):
        """Bound for parameters of the demographic model in order to run DaDi
        optimization."""
        vector_1 = []
        vector_2 = []
        max_size = 0
        for period in self.periods:
            max_size = max(max_size, max(period.get_sizes_of_populations()))
        for period in self.periods:
            if period.is_first_period:
                vector_1.append(1)
                vector_2.append(10 * max_size)
            elif period.is_split_of_population:
                vector_1.append(0)
                vector_2.append(1)
            else:
                vector_1.extend([1] * period.number_of_populations)
                vector_2.extend([10 * max_size] * period.number_of_populations)

        for period in self.periods:
            if not period.is_first_period and not period.is_split_of_population:
                vector_1.append(1)
                vector_2.append(10 * self.get_total_time())
        for period in self.periods:
            if period.migration_rates is not None:
                if period.number_of_populations == 2:
                    vector_1.extend([0] * 2)
                    vector_2.extend([1] * 2)
                else:
                    vector_1.extend([0] * 6)
                    vector_2.extend([1] * 6)

        return vector_1, vector_2

    def print_code_to_file(self, filename):
        """Main function to print python code of the model."""
        if self.params.moments_scenario:
            self.moments_code_to_file(filename)
        else:
            self.dadi_code_to_file(filename)

#    @support.timeout(240)
    def draw_with_moments(self, save_file, title):
        """Function to draw model with moments' tool."""
        import moments
        size_of_first_pop = self.get_N_A()
        self.normalize_by_Nref(1 / size_of_first_pop, remove_fitness_func_value=False)
        if self.is_custom_model:
            model = moments.ModelPlot.generate_model(self.params.model_func, self.as_full_vector(),
                                                 self.params.ns)
        else:
            model = moments.ModelPlot.generate_model(self.moments_code, [],
                                                 self.params.ns)
        draw_scale = self.params.theta is not None
        if self.params.multinom and self.is_custom_model and draw_scale:
            nref = int(self.optimal_theta / self.params.theta)
        else:
            if draw_scale:
                nref = size_of_first_pop
            else:
                nref = None

        if self.params.gen_time is None or (self.params.gen_time_units not in [1, 1000]):
            gen_time = 1.0
            units = 'Genetic units'
        else:
            gen_time = self.params.gen_time / self.params.gen_time_units
            units = 'Years' if self.params.gen_time_units == 1 else 'Thousand years'
        moments.ModelPlot.plot_model(
            model,
            save_file=save_file,
            fig_title=title,
            pop_labels=self.params.pop_labels,
            nref=nref,
            draw_scale=draw_scale,
            gen_time=gen_time,
            gen_time_units=units,
            reverse_timeline=True)
        self.normalize_by_Nref(size_of_first_pop, remove_fitness_func_value=False)

    def draw(self, filename, title):
        """Draw big picture of the model and data."""
        import matplotlib
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings(
            'ignore', category=matplotlib.cbook.MatplotlibDeprecationWarning)

        if self.params.moments_available:
            import moments

        pos = filename.rfind('.')
        if pos == -1:
            pos = len(filename)
        filename_2 = filename[:pos] + '_model' + filename[pos:]
        filename_1 = filename[:pos] + '_sfs' + filename[pos:]

        if self.is_custom_model and not self.params.moments_scenario:
            out = filename
        else:
            if self.params.pil_available:
                buf1 = io.BytesIO()
                out = buf1
            else:
                out = filename_1

        # Draw sfs

        matplotlib.rcParams.update({'font.size': 18})
        fig = plt.figure(1, figsize=(13.8, 10.8))
        if (self.number_of_populations == 1):
            if self.params.moments_scenario:
                moments.Plotting.plot_1d_comp_Poisson(
                    self.get_sfs(), self.params.input_data, show=False)
            else:
                dadi.Plotting.plot_1d_comp_Poisson(
                    self.get_sfs(), self.params.input_data, show=False)
        elif (self.number_of_populations == 2):
            if self.params.moments_scenario:
                moments.Plotting.plot_2d_comp_Poisson(
                    self.get_sfs(), self.params.input_data, vmin=1, show=False)
            else:
                dadi.Plotting.plot_2d_comp_Poisson(
                    self.get_sfs(), self.params.input_data, vmin=1, show=False)
        elif (self.number_of_populations == 3):
            if self.params.moments_scenario:
                moments.Plotting.plot_3d_comp_Poisson(
                    self.get_sfs(), self.params.input_data, vmin=1, show=False)
            else:
                dadi.Plotting.plot_3d_comp_Poisson(
                    self.get_sfs(), self.params.input_data, vmin=1, show=False)
        plt.savefig(out)

        if self.is_custom_model and not self.params.moments_scenario or not self.params.moments_available:
            return

        # Draw model
        if self.params.pil_available:
            buf2 = io.BytesIO()
            out = buf2
        else:
            out = filename_2

        try:
            size_of_first_pop = self.get_N_A()
            self.draw_with_moments(out, title)
        except support.TimeoutError:
            support.warning("Can't draw model to " + filename +
                            ' (Timeout for drawing)')
            if self.get_N_A() == 1.0:
                self.normalize_by_Nref(size_of_first_pop, remove_fitness_func_value=False)
            return
        except RuntimeError as e:
            if e.message == 'Factor is exactly singular':
                support.warning("Can't draw model to " + filename +
                                ' (Scipy version less than 0.19.0)')
                if self.get_N_A() == 1.0:
                    self.normalize_by_Nref(size_of_first_pop, remove_fitness_func_value=False)
                return
            else:
                raise e
    
        if not self.params.pil_available:
            return

        import PIL
        buf1.seek(0)
        buf2.seek(0)

        plt.close('all')
        
        if self.is_custom_model and not self.params.moments_scenario:
            img1 = PIL.Image.new('RGB', (0, 0))
        else:
            img1 = PIL.Image.open(buf2)
        img2 = PIL.Image.open(buf1)

        weight = img1.size[0] + img2.size[0]
        height = max(img1.size[1], img2.size[1])

        new_img = PIL.Image.new('RGB', (weight, height))

        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.size[0], 0))

        new_img.save(filename)


# Some helpfull function of growth types
def _sudden_growth_func(size):
    return (lambda t: size)


def _linear_growth_func(before, after, time):
    return (lambda t: before + (after - before) * (t / time))


def _expon_growth_func(before, after, time):
    return (lambda t: before * ((after / before) ** (t / time)))


def _linear_growth_func_str(before, after, time):
    return 'lambda t: ' + str(before) + ' + ' + '(' + str(after) + ' - ' + str(
        before) + ') * (t / ' + str(time) + ')'


def _expon_growth_func_str(before, after, time):
    return 'lambda t: ' + str(before) + ' * ' + '(' + str(after) + ' / ' + str(
        before) + ') ** (t / ' + str(time) + ')'


# Functions for float representations
def float_representation(f, prec=2):
    if f is None:
        return str(None)
    if f == 0:
        return '0.0'
    s = ('%.' + str(prec) + 'f') % f
    if s == ('%.' + str(prec) + 'f') % (0.0):
        return migration_representation(f, prec)
    return s


def migration_representation(m, prec=2):
    if m is None:
        return str(None)
    return ('%.' + str(prec) + 'e') % m


def list_float_representation(l, prec=2):
    s = '['
    for x in l:
        s += float_representation(x, prec) + ', '
    s = s[:-2]
    s += ']'
    return s


def migr_float_representation(m, prec=2):
    s = '['
    for x in m:
        s += '['
        for y in x:
            s += migration_representation(y, prec) + ', '
        s = s[:-2]
        s += ']'
    s += ']'
    return s
