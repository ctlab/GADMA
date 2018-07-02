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
            self.number_of_parameters = 1 + self.number_of_populations * 2 + (
                0 if self.migration_rates is None else
                (2 if self.number_of_populations == 2 else 6))

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
            self.get_sizes_of_populations()[0] = max(
                self.get_sizes_of_populations()[0], bounds.min_N * N_A)

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

        elif param_index <= 2 * self.number_of_populations:
            self.growth_types[param_index -
                              self.number_of_populations - 1] += sign
            self.growth_types[param_index -
                              self.number_of_populations - 1] %= 3
        else:
            i = param_index - 2 * self.number_of_populations - 1

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
                    bounds.min_M / N_A, min(
                        bounds.max_M / size_of_pop, 10.0 / size_of_pop))
            else:
                self.migration_rates[x][y]
                self.migration_rates[x][
                    y] *= 1 + sign * change
                self.migration_rates[x][y] = max(
                    self.migration_rates[x][y], bounds.min_M / size_of_pop)
                self.migration_rates[x][y] = min(
                    self.migration_rates[x][y], bounds.min_M / size_of_pop)
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
            return '[ ' + float_representation(100 * self.split_prop) + '%, ' + \
                list_float_representation(
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
        self.number_of_parameters = 1
        self.number_of_changes = [0]

    def get_sizes_of_populations(self):
        sizes_of_pops = copy.deepcopy(self.sizes_of_pops_before)
        sizes_of_pops.append(sizes_of_pops[-1])
        sizes_of_pops[-1] *= (1 - self.split_prop)
        sizes_of_pops[-2] *= self.split_prop
        return sizes_of_pops

    def check_prop(self, min_N, N_A):
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
        self.split_prop *= 1 + sign * change
        self.check_prop(min_N, N_A)
        
    def update(self, new_size_of_population_before_split, min_N=None, N_A=None):
        """Update sizes of populations of previous period."""
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
            restore_string=None):
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
        '''
        self.not_r = False
        self.params = params

        self.number_of_periods = 0
        self.periods = []
        self.info = ''

        self.cur_structure = None
        self.get_structure()

        self.dt_fac = 0.1  # for moments
        self.sfs = None
        self.fitness_func_value = None
        self.bic_score = None

        self.split_1_pos = None
        self.split_2_pos = None

        self.param_ids = None
        self.number_of_changes = np.array(
            [], dtype=float)
        self.get_param_ids()

        if restore_string is not None:
            self.from_string(restore_string)

        if random and restore_string is None:
            if structure is None:
                self.init_random_model(self.params.initial_structure)
            else:
                self.init_random_model(structure)

    def init_random_model(self, structure):
        """Generate random model of a given structure."""
        if structure is None:
            return

        # add first "const" period
        if self.params.random_N_A:
            N_A = np.random.uniform(self.params.min_N, self.params.max_N)
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
                    time=np.random.uniform(
                        self.params.min_T, self.params.max_T),
                    sizes_of_populations=[np.random.uniform(
                        self.params.min_N, self.params.max_N)],
                    growth_types=[random.choice([0, 1, 2])]))
        for num_of_pops, number_of_periods in enumerate(structure[1:]):
            num_of_pops += 2
            self.add_period(
                Split(
                    split_prop=np.random.uniform(
                        self.params.min_N, 1 - self.params.min_N),
                    population_to_split=num_of_pops - 2,
                    sizes_of_populations_before_split=self.periods[-1]
                    .get_sizes_of_populations()))
            for i in xrange(number_of_periods):
                sizes_of_pops = [
                    np.random.uniform(self.params.min_N, self.params.max_N)
                    for x in xrange(num_of_pops)
                ]
                self.add_period(
                    Period(
                        time=np.random.uniform(
                            self.params.min_T, self.params.max_T),
                        sizes_of_populations=sizes_of_pops,
                        growth_types=[
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
        NA = None
        string = string.strip()
        if string.endswith(')'):
            last_occ = string.rfind('(')
            NA = float(string[last_occ + len('N_A = '): -1])
            string = string[:last_occ - 1]
        periods = string.split(' ][ ')
        periods[0] = periods[0][2:]
        periods[-1] = periods[-1][:-2]

        for p in periods:
            params = p.split(', [')

            if len(self.periods) == 0 and NA is None:
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
                        proportion,
                        -1,
                        self.periods[-1].get_sizes_of_populations()))
                self.periods[-1].update(self.periods[-2].get_sizes_of_populations(), self.params.min_N, N_A=NA)
                continue
            sizes = ast.literal_eval('[' + params[1])
            self.add_period(
                Period(
                    float(
                        params[0]),
                    sizes,
                    ast.literal_eval(
                        '[' +
                        params[2]),
                    None if len(params) < 4 or self.params.no_mig else ast.literal_eval(
                        ('[' +
                         params[3]).replace(
                            '][',
                            '], ['))))
        if self.params.relative_params:
            self.normalize_by_Nref(NA)

        self.check_time_and_correct_it()

    def construct_from_vector(self, vector):
        """The model can change its parameters to parameters from vector.

        N_A is missed. It is equal to 1.0. Structure of model doesn't
        change.
        """
        cur_index = 0
        vector = list(vector)
        for i, period in enumerate(self.periods):
            if period.is_first_period:
                period.sizes_of_populations = [1.0]
            elif period.is_split_of_population:
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

        self.has_changed()
        self.normalize_by_Nref()
        self.check_time_and_correct_it()

    def get_lower_and_upper_bounds(self):
        lower_bound = []
        upper_bound = []
        for period in self.periods:
            n_pop = period.number_of_populations
            if period.is_first_period:
                continue
            if period.is_split_of_population:
                lower_bound.append(0.0)
                upper_bound.append(1.0)
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
                for p1 in xrange(n_pop):
                    for p2 in xrange(n_pop):
                        if p1 == p2:
                            continue
                        lower_bound.append(self.params.min_M)
                        upper_bound.append(self.params.max_M)
        return lower_bound, upper_bound

    def run_local_search(self, name_of_search, filename, data_sample=None):
        if data_sample is None:
            data = self.params.input_data
        else:
            data = data_sample
        if len(self.get_param_ids()) == int(not self.params.multinom):
            return
        old_func_value = self.get_fitness_func_value(data_sample)
        self.normalize_by_Nref(1 / self.get_N_A())
        lower_bound, upper_bound = self.get_lower_and_upper_bounds()
        p0 = self.as_vector()[1:]
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

        if self.params.moments_scenario:
            p_opt = optimize_func(p0, data,
                                  self.moments_code, **func_kwargs)
        elif name_of_search == 'optimize_powell':
            # we need to run moments function with dadi's one, so we create
            # extra function abd put pts inside
            func_ex = dadi.Numerics.make_extrap_log_func(self.dadi_code)

            def my_func(*args, **kwargs):
                kwargs['pts'] = self.params.dadi_pts
                return func_ex(*args, **kwargs)
            p_opt = optimize_func(
                p0, data, my_func, **func_kwargs)
        else:
            func_ex = dadi.Numerics.make_extrap_log_func(self.dadi_code)
            p_opt = optimize_func(p0, data,
                                  func_ex, self.params.dadi_pts, **func_kwargs)

        if not np.isnan(p_opt).any() and not (p_opt < 0).any() and self.get_fitness_func_value(data_sample) < old_func_value:
            self.construct_from_vector(p_opt)
        else:
            self.construct_from_vector(p0)

    def has_changed(self):
        """If the model has been changed, then old spectra and fitness function
        value aren't value."""
        self.sfs = None
        self.fitness_func_value = None
        self.bic_score = None

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
                self.number_of_changes = np.append(self.number_of_changes, 0)
        self.has_changed()

    def add_list_of_periods(self, list_of_periods):
        """Add several periods to the list of periods."""
        for period in list_of_periods:
            self.add_period(period)

    def get_structure(self):
        """Get a current structure from the list of periods."""
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
        return sum([p.number_of_parameters for p in self.periods]
                   ) - (1 if self.params.multinom else 0)

    def get_param_ids(self):
        """Returns a list of pairs: (number of period, number of parameter of
        this period)."""
        if self.param_ids is not None:
            return self.param_ids
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

        # we could change sizes of populations in previous to slit period, we
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
                n=len(self.get_param_ids()), p=mutation_strength))
        else:
            number_of_params_to_change = len(inds_and_signs)

        if inds_and_signs is None:
            # calculate probabilities and choose parameter
            p = max(self.number_of_changes) + 1 - self.number_of_changes
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
                    start_index_to_divide].is_first_period and len(p) > 1:
                p[0] = min(p[1:])
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
                    growth_types=[0]))
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
                    growth_types=pops_exp))
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

        self.param_ids = None
        self.get_param_ids()
        self.number_of_changes = np.array(number_of_changes)

        self.get_bic_score()
        return period_index_to_divide

    def normalize_by_Nref(self, N_ref=None):
        """Change all parameter accordingly to new N_ref."""
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

        for period_number, period in enumerate(self.periods):
            if period.is_split_of_population:
                period.update(
                    self.periods[
                        period_number -
                        1].get_sizes_of_populations(),
                    self.params.min_N)
                continue
            for i in xrange(period.number_of_populations):
                period.sizes_of_populations[i] *= N_ref
            period.time *= N_ref
            if period.migration_rates is not None:
                for x in xrange(len(period.migration_rates)):
                    for y in xrange(len(period.migration_rates)):
                        if period.migration_rates[x][y] is not None:
                            period.migration_rates[x][y] /= N_ref
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
                                    1]) * np.random.uniform(self.params.min_M,
                                                            min(self.params.max_M,
                                                                10.0 / sizes_of_pops[0])) / N_A],
                    [random.choice([0,
                                    1]) * np.random.uniform(self.params.min_M,
                                                            min(self.params.max_M,
                                                                10.0 / sizes_of_pops[1])) / N_A,
                     None]]
        if number_of_populations == 3:
            return [[
                None,
                random.choice([0, 1]) * np.random.uniform(self.params.min_M,
                                                          min(self.params.max_M, 10.0 / sizes_of_pops[0])) / N_A,
                random.choice([0, 1]) * np.random.uniform(self.params.min_M,
                                                          min(self.params.max_M, 10.0 / sizes_of_pops[0])) / N_A
            ], [
                random.choice([0, 1]) * np.random.uniform(self.params.min_M,
                                                          min(self.params.max_M, 10.0 / sizes_of_pops[1])) / N_A, None,
                random.choice([0, 1]) * np.random.uniform(self.params.min_M,
                                                          min(self.params.max_M, 10.0 / sizes_of_pops[1])) / N_A
            ], [
                random.choice([0, 1]) * np.random.uniform(self.params.min_M,
                                                          min(self.params.max_M, 10.0 / sizes_of_pops[2])) / N_A,
                random.choice([0, 1]) * np.random.uniform(self.params.min_M,
                                                          min(self.params.max_M, 10.0 / sizes_of_pops[2])) / N_A, None
            ]]

    def __str__(self, end='\n'):
        """String representation."""
        if self.params.relative_params and not self.not_r:
            N_A = self.get_N_A()
            self.normalize_by_Nref(1.0 / N_A)
        s = ''
        for period in self.periods:
            s += str(period)
        if self.params.relative_params and not self.not_r:
            s += ' (N_A = ' + float_representation(N_A) + ')'
            self.normalize_by_Nref(N_A)
        s += '\t' + self.info
        return s

    def get_N_A(self):
        return self.periods[0].get_sizes_of_populations()[0]

    def get_sfs(self):
        """Get SFS from model."""
        if self.sfs is None:
            if not self.params.moments_scenario:
                func_ex = dadi.Numerics.make_extrap_log_func(self.dadi_code)
                self.sfs = func_ex(self.params.ns, self.params.dadi_pts)
            else:
                for i in xrange(10):
                    try:
                        self.sfs = self.moments_code(self.params.ns)
                        break
                    except RuntimeError as e:
                        if e.message == 'Factor is exactly singular':
                            if i == 9:
                                raise RuntimeError(
                                    'Factor is exactly singular')
                            self.dt_fac /= 2
        return self.sfs

    def get_fitness_func_value(self, data_sample=None):
        """Calculate fitness function value for the model."""
        if data_sample is not None:
            self.has_changed()
        if self.fitness_func_value is None:
            sfs = self.get_sfs()
            if (self.params.multinom) and data_sample is None:
                self.normalize_by_Nref()

            if data_sample is None:
                data = self.params.input_data
            else:
                data = data_sample
            if not self.params.moments_scenario:
                log_likelihood = dadi.Inference.ll(self.sfs, data)
                self.fitness_func_value = - log_likelihood
            else:
                log_likelihood = moments.Inference.ll(self.sfs, data)
                self.fitness_func_value = - log_likelihood
            self.bic_score = math.log(
                np.prod(
                    np.array(
                        self.params.ns))) * self.get_number_of_params() - 2 * log_likelihood

        return_value = self.fitness_func_value
        if data_sample is not None:
            self.has_changed()
        return return_value

    def get_bic_score(self, data_sample=None):
        """Calculate BIC score for the model."""
        if data_sample is not None:
            self.has_changed()
        if self.bic_score is None:
            self.get_fitness_func_value(data_sample)
        return_value = self.bic_score
        if data_sample is not None:
            self.has_changed()
        return return_value

    def cross_with_other(self, other):
        """Crossover with other model for genetic algorithm."""
        if self.params.multinom or self.params.multinom_cross:
            self_N_A = self.get_N_A()
            self.normalize_by_Nref(1 / self_N_A)
            other_N_A = other.get_N_A()
            other.normalize_by_Nref(1.0 / other_N_A)

        if self.number_of_periods == 1:
            child = copy.deepcopy(self)
            child.info = 'c'
            return child
        # generate from whoom parameters will be taken
        take_from_other = np.array(
            [random.random() for _ in xrange(self.get_number_of_params())]) > 0.5
        while not take_from_other.any():
            take_from_other = np.array(
                [random.random() for _ in xrange(self.get_number_of_params())]) > 0.5

        # create child
        child = copy.deepcopy(self)
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
                if take_from_other[cur_ind + period.number_of_populations]:
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
            self.normalize_by_Nref(self_N_A)
            other.normalize_by_Nref(other_N_A)

        # now check time of splits
        self.check_time_and_correct_it()

        child.info = 'c'
        child.has_changed()
        return child

    def dadi_code(self, params, ns, pts=None):
        """Model function for DaDi to get SFS."""
        theta = self.params.theta if self.params.theta is not None else 1
        if pts is None:
            pts = ns
            ns = params
        else:
            self.construct_from_vector(params)

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
                    if period.growth_types[i] == 0:
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
            output.write('#current best params = ' + str(self.as_vector()) +
                         '\n')
            output.write(
                'import dadi\ndef generated_model(params, ns, pts):\n')
            Ns_len = 0
            Ts_len = 0
            for period in self.periods:
                if period.is_first_period:
                    Ns_len += 1
                elif period.is_split_of_population:
                    Ns_len += 1
                else:
                    Ns_len += period.number_of_populations
                    Ts_len += 1
            output.write('\tNs = params[:' + str(Ns_len) + ']\n')
            output.write('\tTs = params[' + str(Ns_len) + ':' +
                         str(Ns_len + Ts_len) + ']\n')
            output.write('\tMs = params[' + str(Ns_len + Ts_len) + ':]\n')

            cur_index = 0
            extra = 0
            mig_index = Ns_len + Ts_len
            output.write('\ttheta1 = ' + str(theta) + '\n')
            output.write('\txx = dadi.Numerics.default_grid(pts)\n')
            for pos, period in enumerate(self.periods):
                if period.is_first_period:
                    output.write(
                        '\tphi = dadi.PhiManip.phi_1D(xx, theta0=theta1, nu=Ns['
                        + str(cur_index) + '])\n')
                    output.write('\tbefore = [Ns[' + str(cur_index) + ']]\n')
                    extra += 1
                elif period.is_split_of_population:
                    if period.number_of_populations == 2:
                        output.write(
                            '\tphi = dadi.PhiManip.phi_1D_to_2D(xx, phi)\n')
                        output.write('\tbefore.append((1 - Ns[' +
                                     str(cur_index + extra) + ']) * before[-1])\n')
                        output.write(
                            '\tbefore[-2] *= Ns[' + str(cur_index + extra) + ']\n')
                    if period.number_of_populations == 3:
                        if period.population_to_split == 0:
                            output.write(
                                '\tphi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)\n')
                            output.write('\tbefore.append((1 - Ns[' +
                                         str(cur_index + extra) + ']) * before[0])\n')
                            output.write(
                                '\tbefore[0] *= Ns[' + str(cur_index + extra) + ']\n')
                        else:
                            output.write(
                                '\tphi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)\n')
                            output.write('\tbefore.append((1 - Ns[' +
                                         str(cur_index + extra) + ']) * before[-1])\n')
                            output.write(
                                '\tbefore[-2] *= Ns[' + str(cur_index + extra) + ']\n')
                    extra += 1

                else:  # change population size
                    growth_funcs = []
                    output.write('\tT = Ts[' + str(cur_index) + ']\n')
                    output.write('\tafter = Ns[' + str(cur_index + extra) + ':'
                                 + str(cur_index + extra +
                                       period.number_of_populations) + ']\n')
                    extra += period.number_of_populations - 1

                    for i in xrange(period.number_of_populations):
                        if period.growth_types[i] == 0:
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
                               'params[' + str(mig_index) + ']') + ', m21=' +
                            ('0' if period.migration_rates is None else
                             'params[' + str(mig_index + 1) + ']') +
                            ', theta0=theta1)\n')
                        mig_index += 2
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
                               'params[' + str(mig_index) + ']') + ', m13=' +
                            ('0' if period.migration_rates is None else
                             'params[' + str(mig_index + 1) + ']') + ', m21=' +
                            ('0' if period.migration_rates is None else
                             'params[' + str(mig_index + 2) + ']') + ', m23=' +
                            ('0' if period.migration_rates is None else
                             'params[' + str(mig_index + 3) + ']') + ', m31=' +
                            ('0' if period.migration_rates is None else
                             'params[' + str(mig_index + 4) + ']') + ', m32=' +
                            ('0' if period.migration_rates is None else
                             'params[' + str(mig_index + 5) + ']') +
                            ', theta0=theta1)\n')
                        mig_index += 6
                    cur_index += 1
                    output.write('\tbefore = after\n')
            output.write('\tsfs = dadi.Spectrum.from_phi(phi, ns, [xx]*' + str(
                self.params.number_of_populations) + ')\n\treturn sfs\n')

            # main
            # read data
            ext = os.path.splitext(self.params.input_file)[1][1:]
            if ext == 'fs':
                output.write("data = dadi.Spectrum.from_file('" + os.path.
                             abspath(self.params.input_file) + "')\n")
            else:
                output.write("dd = dadi.Misc.make_data_dict('" + os.path.
                             abspath(self.params.input_file) + "')\n")
                output.write('data = dadi.Spectrum.from_data_dict(dd, pop_ids=' +
                             str(self.params.pop_labels) +
                             ', projections=' +
                             str(list(self.params.ns)) +
                             ', polarized=' +
                             str(self.params.input_data.folded) +
                             ')\n')
            output.write('\npopt = ' + str(self.as_vector()) + '\n')
            output.write('pts = ' + str(list(self.params.dadi_pts)) + '\n')
            output.write('ns = ' + str(list(self.params.ns)) + '\n')
            output.write(
                'func_ex = dadi.Numerics.make_extrap_log_func(generated_model)\n'
            )
            output.write('model =  func_ex(popt, ns, pts)\n')
            output.write(
                'll_model = dadi.Inference.ll(model, data)\n'
                'll_true = dadi.Inference.ll(data, data)\n'
                "print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))\n")

    def moments_code(self, params, ns=None):
        """Model function for moments to get SFS."""
        theta = self.params.theta if self.params.theta is not None else 1
        if ns is None:
            ns = params
        else:
            if not params == []:
                self.construct_from_vector(params)

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
                    if period.growth_types[p] == 0:
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
            pop_to_split_3 = None
            if self.number_of_populations == 3:
                for period in reversed(self.periods):
                    if period.is_split_of_population:
                        pop_to_split_3 = period.population_to_split
                        break

            output.write('#current best params = ' + str(self.as_vector()) +
                         '\n')
            output.write(
                'import matplotlib\nmatplotlib.use("Agg")\nimport moments\n'
                'import numpy as np\ndef generated_model(params, ns):\n'
            )
            Ns_len = 0
            Ts_len = 0
            for period in self.periods:
                if period.is_first_period:
                    Ns_len += 1
                elif period.is_split_of_population:
                    Ns_len += 1
                else:
                    Ns_len += period.number_of_populations
                    Ts_len += 1
            output.write('\tNs = params[:' + str(Ns_len) + ']\n')
            output.write('\tTs = params[' + str(Ns_len) + ':' +
                         str(Ns_len + Ts_len) + ']\n')
            output.write('\tMs = params[' + str(Ns_len + Ts_len) + ':]\n')

            cur_index = 0
            extra = 0
            mig_index = Ns_len + Ts_len
            output.write(
                '\ttheta1 = ' + str(self.params.theta if self.params.theta is not None else 1) + '\n')
            for pos, period in enumerate(self.periods):
                if period.is_first_period:
                    output.write(
                        '\tsts = moments.LinearSystem_1D.steady_state_1D(sum(ns), theta=theta1, N=Ns[' +
                        str(cur_index) +
                        '])\n')
                    output.write('\tfs = moments.Spectrum(sts)\n\n')
                    output.write('\tbefore = [Ns[' + str(cur_index) + ']]\n')
                    extra += 1
                elif period.is_split_of_population:
                    if period.number_of_populations == 2:
                        if pop_to_split_3 is None or pop_to_split_3 == 1:
                            output.write(
                                '\tfs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))\n\n'
                            )
                        else:
                            output.write(
                                '\tfs = moments.Manips.split_1D_to_2D(fs, ns[0] + ns[2], ns[1])\n\n'
                            )
                        output.write('\tbefore.append((1 - Ns[' + str(cur_index + extra) +
                                     ']) * before[-1])\n')
                        output.write(
                            '\tbefore[-2] *= Ns[' + str(cur_index + extra) + ']\n')
                    else:  # if period.number_of_populations == 3:
                        if period.population_to_split == 0:
                            output.write(
                                '\tfs = moments.Manips.split_2D_to_3D_1(fs, ns[0], ns[2])\n\n'
                            )
                            output.write('\tbefore.append((1 - Ns[' + str(cur_index + extra) +
                                         ']) * before[0])\n')
                            output.write(
                                '\tbefore[0] *= Ns[' + str(cur_index + extra) + ']\n')
                        else:  # if period.population_to_split == 1
                            output.write(
                                '\tfs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])\n\n'
                            )
                            output.write('\tbefore.append((1 - Ns[' + str(cur_index + extra) +
                                         ']) * before[-1])\n')
                            output.write(
                                '\tbefore[-2] *= Ns[' + str(cur_index + extra) + ']\n')
                    extra += 1

                else:  # change population size
                    output.write('\tT = Ts[' + str(cur_index) + ']\n')
                    output.write('\tafter = Ns[' + str(cur_index + extra) + ':'
                                 + str(cur_index + extra +
                                       period.number_of_populations) + ']\n')
                    extra += period.number_of_populations - 1
                    growth_funcs = '['
                    for i in xrange(period.number_of_populations):
                        if period.growth_types[i] == 0:
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
                        output.write('\tm = np.array([[0, params[' +
                                     str(mig_index) + ']],[params[' +
                                     str(mig_index + 1) + '], 0]])\n')
                        output.write(
                            '\tfs.integrate(Npop=list_growth_funcs, tf=T, m=m, dt_fac=' +
                            str(self.dt_fac) + ', theta=theta1)\n\n'
                        )
                        mig_index += 2
                    else:
                        output.write('\tm = np.array([[0.0, params[' +
                                     str(mig_index) + '], params[' +
                                     str(mig_index + 1) + ']],[params[' +
                                     str(mig_index + 2) + '], 0.0, params[' +
                                     str(mig_index + 3) + ']], [params[' +
                                     str(mig_index + 4) + '], params[' +
                                     str(mig_index + 5) + '], 0.0]])\n')
                        output.write(
                            '\tfs.integrate(Npop=list_growth_funcs, tf=T, m=m, dt_fac=' +
                            str(self.dt_fac) + ', theta=theta1)\n\n'
                        )
                        mig_index += 6
                    cur_index += 1
                    output.write('\tbefore = after\n')
            output.write('\treturn fs\n')

            # main
            # read data
            ext = os.path.splitext(self.params.input_file)[1][1:]
            if ext == 'fs':
                output.write("data = moments.Spectrum.from_file('" + os.path.
                             abspath(self.params.input_file) + "')\n")
            else:
                output.write("dd = moments.Misc.make_data_dict('" + os.path.
                             abspath(self.params.input_file) + "')\n")
                output.write('data = moments.Spectrum.from_data_dict(dd, pop_ids=' +
                             str(self.params.pop_labels) +
                             ', projections=' +
                             str(list(self.params.ns)) +
                             ', polarized=' +
                             str(self.params.input_data.folded) +
                             ')\n')
            output.write('\npopt = ' + str(self.as_vector()) + '\n')
            output.write('ns = ' + str(list(self.params.ns)) + '\n')
            output.write('model = generated_model(popt, ns)\n')
            output.write(
                'll_model = moments.Inference.ll(model, data)\n'
                'll_true = moments.Inference.ll(data, data)\n'
                "print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))\n")

            size_of_first_pop = self.get_N_A()
            self.normalize_by_Nref(1 / size_of_first_pop)
            output.write(
                '#now we need to norm vector of params so that first value is 1:\n'
                'popt_norm = ' + str(
                    self.as_vector()) + '\n')
            self.normalize_by_Nref(size_of_first_pop)
            output.write("print('Drawing model to model_from_GADMA.png')\n"
                         'model = moments.ModelPlot.generate_model(generated_model, popt_norm, ns)\n'
                         'moments.ModelPlot.plot_model(model, \n'
                         "\tsave_file='model_from_GADMA.png',\n"
                         "\tfig_title='Demographic model from GADMA',\n"
                         '\tpop_labels=' +
                         str(self.params.pop_labels) +
                         ',\n'
                         '\tnref=' +
                         str(int(size_of_first_pop)) +
                         ',\n'
                         '\tdraw_scale=True,\n'
                         '\tgen_time=' +
                         str(float(self.params.gen_time) /
                             1000 if self.params.gen_time is not None else 1.0) +
                         ',\n'
                         '\tgen_time_units=' +
                         ('"Thousand years"' if self.params.gen_time is not None else '"Genetic units"') +
                         ',\n'
                         '\treverse_timeline=True)')

    def as_vector(self):
        """Vector representaton of the model in order to put it in python code
        for DaDi."""
        vector = []
        for period in self.periods:
            if period.is_first_period:
                vector.append(period.get_sizes_of_populations()[0])
            elif period.is_split_of_population:
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

        return vector

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
        self.normalize_by_Nref(1 / size_of_first_pop)
        model = moments.ModelPlot.generate_model(self.moments_code, [],
                                                 self.params.ns)
        moments.ModelPlot.plot_model(
            model,
            save_file=save_file,
            fig_title=title,
            pop_labels=self.params.pop_labels,
            nref=int(size_of_first_pop),
            draw_scale=self.params.theta is not None,
            gen_time=float(self.params.gen_time) /
            1000 if self.params.gen_time is not None else 1.0,
            gen_time_units=(
                'Thousand years' if self.params.gen_time is not None else 'Genetic units'),
            reverse_timeline=True)
        self.normalize_by_Nref(size_of_first_pop)

    def draw(self, filename, title):
        """Draw big picture of the model and data."""
        import moments
        import matplotlib
        import matplotlib.pyplot as plt
        import PIL
        import warnings
        warnings.filterwarnings(
            'ignore', category=matplotlib.cbook.MatplotlibDeprecationWarning)
        buf1 = io.BytesIO()

        try:
            size_of_first_pop = self.get_N_A()
            self.draw_with_moments(buf1, title)
        except support.TimeoutError:
            support.warning("Can't draw model to " + filename +
                            ' (Timeout for drawing)')
            self.normalize_by_Nref(size_of_first_pop)
            return
        except RuntimeError as e:
            if e.message == 'Factor is exactly singular':
                support.warning("Can't draw model to " + filename +
                                ' (Scipy version less than 0.19.0)')
                self.normalize_by_Nref(size_of_first_pop)
                return
            else:
                raise e

        buf1.seek(0)

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
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        plt.close('all')

        img1 = PIL.Image.open(buf1)
        img2 = PIL.Image.open(buf2)

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
def float_representation(f):
    if f is None:
        return str(None)
    if f == 0:
        return '0.0'
    s = '%.2f' % f
    if s == '0.00':
        return migration_representation(f)
    return s


def migration_representation(m):
    if m is None:
        return str(None)
    return '%.2e' % m


def list_float_representation(l):
    s = '['
    for x in l:
        s += float_representation(x) + ', '
    s = s[:-2]
    s += ']'
    return s


def migr_float_representation(m):
    s = '['
    for x in m:
        s += '['
        for y in x:
            s += migration_representation(y) + ', '
        s = s[:-2]
        s += ']'
    s += ']'
    return s
