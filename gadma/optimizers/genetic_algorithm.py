from .optimizer import ConstrainedOptimizer, Optimizer
from .global_optimizer import GlobalOptimizer, register_global_optimizer
from ..utils import sort_by_other_list, choose_by_weight
from ..utils import trunc_normal_3_sigma_rule, DiscreteVariable,\
                    WeightedMetaArray, get_correct_dtype
from ..utils import update_by_one_fifth_rule

import numpy as np
import copy
import time


class GeneticAlgorithm(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Genetic Algorithm.

    :param gen_size: Size of generation of genetic algorithm. That is number
                     of individuals/solutions on each step of GA.
    :type gen_size: int
    :param n_elitism: Number of best models from previous generation in GA
                      that will be taken to new iteration.
    :type n_elitism: int
    :param p_mutation: probability of mutation in one generation of GA.
    :type p_mutation: float
    :param p_crossover: probability of crossover in one generation of GA.
    :type p_crossover: float
    :param p_random: Probability of random generated individual in one
                     generation of GA.
    :type p_random: float
    :param mut_rate: Initial mean mutation rate.
    :type mut_rate: float
    :param mut_strength: initial mutation "strength" - mean fraction of
                         model parameters that will be mutated.
    :type mut_strength: float
    :param const_mut_rate: constant to change mutation rate according to
                           one-fifth algorithm. Check GADMA paper for more
                           information.
    :type const_mut_rate: float
    :param eps: const for model's log likelihood compare.
                Model is better if its log likelihood is greater than
                log likelihood of another model by epsilon.
    :type eps: float
    :param n_stuck_gen: Number of iterations for GA stopping: GA stops when
                        it can't improve model during n_stuck_gen generations.
    :type n_stuck_gen: int
    :param selection_type: Type of selection operator in GA. Could be:
                           * 'roulette_wheel'
                           * 'rank'
                           See help(GeneticAlgorithm.selection) for more
                           information.
    :type selection_type: str
    :param selection_random: If True then number of mutants and crossover's
                             offsprings in new generation will be binomial
                             random variable.
    :type selection_type: bool
    :param mutation_type: Type of mutation operator in GA. Could be:
                          * 'uniform'
                          * 'resample'
                          * 'gaussian'
                          See help(GeneticAlgorithm.mutation) for more
                          information.
    :type mutation_type: str
    :param one_fifth_rule: If True then one fifth rule is used in mutation.
    :type one_fifth_rule: bool
    :param crossover_type: Type of crossover operator in GA. Could be:
                           * 'k-point'
                           * 'uniform'
                           See help(GeneticAlgorithm.crossover) for more
                           information.
    :type crossover_type: str
    :param crossover_k: k for 'k-point' crossover type.
    :type crossover_k: int
    :param random_type: Type of random generation of new offsprings. Could be:
                        * 'uniform'
                        * 'resample'
                        * 'custom'
                        See help(GlobalOptimizer.randomize) for more
                        information.
    :type random_type: str
    :param custom_rand_gen: Random generator for 'custom' random_type.
                            Provide generator from variables:
                            custom_rand_gen(variables) = values
    :type custom_rand_gen: func
    :param log_transform: If True then logarithm will be used incide for
                          parameters.
    :type log_transform: bool
    :param maximize: If True then optimization will maximize function.
    :type maximize: bool
    """
    def __init__(self, gen_size=10, n_elitism=2,
                 p_mutation=0.3, p_crossover=0.3, p_random=0.2,
                 mut_strength=0.2, const_mut_strength=1.1,
                 mut_rate=0.2, const_mut_rate=1.2, mut_attempts=2,
                 eps=1e-2, n_stuck_gen=100,
                 selection_type='roulette_wheel', selection_random=False,
                 mutation_type='gaussian', one_fifth_rule=True,
                 crossover_type='uniform', crossover_k=None,
                 random_type='resample', custom_rand_gen=None,
                 log_transform=False, maximize=False):
        # Simple checks
        assert isinstance(gen_size, int)
        assert isinstance(n_elitism, int)
        assert (n_elitism < gen_size)
        assert (p_mutation >= 0 and p_mutation <= 1)
        assert (p_crossover >= 0 and p_crossover <= 1)
        assert (p_random >= 0 and p_random <= 1)
        assert (mut_rate >= 0 and mut_rate <= 1)
        assert (mut_strength >= 0 and mut_strength <= 1)
        assert (const_mut_rate >= 1 and const_mut_rate <= 2)
        assert (const_mut_strength >= 1 and const_mut_strength <= 2)

        self.gen_size = gen_size
        self.n_elitism = n_elitism
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.p_random = p_random

        self.mut_rate = mut_rate
        self.mut_strength = mut_strength
        self.cur_mut_rate = mut_rate
        self.cur_mut_strength = mut_strength
        self.const_mut_rate = const_mut_rate
        self.const_mut_strength = const_mut_strength
        self.mut_attempts = mut_attempts
        self.eps = eps
        self.n_stuck_gen = n_stuck_gen

        self.selection_type = selection_type
        self.selection_random = selection_random
        self.crossover_type = crossover_type
        self.crossover_k = crossover_k
        self.mutation_type = mutation_type
        self.one_fifth_rule = one_fifth_rule
        super(GeneticAlgorithm, self).__init__(random_type, custom_rand_gen,
                                               log_transform, maximize)

    def randomize(self, variables, random_type='resample',
                  custom_rand_gen=None):
        x = super(GeneticAlgorithm, self).randomize(variables, random_type,
                                                    custom_rand_gen)
        x = WeightedMetaArray(x)
        x.metadata = 'r'
        return x

    def mutation_by_ind(self, x, variables, index, mutation_type='gaussian',
                        one_fifth_rule=True):
        """
        Mutation of `x` in index `index`. For more information see
        :meth:`mutation`.
        """
        x_mut = WeightedMetaArray(x)
        var = variables[index]
        # Start mutation procedure
        # 1. Uniform type
        if mutation_type == 'uniform':
            # Case 1 Discrete variable
            if isinstance(var, DiscreteVariable):
                while x[index] == x_mut[index] and len(var.domain) > 1:
                    x_mut[index] = np.random.choice(var.domain)
            # Case 2 Continous variable
            else:
                x_mut[index] = np.random.uniform(var.domain[0], var.domain[1])
        # 2. Resample type
        elif mutation_type == 'resample':
            while x[index] == x_mut[index] and len(var.domain) > 1:
                x_mut[index] = var.resample()
        # 3. Gaussian type
        elif mutation_type == 'gaussian':
            # Choose sign for change
            sign = np.random.choice([-1, 1])
            # Generate mutation rate
            if one_fifth_rule:
                rate = self._sample_mut_rate()
            else:
                rate = self.mut_rate
            # Change value
            # Case 1 Discrete variable
            if isinstance(var, DiscreteVariable):
                i_in_dom = np.where(var.domain == x[index])[0][0]
                new_i = (sign + i_in_dom) % len(var.domain)
                x_mut[index] = var.domain[new_i]
            # Case 2 Continous variable
            else:
                x_mut[index] *= 1 + sign * rate

            # Check if it was something that could be 0
            if var.domain[0] == 0 and sign == -1:
                if x_mut[index] < 1e-5 and np.random.choice([False, True]):
                    x_mut[index] = 0
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}.")

        # Check x and change weights if they exist
        x_mut = self.check_x(variables, x_mut)
        if isinstance(x, WeightedMetaArray):
            x_mut.weights = copy.deepcopy(x.weights)
            x_mut.metadata = x.metadata
        else:
            x_mut.weights = np.ones(len(x))
        x_mut.weights[index] += 1
        x_mut.metadata += 'm'
        return x_mut

    def mutation(self, x, variables, mutation_type='gaussian',
                 one_fifth_rule=True, attemts=1):
        """
        Mutation operator in genetic algorithm of values `x` of variables
        `variables`. The number of parameters to
        mutate will be sampled from binomial distribution with mean equal to
        mutation strength. The type of change of chosen parameters could be
        set to one of three operators:

        * 'uniform' - new values will be sampled uniformly between bounds.

        * 'resample' - new values will be sampled from the random\
        distribution of the variables.

        * 'gaussian' - will adds a unit Gaussian distributed random value.\
        The mean of the Gaussian distribution will be taken from the mutation\
        rate.

        :param x: values to mutate.
        :param variables: variables.
        :param mutation_type: type of mutation operator. Could be 'gaussian',
                              'uniform' and 'resample'.
        :type mutation_type: str
        :param fval: max number of function evaluations.
        :param one_fifth_rule: If True then one fifth rule will be used. For
                         'gaussian' option only.
        :param weights: weights for parameters in x, the greater weight is
                        the greater probability to change it is.
        :param attemts: number of mutation attemts.
        :type attemts: int

        :returns: a mutated offspring. If `attempts` > 1 then a list of\
                  mutated offsprings. All offsprings have information about\
                  the number of changes of each parameter in `weights`\
                  attribute.
        """
        # Simple checks
        assert len(x) == len(variables)
        assert attemts > 0

        # Generate number of parameters to change
        num_inds = self._sample_number_of_changes(n=len(x))

        # Choose parameters to change according to the weights if they are
        # set otherwise uniformly
        if isinstance(x, WeightedMetaArray):
            weights = x.weights
        else:
            weights = np.ones(num_inds)
        inds = choose_by_weight(range(len(x)), weights, num_inds)

        # Copy the array to change
        x_mut = [np.array(x, dtype=get_correct_dtype(x))
                 for _ in range(attemts)]

        # Start mutation procedure
        for attempt in range(attemts):
            i_att = 0
            while np.all(x_mut[attempt] == x) and i_att < 5:
                i_att += 1
                for ind in inds:
                    x_mut[attempt] = self.mutation_by_ind(x_mut[attempt],
                                                          variables, ind,
                                                          mutation_type,
                                                          one_fifth_rule)
        return x_mut

    def crossover(self, parent1, parent2, variables, crossover_type='uniform',
                  k=2, one_child=True):
        """
        Crossover operator in genetic algorithm. Could be of two types:

        * 'k-point' - k points will be chosen among the vector and each part\
        between those points will be taken from parent1 or parent2 (swapping)\
        . By default k=2.

        * 'uniform' - each parameter will be taken from either parent with\
        equal probability.

        :param parent1: array of first parent.
        :param parent2: array of second parent.
        :param crossover_type: type of crossover operator. Could be 'k-point'
                               or 'uniform'.
        :type crossover_type: str
        :param k: value of k for 'k_point' crossover.
        :type k: int
        :param one_child: if True then one child will be generated and
                          returned.
        :type one_child: bool
        """
        assert len(parent1) == len(parent2)

        parent1 = np.array(parent1, dtype=get_correct_dtype(parent1))
        parent2 = np.array(parent2, dtype=get_correct_dtype(parent2))

        # Create two children - copies of parents
        dtype = float
        if parent1.dtype == object or parent2.dtype == object:
            dtype = object
        child1 = WeightedMetaArray(parent1, dtype=dtype)
        child2 = WeightedMetaArray(parent2, dtype=dtype)

        i_att = 0
        while len(parent1) > 1 and np.all(child1 == parent1) and i_att < 5:
            i_att += 1
            if crossover_type == 'k_point':
                assert k > 0
                assert k < len(parent1)
                # Create list of points
                swp_inds = np.random.choice(range(len(parent1)), size=k)
                # One index must be inside array to make child different to
                # parents
                swp_inds[-1] = np.random.choice(range(1, len(parent1) - 1))
                swp_inds = sorted(swp_inds)
                swp_inds.insert(0, 0)
                swp_inds.append(len(parent1))
                # Swap parameters between points
                for i in range(len(swp_inds) - 1):
                    if i % 2 == 1:
                        p1, p2 = swp_inds[i], swp_inds[i+1]
                        child1[p1:p2] = parent2[p1:p2]
                        child2[p1:p2] = parent1[p1:p2]
            elif crossover_type == 'uniform':
                for i in range(len(parent1)):
                    change = np.random.choice([False, True])
                    if change:
                        child1[i] = parent2[i]
                        child2[i] = parent1[i]
            else:
                raise ValueError(f"Unknown crossover type: {crossover_type}.")

        child1 = self.check_x(variables, child1)
        child2 = self.check_x(variables, child2)
        child1.metadata = 'c'
        child2.metadata = 'c'
        ret = (child1, child2)
        if one_child:
            return ret[np.random.choice([0, 1])]
        return ret

    def selection(self, f, variables, X_gen, Y_gen=None,
                  selection_type='roulette_wheel', selection_random=False):
        """
        Perform selection in genetic algorithm.
        Selection could be of different types:

        * Roulette Wheel - the better fitness function is the higher chance
          to be selected for mutation and crossover for the individual is.
        * Rank - almost the same as Roulette Wheel but with rank insted
          fitness function. This means weight=1 for the best individual,
          weight=2 for the second best and so on.

        :param X_gen: previous generation of individuals.
        :param Y_gen: fitnesses of the previous generation. If `None` then
                         will be evaluated.
        :param selection_type: type of selection. Could be 'roulette_wheel' or
                               'rank'.
        :param selection_random: if True then number of mutants and crossover's
                                 offsprings in new generation will be binomial
                                 random variable.

        :returns: new generation and its fitnesses.
        """
        # Evaluate fitness if None
        if Y_gen is None:
            Y_gen = [f(x) for x in X_gen]
        # Sort by value of fitness
        X_gen, Y_gen = sort_by_other_list(X_gen, Y_gen, reverse=False)

        # Simple checks
        assert len(X_gen[0]) == len(variables)
        assert len(X_gen) == len(Y_gen)

        # Start selection procedure
        if selection_type == 'roulette_wheel':
            Y_gen = np.array(Y_gen)
            if (np.all(Y_gen == Y_gen[0]) or
                    not (np.all(Y_gen < 0) or np.all(Y_gen > 0))):
                p = [1 / len(Y_gen) for _ in Y_gen]
            else:
                is_not_inf = np.logical_not(np.isinf(Y_gen))
                if np.sum(is_not_inf) == 1:  # special case
                    p = [float(x) for x in is_not_inf]
                else:
                    p = Y_gen / np.sum(Y_gen[is_not_inf])
                    p[np.isinf(p)] = 1  # will be inversed to 0
                    p[np.isnan(p)] = 1  # will be inversed to 0
                    # We need to reverse probs as we have minimization problem
                    p = 1 - p
                    p /= np.sum(p)
        elif selection_type == 'rank':
            n = len(X_gen)
            p = np.arange(1, n+1) / (n * (n - 1))
            p /= np.sum(p)
        else:
            raise ValueError(f"Unknown selection type: {selection_type}.")

        # Generate numbers for each operation
        if selection_random:
            n_mutants = np.random.binomial(self.gen_size, self.p_mutation)
            n_offsprings = np.random.binomial(self.gen_size, self.p_crossover)
            n_random_gen = np.random.binomial(self.gen_size, self.p_random)
        else:
            n_mutants = int(self.gen_size * self.p_mutation)
            n_offsprings = int(self.gen_size * self.p_crossover)
            n_random_gen = int(self.gen_size * self.p_random)

        # 1. Elitism
        new_X_gen = list(X_gen[:self.n_elitism])
        new_Y_gen = list(Y_gen[:self.n_elitism])

        # 2. Mutation
        for i in range(n_mutants):
            x_ind = np.random.choice(range(len(X_gen)), p=p)
            x = X_gen[x_ind]
            mutants = self.mutation(x, variables, self.mutation_type,
                                    self.one_fifth_rule, self.mut_attempts)
            fitness = [f(x_mut) for x_mut in mutants]

#            print("Time of main part of mutation: " + str(t3 - t1))
            # Take best mutant
            new_Y_gen.append(np.min(fitness))
            new_X_gen.append(mutants[fitness.index(new_Y_gen[-1])])

            # One more check for weights.
            # If new x is better, then we would like to decrease weights of
            # parameters back as this change was good.
            if new_Y_gen[-1] < Y_gen[x_ind]:
                if isinstance(x, WeightedMetaArray):
                    new_X_gen[-1].weights = x.weights

        # 3. Crossover
        for i in range(n_offsprings):
            ind1, ind2 = np.random.choice(range(len(X_gen)), size=2, p=p)
            parent1, parent2 = X_gen[ind1], X_gen[ind2]
            x = self.crossover(parent1, parent2, variables,
                               self.crossover_type, self.crossover_k)
            new_X_gen.append(x)
            new_Y_gen.append(f(x))

        # 4. Random individuals
        for i in range(n_random_gen):
            x = WeightedMetaArray(self.randomize(variables, self.random_type,
                                                 self.custom_rand_gen))
            x.metadata = 'r'
            new_X_gen.append(x)
            new_Y_gen.append(f(x))

        # Sort by fitness and return new generation
        new_X_gen, new_Y_gen = sort_by_other_list(new_X_gen, new_Y_gen,
                                                  reverse=False)
        new_X_gen = new_X_gen[:self.gen_size]
        new_Y_gen = new_Y_gen[:self.gen_size]
        return new_X_gen, new_Y_gen

    def _sample_mut_rate(self, mode='normal'):
        if mode == 'normal':
            # TODO: Think about std for this distribution
            return trunc_normal_3_sigma_rule(self.cur_mut_rate, 0.0, 1.0)
        elif mode == 'uniform':
            return np.random.uniform(0.0, 1.0)

    def _sample_number_of_changes(self, n):
        sample = np.random.binomial(n=n, p=self.cur_mut_strength)
        return max(1, int(sample))

    def check_x(self, variables, x, raises=False):
        for val, var in zip(x, variables):
            if raises:
                if not var.correct_value(val):
                    raise ValueError("Values in values vector are"
                                     " not in bounds.")
        for i in range(len(x)):
            if not isinstance(variables[i], DiscreteVariable):
                x[i] = min(x[i], variables[i].domain[1])
                x[i] = max(x[i], variables[i].domain[0])
                assert variables[i].correct_value(x[i])
            else:
                if not variables[i].correct_value(x[i]):
                    raise ValueError(f"Value ({x[i]}) of Discrete variable "
                                     f"is bad. Domain: {variables[i].domain}")
        return x

    def is_stopped(self, n_gen, n_eval, impr_gen=None, maxiter=None,
                   maxeval=None, ret_status=False):
        """
        Returns if genetic algorithm must stop.

        :param n_gen: current number of generations.
        :param n_eval: current number of function evaluations.
        :param impr_gen: number of last generation that improved value of
                         fitness function.
        :param maxiter: maximum number of generations.
        :param maxeval: maximum number of evaluation.
        :param ret_status: If True then return status and message.
        """
        status = 1
        message = "OPTIMIZATION IS NOT STOPPED"
        if impr_gen is None:
            impr_gen = n_gen
        is_stuck = (n_gen - impr_gen) >= self.n_stuck_gen
        if maxeval is not None:
            expect_feval = int(self.gen_size * self.p_mutation *
                               self.mut_attempts)\
                         + int(self.gen_size * self.p_crossover)\
                         + int(self.gen_size * self.p_random)
            stop_by_n_eval = (n_eval + expect_feval >= maxeval)
        else:
            stop_by_n_eval = False

        stop_by_n_gen = False
        if maxiter:
            stop_by_n_gen = n_gen >= maxiter

        if stop_by_n_eval:
            status = 2
            message = "MAXIMUM NUMBER OF FUNCTION EVALUATIONS ACHIEVED"
        if stop_by_n_gen:
            status = 3
            message = "MAXIMUM NUMBER OF GENERATIONS ACHIEVED"
        if is_stuck:
            status = 0
            message = f"CONVERGENCE: NO IMPROVEMENT DURING {self.n_stuck_gen}"\
                      " ITERATIONS"
        ret_value = is_stuck or stop_by_n_eval or stop_by_n_gen
        if ret_status:
            return ret_value, status, message
        return ret_value

    @staticmethod
    def _write_report_to_stream(variables, run_info, stream):
        """
        Write report about one generation in report file.

        :param run_info: Run info that should have at least the following
                         fields:
                         * `result` (:class:`gadma.optimizers.OptimizerResult`\
                         type) - current result,
                         * `gen_times` - list of iteration times.
        :param report_file: File to write report. If None then to stdout.

        :note: All values are reported as is, i.e. `X_gen`, `x_best` should be\
               already translated from log scale if optimization did so;\
               `Y_gen` and `y_best` must be already multiplied by -1 if we\
               have maximization instead of minimization.
        """
        n_gen = run_info.result.n_iter
        X_gen = run_info.result.X_out
        Y_gen = run_info.result.Y_out
        x_best = run_info.result.x
        y_best = run_info.result.y
        mean_time = np.mean(run_info.gen_times)

        print(f"Generation #{n_gen}.", file=stream)
        print("Current generation of solutions:", file=stream)
        print("N", "Value of fitness function", "Solution",
              file=stream, sep='\t')

        for i, (x, y) in enumerate(zip(X_gen, Y_gen)):
            # Use parent's report write function
            string = Optimizer._n_iter_string(
                n_iter=i,
                variables=variables,
                x=x,
                y=f'{y: 5f}',
            )
            print(string, file=stream)

        print(f"Current mean mutation rate:\t{run_info.cur_mut_rate: 3f}",
              file=stream)
        print(f"Current mean number of params to change during mutation:\t"
              f"{max(int(run_info.cur_mut_strength * len(variables)), 1): 3d}",
              file=stream)

        print("\n--Best solution by value of fitness function--", file=stream)
        print("Value of fitness:", y_best, file=stream)
        print("Solution:", file=stream, end='')

        string = Optimizer._n_iter_string(
            n_iter='',
            variables=variables,
            x=x_best,
            y='',
        )
        print(string, file=stream)

        if mean_time is not None:
            print(f"\nMean time:\t{mean_time:.3f} sec.\n", file=stream)
        print("\n", file=stream)

    def _create_run_info(self):
        """
        Creates the initial run_info. It has the following fields:
        * `result` - empty :class:`gadma.optimizers.OptimizerResult` with\
          `n_iter`==-1.
        * `cur_mut_rate` - current value of mutation rate.
        * `cur_mut_strength` - current value of mutation strength.
        * `n_impr_gen` - number of iteration when improvement happened.
        * `gen_times` - list of times of iterations.
        """
        run_info = super(GeneticAlgorithm, self)._create_run_info()
        run_info.cur_mut_rate = self.mut_rate
        run_info.cur_mut_strength = self.mut_strength
        run_info.n_impr_gen = 0
        run_info.gen_times = []
        return run_info

    def _update_run_info(self, run_info, x_best, y_best, X, Y,
                         n_eval, gen_time=None, n_impr_gen=None,
                         maxiter=None, maxeval=None):
        """
        Updates fields of `run_info`after one iteration of GA.

        Fields of run_info like `cur_mut_rate`, `cur_mut_strength`, `gen_times`
        are updated. Also message, success and status from :meth:`.is_stopped`
        are recorded to `result` field.

        :param run_info: Run info to update.
        :param gen_time: Time of iteration.
        :param n_impr_gen: Number of iteration when improvement happened.
        :param maxiter: Maximum number of iterations.
        :param maxeval: Maximum number of evaluations.
        """
        super(GeneticAlgorithm, self)._update_run_info(run_info=run_info,
                                                       x_best=x_best,
                                                       y_best=y_best,
                                                       X=X,
                                                       Y=Y,
                                                       n_eval=n_eval)
        # Update mutation rates and strength
        if n_impr_gen is not None:
            run_info.n_impr_gen = n_impr_gen

            # Our n_iter was already increased so -1 is applied
            is_impr = (n_impr_gen == run_info.result.n_iter - 1)
            if self.one_fifth_rule:
                run_info.cur_mut_rate = update_by_one_fifth_rule(
                    run_info.cur_mut_rate,
                    self.const_mut_rate,
                    is_impr
                )
                run_info.cur_mut_rate = min(1.0, run_info.cur_mut_rate)
            is_mut_best = False
            x_best = run_info.result.x
            if hasattr(x_best, 'weights') and len(x_best.metadata) > 0:
                is_mut_best = x_best.metadata[-1] == 'm'
            run_info.cur_mut_strength = update_by_one_fifth_rule(
                run_info.cur_mut_strength,
                self.const_mut_strength,
                is_impr and is_mut_best
            )
            run_info.cur_mut_strength = min(1.0, run_info.cur_mut_strength)

        # Save gen_time
        if gen_time is not None:
            run_info.gen_times.append(gen_time)

        # Create message and success status
        stoped, status, message = self.is_stopped(run_info.result.n_iter,
                                                  run_info.result.n_eval,
                                                  run_info.n_impr_gen,
                                                  maxiter,
                                                  maxeval,
                                                  ret_status=True)
        run_info.success = stoped
        run_info.result.status = status
        run_info.result.message = message
        return run_info

    def valid_restore_file(self, save_file):
        """
        Returns True if save_file contains valid run_info and False otherwise.
        """
        try:
            run_info = self.load(save_file)
        except Exception:
            return False
        if (not isinstance(run_info.result.n_eval, int) or
                not isinstance(run_info.result.n_iter, int) or
                not isinstance(run_info.n_impr_gen, int)):
            return False
        if (not isinstance(run_info.cur_mut_rate, float) or
                not isinstance(run_info.cur_mut_strength, float)):
            return False
        if not isinstance(run_info.gen_times, list):
            return False
        return True

    def _optimize(self, f, variables, X_init, Y_init, maxiter, maxeval,
                  iter_callback):
        """
        Runs genetic algorithm to minimize value of objective function `f`.

        :param f: Function to minimize.
        :param variables: Variables of objective function.
        :param X_init: Initial points.
        :param Y_init: Value of `f` on `X_init`.
        :param maxiter: Maximum number of iterations.
        :param maxeval: Maximum number of evaluations.
        :param iter_callback: Callback to call after each iteration.
        """
        X_init, Y_init = sort_by_other_list(X_init, Y_init, reverse=False)
        X_gen = X_init[:self.gen_size]
        Y_gen = Y_init[:self.gen_size]

        x_best = X_gen[0]
        y_best = Y_gen[0]

        iter_callback(x_best, y_best, X_init, Y_init)

        n_impr_gen = self.run_info.n_impr_gen
        # Begin to create generations
        while (len(variables) > 0 and
                not self.is_stopped(n_gen=self.run_info.result.n_iter,
                                    n_eval=self.run_info.result.n_eval,
                                    impr_gen=self.run_info.n_impr_gen,
                                    maxiter=maxiter,
                                    maxeval=maxeval)):
            # record time of generation start
            start_time = time.time()
            # Form new generation
            X_gen, Y_gen = self.selection(f, variables, X_gen, Y_gen,
                                          self.selection_type,
                                          self.selection_random)

            # Check if we improve the result
            if y_best - Y_gen[0] >= self.eps:
                n_impr_gen = self.run_info.result.n_iter
            if y_best - Y_gen[0] > 0:
                x_best = X_gen[0]
                y_best = Y_gen[0]

            gen_time = time.time() - start_time
            update_kwargs = {"gen_time": gen_time,
                             "n_impr_gen": n_impr_gen,
                             "maxiter": maxiter,
                             "maxeval": maxeval}
            iter_callback(x_best, y_best, X_gen, Y_gen, **update_kwargs)
        return self.run_info.result


register_global_optimizer('Genetic_algorithm', GeneticAlgorithm)
