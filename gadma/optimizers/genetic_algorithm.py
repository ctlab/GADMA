from .optimizer import ConstrainedOptimizer
from .global_optimizer import GlobalOptimizer, register_global_optimizer
from .optimizer_result import OptimizerResult
from ..utils import sort_by_other_list, choose_by_weight, eval_wrapper
from ..utils import trunc_normal_3_sigma_rule, DiscreteVariable, WeightedMetaArray
from ..utils import update_by_one_fifth_rule
from functools import partial
import numpy as np
import copy
import pickle
import sys


class GeneticAlgorithm(GlobalOptimizer, ConstrainedOptimizer):
    """
    Class for Genetic Algorithm.

    :param step_size: number of solutions that will be used on each
                      iteration of genetic algorithm.
    :type step_size: int
    :param frac_best_models: fraction of best models from previous step
                             that will be taken to new iteration.
    :type frac_best_models: float
    :param frac_mutate_models: fraction of mutated models that will be
                               taken to new iteration.
    :type frac_mutate_models: float
    :param frac_cross_models: fraction of crossed models that will be
                              taken to new iteration.
    :type frac_cross_models: float
    :param mut_rate: initial mutation rate.
    :type mut_rate: float
    :param mut_strength: initial mutation "strength" - mean fraction of
                         model parameters that will be mutated.
    :type mut_strength: float
    :param const_mut_rate: constant to change mutation rate according to
                           one-fifth algorithm. Check GADMA paper for more
                           information.
    :type const_mut_rate: float
    :param const_mut_strength: constant to change mutation strength according
                               to one-fifth algorithm. Check GADMA paper for
                               more information.
    :type const_mut_strength: float
    :param eps: const for model's log likelihood compare.
                Model is better if its log likelihood is greater than 
                log likelihood of another model by epsilon.
    :type eps: float
    :param no_improv_iter: Number of iterations for GA stopping: GA stops when 
                           it can't improve model during max_iter iterations.
    :type no_improv_iter: int
    :param local_opt_id: id of local optimizer to use.
    :type local_opt_id: str
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
                 crossover_type='uniform', random_type='resample',
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
        self.mutation_type = mutation_type
        self.random_type = random_type
        self.one_fifth_rule = one_fifth_rule
        super(GeneticAlgorithm, self).__init__(log_transform, maximize)

    def mutation_by_ind(self, x, variables, index, mutation_type='gaussian',
                        one_fifth_rule=True):
        x_mut = WeightedMetaArray(x, dtype=object)
        var = variables[index]
        # Start mutation procedure
        # 1. Uniform type
        if mutation_type == 'uniform':
            # Case 1 Discrete variable
            if isinstance(var, DiscreteVariable):
                x_mut[index] = np.random.choice(var.domain)
            # Case 2 Continous variable
            else:
                x_mut[index] = np.random.uniform(var.domain[0], var.domain[1])
        # 2. Resample type
        elif mutation_type == 'resample':
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

        * 'resample' - new values will be sampled from the random distribution
        of the variables.

        * 'gaussian' - will adds a unit Gaussian distributed random value.
        The mean of the Gaussian distribution will be taken from the mutation
        rate.

        :param x: values to mutate.
        :param variables: variables.
        :param mutation_type: type of mutation operator. Could be 'gaussian',
                              'uniform' and 'resample'.
        :type mutation_type: str
        :param fval: max number of function evaluations.
        :one_fifth_rule: If True then one fifth rule will be used. For
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
        x_mut = [np.ones(len(x)) for _ in range(attemts)]

        # Start mutation procedure
        for attempt in range(attemts):
            for ind in inds:
                x_mut[attempt] = self.mutation_by_ind(x, variables, ind,
                                                           mutation_type,
                                                           one_fifth_rule)
        return x_mut

    def crossover(self, parent1, parent2, variables, crossover_type='uniform',
                  k=2, one_child=True):
        """
        Crossover operator in genetic algorithm. Could be of two types:

        * 'k-point' - k points will be chosen among the vector and each part
        between those points will be taken from parent1 or parent2 (swapping).
        By default k=2.

        * 'uniform' - each parameter will be taken from either parent with
        equal probability.

        :param parent1: array of first parent.
        :param parent2: array of second parent.
        :param crossover_type: type of crossover operator. Could be 'k-point' or
                               'uniform'.
        :type crossover_type: str
        :param k: value of k for 'k_point' crossover.
        :type k: int
        :param one_child: if True then one child will be generated and returned.
        :type one_child: bool
        """
        assert len(parent1) == len(parent2)

        # Create two children - copies of parents
        child1 = WeightedMetaArray(parent1, dtype=object)
        child2 = WeightedMetaArray(parent2, dtype=object)

        if crossover_type == 'k_point':
            assert k > 0
            assert k < len(parent1)
            # Create list of points
            swp_inds = np.random.choice(range(len(parent1)), size=k)
            # One index must be inside array to make child different to parents
            swp_inds[0] = np.random.choice(range(1, len(parent1) - 1))
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
        X_gen, Y_gen = sort_by_other_list(X_gen, Y_gen)

        # Simple checks
        assert len(X_gen[0]) == len(variables)
        assert len(X_gen) == len(Y_gen)

        # Start selection procedure
        if selection_type == 'roulette_wheel':
            p = np.array(Y_gen) / np.sum(Y_gen)
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
                               self.crossover_type)
            new_X_gen.append(x)
            new_Y_gen.append(f(x))

        # 4. Random individuals
        for i in range(n_random_gen):
            x = WeightedMetaArray(self.randomize(variables, self.random_type), dtype=object)
            x.metadata = 'r'
            new_X_gen.append(x)
            new_Y_gen.append(f(x))

        # Sort by fitness and return new generation
        new_X_gen, new_Y_gen = sort_by_other_list(new_X_gen, new_Y_gen)
        new_X_gen = new_X_gen[:self.gen_size]
        new_Y_gen = new_Y_gen[:self.gen_size]
        return new_X_gen, new_Y_gen

    def initial_design(self, f, variables, num_init,
                       X_init=None, Y_init=None):
        if Y_init:
            _Y_init = [self.sign * y for y in Y_init]
        else:
            _Y_init = None
        X, Y = super(GeneticAlgorithm, self).initial_design(f, variables,
                                                            num_init, X_init,
                                                            _Y_init,
                                                            self.random_type)
        X, Y = sort_by_other_list(X, Y)
        return X[:self.gen_size], Y[:self.gen_size]

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
            else:
                if not variables[i].correct_value(x[i]):
                    raise ValueError("Value of Discrete variable is bad.")
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
            expect_feval = int(self.gen_size * self.p_mutation * self.mut_attempts)\
                         + int(self.gen_size *self.p_crossover)\
                         + int(self.gen_size * self.p_random)
            stop_by_n_eval = (n_eval + expect_feval > maxeval)
        else:
            stop_by_n_eval = False

        stop_by_n_gen = False
        if maxiter:
            stop_by_n_gen = n_gen > maxiter

        if stop_by_n_eval:
            status = 1
            message = "MAXIMUM NUMBER OF FUNCTION EVALUATIONS ACHIEVED"
        if is_stuck:
            status = 0
            message = f"CONVERGENCE: NO IMPROVEMENT DURING {self.n_stuck_gen}"\
                      " ITERATIONS"
        if ret_status:
            return is_stuck or stop_by_n_eval or stop_by_n_gen, status, message
        return is_stuck or stop_by_n_eval or stop_by_n_gen

    def initialize_report_file(self, report_file):
        """
        Write first line in report file.
        """
        if report_file:
            ensure_file_existence(report_file)
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        print("--Start genetic algorithm pipeline--", file=stream)
        if report_file:
            stream.close()

    def write_report(self, n_gen, variables, X_gen, Y_gen, x_best, y_best,
                     report_file):
        """
        Write report about one generation in report file.
        """
        if report_file:
            stream = open(report_file, 'a')
        else:
            stream = sys.stdout
        print(f"Generation #{n_gen}.", file=stream)
        print("Current generation of solutions:", file=stream)
        print("N", "Value of fitness function", "Solution",
              file=stream, sep='\t')
        for i, (x, y) in enumerate(zip(X_gen, Y_gen)):
            # Use parent's report write function
            super(GeneticAlgorithm, self).write_report(i, variables, x,
                                                       f'{self.sign * y: 5f}',
                                                       report_file)
        if self.one_fifth_rule:
            print(f"Current mean mutation rate:\t{self.cur_mut_rate: 3f}",
                  file=stream)
        print(f"Current mean number of params to change during mutation:\t"
              f"{min(int(self.cur_mut_strength * self.gen_size), 1): 3f}",
              file=stream)

        print("\n--Best solution by value of fitness function--", file=stream)
        print("Value of fitness:", self.sign * y_best, file=stream)
        print("Solution:", file=stream, end='')
        super(GeneticAlgorithm, self).write_report('', variables, x_best,
                                                   '', report_file)
        print("\n", file=stream)

        if report_file:
            stream.close()

    def save(self, n_gen, X_gen, Y_gen, X_total, Y_total, save_file):
        """
        Save some values of genetic algorithm to file.
        """
        if save_file is None:
            return
        with open(save_file, 'wb') as fl:
            pickle.dump((n_gen, X_gen, Y_gen, X_total, Y_total), fl)

    @staticmethod
    def load(save_file):
        """
        Load some values of genetic algorithm from file.
        """
        with open(save_file, 'rb') as fl:
            n_gen, X_gen, Y_gen, X_total, Y_total = pickle.load(fl)
        return n_gen, X_gen, Y_gen, X_total, Y_total

    def optimize(self, f, variables, args=(), num_init=50,
                 X_init=None, Y_init=None, n_gen_init=0,
                 maxiter=None, maxeval=None,
                 verbose=0, callback=None, report_file=None, eval_file=None,
                 save_file=None):
        """
        Return best values of :param:`variables` that minimizes/maximizes
        the function :param:`f`.

        :param f: function to minimize/maximize. The usage must be the
                  following: f(x, *args), where x is list of values.
        :param variables: list of variables (instances of
                          :class:`gadma.Variable` class) of the function.
        :param X_init: list of initial values.
        :param Y_init: value of function `f` on initial values from `X_init`.
        :param args: arguments of function `f`.
        :param maxiter: maximum number of genetic algorithm's generations.
        :param maxeval: maximum number of function evaluations.
        :param callback: callback to call after each generation.
                         It will be called as callback(x, y), where x, y -
                         best_solution of generation and its fitness.
        """
        # First we initialize initial values of some options
        self.cur_mut_rate = self.mut_rate
        self.cur_mut_strength = self.mut_strength

        # Create logging files
        if eval_file:
            ensure_file_existence(eval_file)
        if report_file:
            ensure_file_existence(report_file)
        if save_file:
            ensure_file_existence(report_file)

        # Prepare function to use it. 
        # Fix args and cache
        prepared_f = self.prepare_f_for_opt(f, args)
        # Wrap for automatic evaluation logging
        finally_wrapped_f = eval_wrapper(prepared_f, eval_file)
        f_in_opt = partial(self.evaluate, finally_wrapped_f)

        # Write first line of report
        if verbose > 0:
            self.initialize_report_file(report_file)

        # Perform 0 generation of GA - initial design.
        X_gen, Y_gen = self.initial_design(f_in_opt, variables, num_init,
                                    X_init, Y_init)
        X_total, Y_total = copy.deepcopy(X_gen), copy.deepcopy(Y_gen)

        # Initialize number of generations, evaluations, best values and so on
        n_gen = 0
        n_impr_gen = n_gen
        n_eval = prepared_f.cache_info.misses
        x_best = X_gen[0]
        y_best = Y_gen[0]
        assert n_eval > 0

        # Write report about 0 generation
        if verbose > 0:
            self.write_report(n_gen, variables, X_gen, Y_gen, x_best, y_best, report_file)
        self.save(n_gen, X_gen, Y_gen, X_total, Y_total, save_file)

        # Begin to create generations
        while not self.is_stopped(n_gen, n_eval, n_impr_gen, maxiter, maxeval):
            # Form new generation
            X_gen, Y_gen = self.selection(f_in_opt, variables, X_gen, Y_gen,
                                          self.selection_type,
                                          self.selection_random)
            # Save all generations.
            X_total.extend(copy.deepcopy(X_gen))
            Y_total.extend(copy.deepcopy(Y_gen))

            # Check if we improve the result
            if (y_best - Y_gen[0]) >= self.eps:
                n_impr_gen = n_gen
                x_best = X_gen[0]
                y_best = Y_gen[0]

            # Update mutation rates and strength
            is_impr = (n_impr_gen == n_gen)
            if self.one_fifth_rule:
                self.cur_mut_rate = update_by_one_fifth_rule(
                    self.cur_mut_rate, self.const_mut_rate, is_impr)
            is_mut_best = False
            if hasattr(x_best, 'weights'):
                is_mut_best = x_best.metadata[-1] == 'm'
            self.cur_mut_strength = update_by_one_fifth_rule(
                self.cur_mut_strength, self.const_mut_strength,
                is_impr and is_mut_best)

            # Update numbers
            n_gen += 1
            n_eval = prepared_f.cache_info.misses

            # Callback
            if callback is not None:
                callback(x_best, y_best)

            # Write report about current generation
            if verbose > 0 and n_gen % verbose == 0:
                self.write_report(n_gen, variables, X_gen, Y_gen,
                                  x_best, y_best, report_file)
            # Save generation
            self.save(n_gen, X_gen, Y_gen, X_total, Y_total, save_file)

        # Construct OptimizerResult object to return
        _, status, message = self.is_stopped(n_gen, n_eval, n_impr_gen,
                                             maxiter, maxeval, ret_status=True)
        sign = -1 if self.maximize else 1
        Y = [self.sign * y for y in Y_total]
        Y_out = [self.sign * y for y in Y_gen]
        result = OptimizerResult(x=x_best, y=self.sign*y_best, success=True,
                                 status=status, message=message, X=X_total,
                                 Y=Y, n_eval=n_eval, n_iter=n_gen,
                                 X_out=X_gen, Y_out=Y_out)

        return result
            
register_global_optimizer('Genetic_algorithm', GeneticAlgorithm)
