from .optimizer import Optimizer, ConstrainedOptimizer

import copy
import numpy as np
import scipy
from functools import partial

_registered_global_optimizers = {}


class GlobalOptimizer(Optimizer):

    def __init__(self, log_transform=False, maximize=False):
        super(GlobalOptimizer, self).__init__(log_transform, maximize)
        self.X = list()
        self.Y = list()    

    def fix_f_and_args(self, f, args):
        def wrapper(x):
            if x in self.X:
                return self.Y[self.X.index(x)]
            y = self.evaluate(f, x, *args)
            self.X.append(x)
            self.Y.append(y)
            return y
        return wrapper

    def initial_design(self, f, variables, num_init,
                       X_init=None, Y_init=None):
        """
        Performs initial design for optimization.

        :param f: function to use for evaluations. Note that it should be
                  without arguments. Use :method:`self.fix_f_and_args` to
                  get such function from another one with arguments.
        :param variables: variables of function. They are used for random
                          generation of their values.
        :param num_init: number of initial solutions.
        :param X_init: list of some initial solutions.
        :param Y_init: list of function values on the initial solutions.

        :returns: pair of lists X and Y. Initial points and value of fitness\
                  function on them.
        """
        X = list()
        Y = list()
        if X_init is not None:
            if Y_init is not None:
                X = X_init
                Y = Y_init
                assert len(X) == len(Y)
            else:
                for x in X_init:
                    X.append(x)
                    Y.append(f(x))
        for _ in (num_init - len(X)):
            x = [var.resample() for var in variables]
            X.append(x)
            Y.append(f(x))
        return X, Y

    def optimize(self, f, variables, num_init, X_init=None, Y_init=None,
                 args=(), opitions={}, maxiter=None):
        raise NotImplementedError


def register_global_optimizer(id, optimizer):
    """
    Registers the specified global optimizer.
    """
    if id in _registered_global_optimizers:
        raise ValueError(f"Optimizer '{id}' is already registered.")
    if not isinstance(optimizer, LocalOptimizer):
        raise ValueError("Optimizer is not global.")
    _registered_global_optimizers[id] = optimizer
    optimizer.id = id


def get_global_optimizer(id):
    """
    Returns the global optimizer with the specified id.
    """
    if id not in _registered_global_optimizers:
        raise ValueError(f"Optimizer '{id}' is not registered")
    return _registered_global_optimizers[id]


def all_global_optimizers():
    """
    Returns an iterator over all registered global optimizers.
    """
    for optim in _registered_global_optimizers.values():
        yield optim


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
    def __init__(self, step_size=10, frac_best_models=0.2, 
                 frac_mutate_models=0.3, frac_cross_models=0.3,
                 mut_rate=0.2, const_mut_rate=1.2,
                 mut_strength=0.2, const_mut_strength=1.1,
                 eps=1e-2, no_improv_iter=100, local_opt_id='BFGS_log',
                 log_transform=False, maximize=False):
        self.step_size = step_size
        self.num_best_sol = int(frac_best_models * self.step_size)
        self.num_mutate_sol = int(frac_mutate_models * self.step_size)
        self.num_cross_sol = int(frac_cross_models * self.step_size)
        self.num_random_sol = self.step_size - self.num_mutate_sol\
                              - self.num_best_sol - self.num_cross_sol
        self.mut_rate = mut_rate
        self.mut_strength = mut_strength
        self.cur_mut_rate = mut_rate
        self.cur_mut_strength = mut_strength
        self.const_mut_rate = const_mut_rate
        self.const_mut_strength = const_mut_strength
        self.eps = eps
        self.is_stuck_step = no_improv_iter
        self.local_optimizer = get_local_optimizer(local_opt_id)
        super(GeneticAlgorithm, self).__init__(log_transform, maximize)

        assert isinstance(self.step_size, int)
        assert (self.frac_best_models >= 0 and self.frac_best_models <= 1)
        assert (self.frac_mutate_models >= 0 and self.frac_mutate_models <= 1)
        assert (self.frac_cross_models >= 0 and self.frac_cross_models <= 1)
        assert (self.frac_best_models + self.frac_mutate_models
                + self.frac_cross_models <= 1)
        assert (mut_rate >= 0 and mut_rate <= 1)
        assert (mut_strength >= 0 and mut_strength <= 1)
        assert (const_mut_rate >= 1 and const_mut_rate <= 2)
        assert (const_mut_strength >= 0 and const_mut_strength <= 1)

    def mutation(self, x, variables, mutation_type='gaussian',
                 one_fifth_rule=True, attemts=1):
        """
        Mutation operator in genetic algorithm of values `x` of variables
        `variables`. The number of parameters to
        mutate will be sampled from binomial distribution with mean equal to
        mutation strength. The type of change of chosen parameters could be set
        to one of three operators:

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
        if hasattr(x, weights):
            weights = x.weights
        else:
            weights = np.ones(num_inds)
        inds = self.choose_by_weight(range(len(x)), weights, num_inds) 

        # Copy the array to change
        x_mut = np.array(x) * np.ones((attemts, len(x)))]

        # Start mutation procedure
        # 1. Uniform type
        if mutation_type == 'uniform':
            for i, var in zip(inds, variables[inds]):
                # Case 1 Discrete variable
                if isinstance(var, DiscreteVariable):
                    x_mut[:, i] = np.random.choice(var.domain, size=attemts)
                # Case 2 Continous variable
                else:
                    for at in range(attemts):
                        x_mut[at, i] = np.random.uniform(var.domain[0],
                                                         var.domain[1])
        # 2. Resample type
        elif mutation_type == 'resample':
            for i in range(attemts):
                x_mut[i, inds] == [var.resample()
                                   for var in variables[inds]]
        # 3. Gaussian type
        elif mutation_type == 'gaussian':
            # Choose signs for each change
            signs = np.choice([-1, 1], size=(attempts, num_inds))
            # Generate mutation rate
            if one_fifth_rule:
                rates = [self._sample_mut_rate()
                         for _ in range(attemts * len(inds))]
            else:
                rates = [self.mut_rate for _ in range(attemts * len(inds))]
            rates = np.array(rates).reshape((attemts, len(inds)))
            # Change values
            for i, var in zip(inds, variables[inds]):
                # Case 1 Discrete variable
                if isinstance(var, DiscreteVariable):
                    i_in_dom = var.domain.index(x[i])
                    new_i = np.mod(signs[:, i] + i_in_dom, len(var.domain))
                    x_mut[:, i] = var.domain[new_i]
                # Case 2 Continous variable
                else:
                    x_mut[:, i] *= np.multiply(signs[:, i], rates[:, i]) + 1
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}.")

        # Check x and change weights if they exist
        for i in range(attemts):
            x_mut[i] = self.check_x(variables, x_mut[i])
            if hasattr(x, weights):
                x_mut[i].weights = copy.deepcopy(x.weights)
            else:
                x_mut[i].weights = np.ones(len(x))
            x_mut[i].weights[inds] += 1
        if attemts == 1:
            return x_mut[0]
        return x_mut

    def crossover(self, parent1, parent2, crossover_type='uniform',
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
        child1 = np.array(parent1)
        child2 = np.array(parent2)

        if crossover_type == 'k_point':
            assert k > 0
            assert k < len(parent1)
            # Create list of points
            swp_inds = np.random.choice(range(len(parent1)), size=k)
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
        if one_child:
            return np.random.choice([child1, child2])
        return (child1, child2)

    def selection(self, f, variables, X_gen, Y_gen=None,
                  selection_type='roulette_wheel'):
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

        :returns: new generation and its fitnesses.
        """
        # Evaluate fitness if None
        if Y_gen is None:
            Y_gen = [f(x) for x in X_gen]
        # Sort by value of fitness
        X_gen, Y_gen = sort_by_other_list(X_gen, Y_gen)

        # Simple checks
        X_gen = np.array(X_gen)
        assert X_gen.ndim == 2
        assert X_gen.shape[1] == len(variables)
        assert len(X_gen) == len(Y_gen)

        # Start selection procedure
        if selection_type == 'roulette_wheel'
            p = np.array(Y_gen) / np.sum(Y_gen)
            # We need to reverse probs as we have minimization problem
            p = 1 - p
        elif selection_type == 'rank':
            n = len(X_gen) 
            p = np.arange(1, n+1) / (n * (n - 1))
        else:
            raise ValueError(f"Unknown selection type: {selection_type}.")

        # 1. Elitism
        new_X_gen = copy.deepcopy(X_gen[:self.num_best_sol])
        new_Y_gen = copy.deepcopy(Y_gen[:self.num_best_sol])

        # 2. Mutation
        for i in range(self.num_mutate_sol):
            x = np.random.choice(X_gen, p=p)
            mutants = self.mutation(x, self.mutation_type,
                                    self.one_fifth_rule, self.mut_attempts)
            fitness = [f(x_mut) for x_mut in mutants]
            
            # Take best mutant
            new_Y_gen.append(np.min(fitness))
            new_X_gen.append(mutants[fitness.index(new_Y_gen[-1])])

            # One more check for weights.
            # If new x is better, then we would like to decrease weights of
            # parameters back as this change was good.
            if new_Y_gen[-1] < Y_gen[X_gen.index(x)]:
                if hasattr(x, weights):
                    new_X_gen[-1].weights = x.weights
                else:
                    del new_X_gen[-1].weights

        # 3. Crossover
        for i in range(self.num_cross_sol):
            parent1, parent2 = np.random.choice(X_gen, size=2, p=p)
            x = self.crossover(individual, self.crossover_type))
            new_gen.append(x)
            new_fit.append(f(x))

        # 4. Random individuals
        for i in range(self.num_random_sol):
            x = np.array([var.resample() for var in variables])
            new_gen.append(x)
            new_fit.append(f(x))

        # Sort by fitness and return new generation
        new_X_gen, new_Y_gen = sort_by_other_list(new_X_gen, new_Y_gen)
        new_X_gen = new_X_gen[:self.gen_size]
        new_Y_gen = new_Y_gen[:self.gen_size]
        return new_X_gen, new_Y_gen

    def initial_design(self, f, variables, num_init,
                       X_init=None, Y_init=None):
        X, Y = super(GeneticAlgorithm, self).initial_design(f, variables, num_init,
                                                     X_init, Y_init)
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
            if raises and (val < var.domain[0] or val > var.domain[1]):
                raise ValueError("Values in values vector are not in bounds.")
        for i in range(len(x)):
            x[i] = min(x[i], variables[i].domain[1])
            x[i] = max(x[i], variables[i].domain[0])
        return x

    def is_stopped(self, n_gen, n_eval, impr_gen=None, maxeval=None):
        """
        Returns if genetic algorithm must stop.

        :param n_gen: current number of generations.
        :param n_eval: current number of function evaluations.
        :param impr_gen: number of last generation that improved value of
                         fitness function.
        :maxeval: maximum number of evaluation.
        """
        if impr_gen is None:
            impr_gen = n_gen
        is_stuck = (n_gen - impr_gen) >= self.is_stuck_step
        if maxiter is not None:
            n_eval_next_gen = self.num_cross_sol
                            + self.num_mutate_sol * self.mut_attempts
                            + self.num_random_sol
            return (n_eval + n_eval_next_gen > maxeval) and is_stuck
        return is_stuck

    def optimize(self, f, variables, args=(), num_init=50,
                 X_init=None, Y_init=None, n_gen_init=0, maxiter=None,
                 report_file=None, eval_file=None, save_file=None):
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
        :param maxiter: maximum number of function evaluations.
        """
        # First we initialize initial values of some options
        self.cur_mut_rate = self.mut_rate
        self.cur_mut_strength = self.mut_strength
        self.weights = np.zeros(len(variables))

        f_in_opt = self.prepare_f_for_opt(f, args, eval_file)
        X_gen, Y_gen = self.perform_initial_design(self.f, self.variables, num_init,
                                    X_init, Y_init)

        n_gen = 0
        impr_gen = n_gen
        n_eval = f_in_opt.cache_info().misses
        x_best = X_gen[0]
        y_best = Y_gen[0]
        assert n_eval > 0
        while not self.is_stopped(n_gen, n_eval, impr_gen, maxiter):
            X_gen, Y_gen = self.selection(f_in_opt, variables, X_gen, Y_gen)
            n_eval = f_in_opt.cache_info().misses
            n_gen += 1
            if y_best > Y_gen[0]:
                x_best = X_gen[0]
                y_best = Y_gen[0]
        return x_best, y_best
            

        

