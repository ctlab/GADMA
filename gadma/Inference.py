import os,sys

import support
from genetic_algorithm import GA

import numpy as np
import math
import random
import copy
from gadma import options
from gadma.demographic_model import Demographic_model

try:
    import dadi as sim_sfs_lib
    import moments as sim_sfs_lib
except ImportError:
    try:
        import moments as sim_sfs_lib
    except ImportError:
        try:
            import dadi as sim_sfs_lib
        except ImportError:
            support.error("None of the dadi or the moments are installed")


def load_bootstrap_data_from_dir(dirname,  projections=None, pop_ids=None, filenames=False):
    from gadma.support import READ_ALLOWED_EXTENSIONS
    all_boot = []
    set_of_seen_files = set()
    for filename in os.listdir(dirname):
        extention = '.' + filename.split('.')[-1]
        filename_without_ext = '.'.join(filename.split('.')[:-1])
        if extention in READ_ALLOWED_EXTENSIONS.keys():
            if filename_without_ext in set_of_seen_files:
                continue
            filename = os.path.join(dirname, filename)
            fs, _1, _2 = support.load_spectrum(filename, projections, pop_ids)
            if filenames:
                all_boot.append([filename, fs])
            else:
                all_boot.append(fs)
            set_of_seen_files.add(filename_without_ext)
            
    return all_boot


def get_claic_component(func_ex, all_boot, p0, data, pts=None, eps=1e-2, version=3):
    '''
    if pts is None, then moments is used.
    Some help:
    moments.Godambe.get_hess(func, p0, eps, args=())
    moments.Godambe.get_grad(func, p0, eps, args=())
    '''
    from dadi import Godambe
    if pts is None:
        import moments
        func_ex_ = func_ex
    else:
        import dadi as sim_lib
        import moments
        def func_ex_(p, ns):
            return func_ex(p, ns, pts)

    ns = data.sample_sizes

    # Cache evaluations of the frequency spectrum inside our hessian/J 
    # evaluation function
    cache = {}
    def func(params, data):
        key = (tuple(params), tuple(ns))
        if key not in cache:
            cache[key] = func_ex_(params, ns)
        fs = cache[key]
        return moments.Inference.ll(fs, data)
        
    H = - Godambe.get_hess(func, p0, eps, args=[data], version=version)
    H_inv = np.linalg.inv(H)

    J = np.zeros((len(p0), len(p0)))
    for ii, boot in enumerate(all_boot):
        boot = moments.Spectrum(boot)
        grad_temp = Godambe.get_grad(func, p0, eps, args=[boot], version=version)

        J_temp = np.outer(grad_temp, grad_temp)
        J += J_temp
        
    J = J/len(all_boot)

    # G = J*H^-1
    G = np.dot(J, H_inv)

    return np.trace(G)


def get_claic_score(func_ex, all_boot, p0, data, pts=None, eps=1e-2):
    '''
    if pts is None, then moments is used.
    '''
    ns = data.sample_sizes
    if pts is None:
        import moments
        model = func_ex(p0, ns)
        ll_model = moments.Inference.ll_multinom(model, data)
    else:
        import dadi
        model = func_ex(p0, ns, pts)
        ll_model = dadi.Inference.ll_multinom(model, data)
    return 2 * get_claic_component(func_ex, all_boot, p0, data, pts=pts, eps=eps) - 2 * ll_model


def get_95_confidence_intervals(func_ex, p0, data, pts=None, log=False, multinom=True, eps=1e-2):
    '''
    if pts is None, then moments is used.
    '''
    ns = data.sample_sizes
    if pts is None:
        import moments as sim_lib
        stds = sim_lib.Godambe.FIM_uncert(func_ex, p0, data, log=log, multinom=multinom, eps=eps)
    else:
        import dadi as sim_lib
        stds = FIM_uncert(func_ex, pts, p0, data, log=log, multinom=multinom, eps=eps)
    return np.array([[7000* (m - 1.96 * s), 7000 * (m + 1.96 * s)] for m,s in zip(p0, stds)])


def optimize_ga(number_of_params, data, model_func, pts=None, lower_bound=None, upper_bound=None, p0=None,
                 multinom=True, p_ids = None, mutation_strength=0.2, const_for_mut_strength=1.1, mutation_rate=0.2, const_for_mut_rate=1.2,
                 epsilon=1e-2, stop_iter=100, size_of_generation_in_ga=10, frac_of_old_models=0.2, frac_of_mutated_models=0.3, 
                 frac_of_crossed_models=0.3, optimization_name='optimize_log'):
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

    p_ids :             is list of special symbols, that define parameters as N, m, T or s.
                        N - size of population, (in Nref units)
                        m - migration rate, (in 1/Nref units)
                        T - time,  (in Nref units)
                        s - split ratio. (in 1 units)

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

    size_of_generation_in_ga: Number of models in one generation of GA.

    frac_of_old_models :    Fraction of models from previous ga generation that are taken 
                            to new generation.

    frac_of_mutated_models: Fraction of mutated models in new generation.

    frac_of_crossed_models: Fraction of crossed models in new generation.
    
    optimization_name:      Name of local optimization that will be run after genetic 
                            algorithm. By default, it is 'optimize_log'. If None then no
                            local optimization is run.
    """
    
    params = options.Options_storage()
    params.number_of_params = number_of_params
    params.input_data = data
    params.ns = np.array(data.shape) - 1 
    params.model_func = model_func
    params.dadi_pts = pts
    params.moments_scenario = pts is None
    params.lower_bound = lower_bound
    params.upper_bound = upper_bound
    params.multinom = multinom or p_ids is None
    params.optimize_name = optimization_name
    
    #create normalize funcs
    if p_ids is None:
        params.normalize_funcs = None
        params.p_ids = None
    else:
        params.p_ids = [x.lower()[0] for x in p_ids]
        params.normalize_funcs = []
        for i in xrange(number_of_params):
            id_v = p_ids[i][0].lower()
            if id_v == 'n' or id_v == 't':
                params.normalize_funcs.append(lambda x, y: x * y)
            elif id_v == 'm':
                params.normalize_funcs.append(lambda x, y: x / y)
            else:
                params.normalize_funcs.append(lambda x, y: x)
    
    params.mutation_strength = mutation_strength
    params.const_for_mut_strength = const_for_mut_strength
    params.mutation_rate = mutation_rate
    params.const_for_mut_rate = const_for_mut_rate
    params.epsilon = epsilon
    params.stop_iter = stop_iter
    params.size_of_generation = size_of_generation_in_ga
    params.frac_of_old_models = frac_of_old_models
    params.frac_of_mutated_models = frac_of_mutated_models
    params.frac_of_crossed_models = frac_of_crossed_models

    params.final_check()
    
    ga_instance = GA(params, one_initial_model=Demographic_model(params, initial_vector=p0))
    best_model = ga_instance.run()
    return best_model.params

