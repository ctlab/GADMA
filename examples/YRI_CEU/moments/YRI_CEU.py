# Numpy is the numerical library moments is built upon
from numpy import array

#import dadi
import moments
# In demographic_models.py, we've defined a custom model for this problem
import demographic_models

import gadma

# Load the data
data = moments.Spectrum.from_file('YRI_CEU.fs')
ns = data.sample_sizes

# The Demographics1D and Demographics2D modules contain a few simple models,
# mostly as examples. We could use one of those.
func = moments.Demographics2D.split_mig
# Instead, we'll work with our custom model
func = demographic_models.prior_onegrow_mig

# Now let's optimize parameters for this model.

# The upper_bound and lower_bound lists are for use in optimization.
# Occasionally the optimizer will try wacky parameter values. We in particular
# want to exclude values with very long times, very small population sizes, or
# very high migration rates, as they will take a long time to evaluate.
# Parameters are: (nu1F, nu2B, nu2F, m, Tp, T)
upper_bound = [100, 100, 100, 10, 3, 3]
lower_bound = [1e-2, 1e-2, 1e-2, 0, 0, 0]

# This is our initial guess for the parameters, which is somewhat arbitrary.
p0 = [2,0.1,2,1,0.2,0.2]

# Perturb our parameters before optimization. This does so by taking each
# parameter a up to a factor of two up or down.
p0 = moments.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
                              lower_bound=lower_bound)

# Do the optimization. By default we assume that theta is a free parameter,
# since it's trivial to find given the other parameters. If you want to fix
# theta, add a multinom=False to the call.
# It is also possible to give initial parameters as p0=p0 argument, however
# this will not be a real global search as we want from genetic algorithm.
# As it is moments run we don't set pts argument.
# For more information: help(gadma.Inference.optimize_ga)
print('Beginning optimization ************************************************')
popt = gadma.Inference.optimize_ga(len(p0), data, func, #p0=p0,
                                   p_ids = ['n', 'n', 'n', 'm', 't', 't'],
                                   lower_bound=lower_bound,
                                   upper_bound=upper_bound,
                                   size_of_generation_in_ga=10, 
                                   stop_iter=50, 
                                   optimization_name='optimize_powell')
print('Finished optimization **************************************************')
print('Found parameters: {0}'.format(popt))

# Now we can compare our parameters with those that were obtained before:
print('From Gutenkunst et al 2009:')

# These are the actual best-fit model parameters, which we found through
# longer optimizations and confirmed by running multiple optimizations.
# We'll work with them through the rest of this script.
popt = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]
print('Best-fit parameters: {0}'.format(popt))

# Calculate the best-fit model AFS.
model = func(popt, ns)
# Likelihood of the data given the model AFS.
ll_model = moments.Inference.ll_multinom(model, data)
print('Maximum log composite likelihood: {0}'.format(ll_model))
# The optimal value of theta given the model.
theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))

