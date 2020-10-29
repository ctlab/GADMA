"""
Custom demographic model for our example.
"""
import numpy
import moments
import time
def model_func(params, ns):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    m: The scaled migration rate
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """

    nu1F, nu2B, nu2F, m, Tp, T = params
    # f for the equilibrium ancestral population
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0]+ns[1])
    fs = moments.Spectrum(sts)

    
    # Now do the population growth event.
    fs.integrate([nu1F], Tp)
    # The divergence
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: [nu1F, nu2B*(nu2F/nu2B)**(t/T)]
    fs.integrate(nu2_func, T, m=numpy.array([[0, m],[m, 0]]))

    return fs

def prior_onegrow_nomig(params, ns):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, no migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    """
    nu1F, nu2B, nu2F, Tp, T = params
    return prior_onegrow_mig((nu1F, nu2B, nu2F, 0, Tp, T), ns)
