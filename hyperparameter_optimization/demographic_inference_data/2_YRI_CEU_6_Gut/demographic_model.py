import moments
import numpy


def model_func(params, ns):
    """
    Demographic model for two modern human populations: YRI and CEU.
    Data and model are from Gutenkunst et al., 2009.

    Model with sudden growth of ancestral population size, followed by split,
    bottleneck in second population (CEU) with exponential recovery and
    symmetric migration.

    :param nu1F: The ancestral population size after growth.
    :param nu2B: The bottleneck size for second population (CEU).
    :param nu2F: The final size for second population (CEU).
    :param m: The scaled symmetric migration rate.
    :param Tp: The scaled time between ancestral population growth
               and the split.
    :param T: The time between the split and present.
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
	
	
