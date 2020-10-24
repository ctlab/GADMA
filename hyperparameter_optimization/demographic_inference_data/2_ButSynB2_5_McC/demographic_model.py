import moments
import numpy as np


def model_func(params, ns):
    """
    Demographic model with asymmetric migrations for two populations of
    butterflies. Data and model are from McCoy et al. 2013.
    Ancestral population split into two new formed populations
    with following continuous migrations between them.

    :param nuW: Size of first new formed population.
    :param nuC: Size of second new formed population.
    :param T: Time of ancestral population split.
    :param m12: Migration rate from second population to first one.
    :param m21: Migration rate from first population to second one.
    """
    nuW, nuC, T, m12, m21 = params
    sfs = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sfs)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

    m = np.array([[0, m12], [m21, 0]])
    fs.integrate(Npop=[nuW, nuC], tf=T, m=m, dt_fac=0.1)
    return fs
	
	
