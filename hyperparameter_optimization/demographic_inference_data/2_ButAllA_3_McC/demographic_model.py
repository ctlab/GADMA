import moments
import numpy as np


def model_func(params, ns):
    """
    Demographic model without migration for two populations of butterflies.
    Data and model are from McCoy et al. 2013.
    Model is very simple: ancestral population splits into two new populations
    of constant size.

    :param nuW: Size of first subpopulation.
    :param nuC: Size of second subpopulation.
    :param T: Time of ancestral population split.
    """
    nuW, nuC, T = params
    sfs = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sfs)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

    fs.integrate(Npop=[nuW, nuC], tf=T, dt_fac=0.1)
    return fs
	
	
