import moments
import numpy as np


def model_func(params, ns):
    """
    Demographic model of isolation for two populations with exponential growth
    of an ancestral population followed by split.

    :param nu: Size of ancestral population after exponential growth.
    :param nu1: Size of population 1 after split.
    :param nu2: Size of population 2 after split.
    :param T1: Time between exponential growth of ancestral population and its
               split.
    :param T2: Time of ancestral population split.
    """
    nu, nu1, nu2, T1, T2 = params
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs.integrate(Npop=lambda t: [(nu) ** (t / T1)], tf=T1)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

    fs.integrate(Npop=[nu1, nu2], tf=T2)
    return fs
    
    
