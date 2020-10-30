import moments
import numpy as np


def model_func(params, ns):
    """
    Simple two populations model. Ancestral population of constant size splits
    into two subpopulations of constant size with asymetrical migrations.

    :param nu1: Size of subpopulation 1 after split.
    :param nu2: Size of subpopulation 2 after split.
    :param m12: Migration rate from subpopulation 2 to subpopulation 1.
    :param m21: Migration rate from subpopulation 1 to subpopulation 2.
    :param T: Time of split.
    """

    nu1, nu2, m12, m21, T = params
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

    fs.integrate(Npop=[nu1, nu2], tf=T, m = np.array([[0,m12],[m21,0]]))
    return fs
    
    
