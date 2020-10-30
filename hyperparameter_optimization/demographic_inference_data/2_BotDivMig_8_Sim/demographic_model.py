import moments
import numpy as np


def model_func(params, ns):
    """
    Demographic history of two populations with bottleneck of ancestral
    population followed by split and growth of both new formed populations
    exponentially and linearly correspondingly.

    :param nu: Size of ancestral population after sudden decrease.
    :param f: Fraction in which ancestral population splits.
    :param nu1: Size of population 1 after exponential growth.
    :param nu2: Size of population 2 after linear growth.
    :param m12: Migration rate from subpopulation 2 to subpopulation 1.
    :param m21: Migration rate from subpopulation 1 to subpopulation 2.
    :param T1: Time between sudden growth of ancestral population and its
               split.
    :param T2: Time of ancestral population split.
    """
    nu, f, nu1, nu2, m12, m21, T1, T2 = params
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs.integrate(Npop=[nu], tf=T1)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

    nu1_init = nu * f
    nu2_init = nu * (1 - f)
    nu_func = lambda t: [nu1_init + (nu1 - nu1_init) * (t / T2),
                         nu2_init * (nu2 / nu2_init) ** (t / T2)]

    fs.integrate(Npop=nu_func, tf=T2, m = np.array([[0,m12],[m21,0]]))
    return fs
    
    
