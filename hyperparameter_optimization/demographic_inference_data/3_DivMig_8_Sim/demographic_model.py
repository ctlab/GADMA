import moments
import numpy as np


def model_func(params, ns):
    """
    Three populations demographic history with small number of parameters.
    In the model ancestral population is split into population 1 and
    population 2, each of which had constant size till now days. Population 3
    is formed by split from population 2 without change of its size and had
    constant size till now too. Migration rates are symmetrical.

    :param nu1: Size of population 1.
    :param nu2: Size of population 2.
    :param nu3: Size of population 3 after split from population 2.
    :param m12: Migration rate between population 1 and population 2.
    :param m13: Migration rate between population 1 and population 3.
    :param m23: Migration rate between population 2 and population 3.
    :param T1: Time between ancestral population split and divergence of
               population 3 from population 2.
    :param T2: Time of population 3 divergence from population 2.
    """
    nu1, nu2, nu3, m12, m13, m23, T1, T2 = params
    m112 = 0
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1] + ns[2])

    fs.integrate(Npop=[nu1, nu2], tf=T1, m = np.array([[0,m112],[m112,0]]))
    
    fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])
    
    fs.integrate(Npop=[nu1, nu2, nu3], tf=T2, m = np.array([[0,m12,m13],[m12,0,m23],[m13,m23,0]]))

    return fs
