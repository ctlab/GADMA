import moments
import numpy as np


def model_func(params, ns):
    """
    Four population demographic history with 11 parameters. Ancestral
    population of constant size was split (T1 + T2 + T3) time ago into two new
    populations. First (1) population has constant size till nowdays. After
    divergence there was migration to population 1 from another population.
    The second formed population turned out to be a common population to
    another three populations: (T2 + T3) time ago it splits and formed
    so-called population 2 and T3 time ago the second formed population
    divided into population 3 and population 4. There were almost no
    migrations between populations except symmetric one between populations 3
    and 4.
    

    :param nu1: Size of population 1 after split of ancestral population.
    :param nu234: Size of common ancestor population of populations 2,
                  3 and 4 after split of ancestral population.
    :param nu2: Size of population 2 after split of common ancestor population
                of populations 2, 3 and 4.
    :param nu34: Size of common ancestor population of populations 3 and 4
                 after division of population 2 from their common ancestor
                 population.
    :param nu3: Size of population 3.
    :param nu4: Size of population 4.
    :param m12_anc: Migration rate to population 1 from common ancestor
                    population of populations 2, 3 and 4.
    :param m34_sym: Symmetric migration rate between populations 3 and 4.
    :param T1: Time between ancestral population split, population 1 formation
               and next split.
    :param T2: Time between ancestral population of populations 2, 3 and 4
               split, population 2 formation and next split.
    :param T3: Time of ancestral population of populations 3 and 4 split and
               formations of population 3 and population 4.
    """
    nu1, nu234, nu2, nu34, nu3, nu4, m12_anc, m34_sym, T1, T2, T3 = params
    sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sts)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

    m = np.zeros(shape=(2,2))
    m[1, 0] = m12_anc
    fs.integrate(Npop=[nu1, nu234], tf=T1, m=m)
    
    fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], sum(ns[2:]))
    
    fs.integrate(Npop=[nu1, nu2, nu34], tf=T2)

    fs = moments.Manips.split_3D_to_4D_3(fs, ns[2], sum(ns[3:]))

    m = np.zeros(shape=(4,4))
    m[2, 3] = m34_sym
    m[3, 2] = m34_sym
    fs.integrate(Npop=[nu1, nu2, nu3, nu4], tf=T3, m=m)

    return fs
