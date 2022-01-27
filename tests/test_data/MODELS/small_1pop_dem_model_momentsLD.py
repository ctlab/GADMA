import moments.LD
import numpy as np


def model_func(params, rho, theta):
    # r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    # rhos = 4 * 10000 * r_bins
    """
    Some model
    """
    nuB, nuF, TB, TF, Nanc_size = params
    Y = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
    Y = moments.LD.LDstats(Y, num_pops=1)

    Y.integrate(tf=TB, nu=[nuB], rho=rho, theta=theta)
    Y.integrate(tf=TF, nu=[nuF], rho=rho, theta=theta)

    return Y
