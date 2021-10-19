import numpy as np
import moments.LD


def model_func(params, rho, theta):
    nu1F, nu2B, nu2F, m, Tp, T = params
    Y = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
    Y = moments.LD.LDstats(Y, num_pops=1, pop_ids=None)
    Y.integrate(tf=Tp, nu=[nu1F], rho=rho, theta=theta)
    Y = Y.split(0)
    nu2_func = lambda t: nu2B * (nu2F / nu2B) ** (t / T)
    migs = np.array([[0, m], [m, 0]])
    Y.integrate(tf=T, nu=lambda t: [nu1F, nu2_func(t)], m=migs, rho=rho, theta=theta)

    Y = moments.LD.LDstats(
        [(y_l + y_r) / 2 for y_l, y_r in zip(
            Y[:-2], Y[1:-1])]
        + [Y[-1]],
        num_pops=Y.num_pops,
        pop_ids=Y.pop_ids,
    )
    Y = moments.LD.Inference.sigmaD2(Y)
    return Y
