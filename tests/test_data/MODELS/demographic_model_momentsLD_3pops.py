import numpy
import moments.LD



def model_func(params, rho, theta):
    nu1F, nu2B, nu2F, m, Tp, T = params
    # f for the equilibrium ancestral population
    Y = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
    Y = moments.LD.LDstats(Y, num_pops=1)

    # Now do the population growth event.
    Y.integrate([nu1F], Tp, rho=rho, theta=theta)
    # The divergence
    Y = Y.split(0)
    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: [nu1F, nu2B * (nu2F / nu2B) ** (t / T)]
    Y.integrate(nu2_func, T, m=numpy.array([[0, m], [m, 0]]), rho=rho, theta=theta)

    Y = Y.split(1)
    Y.integrate([nu1F, nu2B, nu2F], T, m=numpy.array([[0, m, 0], [m, 0, 0], [0, 0, 0]]), rho=rho, theta=theta)
    return Y
