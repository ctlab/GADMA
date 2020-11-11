"""
Custom demographic model for our example.
"""
import numpy
import moments
import time
def model_func(params, ns):
    nu1F, nu2B, nu2F, m, Tp, T = params
    # f for the equilibrium ancestral population
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0]+ns[1]+ns[2])
    fs = moments.Spectrum(sts)

    
    # Now do the population growth event.
    fs.integrate([nu1F], Tp)
    # The divergence
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1]+ns[2])
    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: [nu1F, nu2B*(nu2F/nu2B)**(t/T)]
    fs.integrate(nu2_func, T, m=numpy.array([[0, m],[m, 0]]))

    fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])
    fs.integrate([nu1F, nu2B, nu2F], T, m=numpy.array([[0, m, 0],[m, 0, 0],[0, 0, 0]]))
    return fs

