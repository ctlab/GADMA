"""
Custom demographic model for our example.
"""
import numpy
import dadi

def model_func(params, ns, pts):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    m: The scaled migration rate
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """

    nu1F, nu2B, nu2F, m, Tp, T = params
    # Define the grid we'll use
    xx = yy = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the population growth event.
    phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F)

    # The divergence
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
    phi = dadi.Integration.two_pops(phi, xx, T, nu1=nu1F, nu2=nu2_func, 
                                    m12=m, m21=m)

    # Finally, calculate the spectrum.
    sfs = dadi.Spectrum.from_phi(phi, ns, (xx,yy))
    return sfs

def prior_onegrow_mig_mscore(params):
    """
    ms core command corresponding to prior_onegrow_mig
    """
    nu1F, nu2B, nu2F, m, Tp, T = params
    # Growth rate
    alpha2 = numpy.log(nu2F/nu2B)/T

    command = "-n 1 %(nu1F)f -n 2 %(nu2F)f "\
            "-eg 0 2 %(alpha2)f "\
            "-ma x %(m)f %(m)f x "\
            "-ej %(T)f 2 1 "\
            "-en %(Tsum)f 1 1"

    # There are several factors of 2 necessary to convert units between dadi
    # and ms.
    sub_dict = {'nu1F':nu1F, 'nu2F':nu2F, 'alpha2':2*alpha2,
                'm':2*m, 'T':T/2, 'Tsum':(T+Tp)/2}

    return command % sub_dict

def prior_onegrow_nomig(params, ns, pts):
    """
    Model with growth, split, bottleneck in pop2, exp recovery, no migration

    nu1F: The ancestral population size after growth. (Its initial size is
          defined to be 1.)
    nu2B: The bottleneck size for pop2
    nu2F: The final size for pop2
    Tp: The scaled time between ancestral population growth and the split.
    T: The time between the split and present

    n1,n2: Size of fs to generate.
    pts: Number of points to use in grid for evaluation.
    """
    nu1F, nu2B, nu2F, Tp, T = params
    return prior_onegrow_mig((nu1F, nu2B, nu2F, 0, Tp, T), ns, pts)
