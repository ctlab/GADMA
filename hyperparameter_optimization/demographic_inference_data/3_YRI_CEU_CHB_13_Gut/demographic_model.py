import moments
import numpy


def model_func(params, ns):
    """
    Demographic model for three modern human populations: YRI, CEU and CHB.
    Data and model are from Gutenkunst et al., 2009.

    Model with sudden growth of ancestral population size, followed by split
    into population YRI and common population of CEU and CHB,
    which experience bottleneck and split with exponential recovery of both
    populations. Migrations between populations are symmetrical.

    :param nuAf: The ancestral population size after sudden growth
                 and size of YRI population.
    :param nuB: The bottleneck size of CEU+CHB common population.
    :param nuEu0: The bottleneck size for CEU population.
    :param nuEu: The final size of CEU population after exponential growth.
    :param nuAs0: The bottleneck size for CHB population.
    :param nuAs: The final size of CHB population after exponential growth.
    :param mAfB: The scaled symmetric migration rate between YRI and CEU+CHB  
                 populations.
    :param mAfEu: The scaled symmetric migration rate between YRI and CEU
                  populations.
    :param mAfAs: The scaled symmetric migration rate between YRI and CHB
                  populations.
    :param mEuAs: The scaled symmetric migration rate between CEU and CHB
                  populations.
    :param TAf: The scaled time between ancestral population growth
                and first split.
    :param TB: The time between the first split and second. Time of CEU+CHB
               population existence.
    :param TEuAs: The time between second split and present.
    """

    nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs = params
    n1, n2, n3 = ns
    theta = 0.37976
    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2+n3, theta=theta)
    fs = moments.Spectrum(sts)

    fs.integrate([nuAf], TAf, 0.05, theta=theta)
    
    fs = moments.Manips.split_1D_to_2D(fs, n1, n2+n3)
    
    mig1=numpy.array([[0, mAfB],[mAfB, 0]])
    fs.integrate([nuAf, nuB], TB, 0.05, m=mig1, theta=theta)
    
    fs = moments.Manips.split_2D_to_3D_2(fs, n2, n3)

    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    nu2 = lambda t: [nuAf, nuEu_func(t), nuAs_func(t)]
    mig2=numpy.array([[0, mAfEu, mAfAs],[mAfEu, 0, mEuAs],[mAfAs, mEuAs, 0]])
    
    fs.integrate(nu2, TEuAs, 0.05, m=mig2, theta=theta)
                                
    return fs
	
	
