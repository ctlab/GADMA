import moments

import numpy as np

def model_func(params, ns):
    """
    Classical one population bottleneck model.

    :param nuB: Size of population during bottleneck.
    :param nuF: Size of population now.
    :param tB: Time of bottleneck duration.
    :param tF: Time after bottleneck finished.
    """
    nuB, nuF, tB, tF = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    fs.integrate([nuB], tB)
    fs.integrate([nuF], tF)
    return fs
