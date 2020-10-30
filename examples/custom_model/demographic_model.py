import moments
import numpy as np

def model_func(params, ns):
    nuW, nuC, T, m12, m21 = params
    sfs = moments.LinearSystem_1D.steady_state_1D(sum(ns))
    fs = moments.Spectrum(sfs)

    fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

    m = np.array([[0, m12], [m21, 0]])
    fs.integrate(Npop=[nuW, nuC], tf=T, m=m, dt_fac=0.1)
    return fs
