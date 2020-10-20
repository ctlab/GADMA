import moments

def model_func(params, ns):
    """
    Some model
    """
    nuB, nuF, TB, TF = params
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)

    fs.integrate(tf=TB, Npop=[nuB])
    fs.integrate(tf=TF, Npop=[nuF])

    return fs
