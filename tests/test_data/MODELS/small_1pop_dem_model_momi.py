import momi

def model_func(params):
    """
    Some model
    """
    nuB, nuF, TB, TF = params
    model = momi.DemographicModel(N_e=1, gen_time=1,
                                  muts_per_gen=1e-8)

    model.add_leaf("Pop1", N=nuF)
    model.set_size("Pop1", N=nuB, g=0, t=TF)
    model.set_size("Pop1", N=10000, g=0, t=TB)

    return model
