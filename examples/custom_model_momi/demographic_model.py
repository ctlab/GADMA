import momi
import numpy as np

def model_func(params):
    N_Anc, N_1F, r_2, nu_2F_gen, Tp, T = params

    model = momi.DemographicModel(N_e=1e5)

    model.add_leaf("YRI", N=N_1F)
    # we have nu_2F_gen is in genetic units so we translate it
    model.add_leaf("CEU", N=nu_2F_gen * N_Anc, g=r_2)
    # Time of population split is T
    model.move_lineages("CEU", "YRI", t=T, N=N_1F)
    # Time of ancestral size change is T + Tp
    model.set_size("YRI", N=N_Anc, g=0, t=T + Tp)

    return model

