import momi
import numpy as np

def model_func(params):
    """
    N_anc: Ancestral population size
    N_1F: Size of ancestral population after growth and YRI population
    N_2B: Bottleneck size of Eurasian population
    nu_2F_gen: Final size of CEU population (in genetic units)
    r2: Growth rate for CEU population
    N_3F: Final size of CHB population
    r3: Growth rate for CHB population
    T1: Time of epoch between ancestral size growth and first split
    T2: Time of second epoch between two splits
    T3: Time of last epoch, time of last split 
    """
    N_Anc, N_1F, N_2B, nu_2F_gen, r2, N_3F, r3, T1, T2, T3 = params

    model = momi.DemographicModel(N_e=1e5)

    # momi2 uses backward in time models
    # 1. Set leafs at time = 0
    model.add_leaf("YRI", N=N_1F)
    # we have nu_2F_gen is in genetic units so we translate it
    model.add_leaf("CEU", N=nu_2F_gen * N_Anc, g=r2)
    model.add_leaf("CHB", N=N_3F, g=r3)

    # 2. Merge CHB to CEU (T3 time ago) and set size to bottleneck with g=0
    model.move_lineages("CHB", "CEU", t=T3, N=N_1F, g=0)

    # 3. Merge CEU to YRI (T3+T2 time ago)
    model.move_lineages("CEU", "YRI", t=T3+T2)

    # 4. Change size of ancestral (YRI) population to N_anc (T3+T2+T1 time ago)
    model.set_size("YRI", N=N_Anc, g=0, t=T3+T2+T1)

    return model
