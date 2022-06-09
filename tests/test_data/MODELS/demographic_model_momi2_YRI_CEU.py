"""
Custom demographic model for our example.
"""
import time
import momi
from matplotlib import pyplot as plt
import numpy as np

def model_func(params):
    Nanc, N_1F, r_2, N_2F, Tp, T = params
    model = momi.DemographicModel(N_e=1, gen_time=1,
                                  muts_per_gen=1e-8)
    model.add_size_param('N_Anc')
    model.add_size_param("N_1F")
    model.add_growth_param('r_2')
    model.add_size_param("N_2F")
    model.add_time_param("Tp")
    model.add_time_param("T")

    model.add_leaf("YRI", N="N_1F")
    model.add_leaf("CEU", N="N_2F", g='r_2')
    model.move_lineages("CEU", "YRI", t="T", N="N_1F")
    model.set_size("YRI", N='N_Anc', g=0, t="Tp")
    
    model.set_params({
        'N_Anc': Nanc,
        'N_1F': N_1F,
        'r_2': r_2,
        'N_2F': N_2F,
        'Tp': Tp,
        'T': T,
    })
    return model

