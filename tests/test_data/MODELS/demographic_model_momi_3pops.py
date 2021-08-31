"""
Custom demographic model for our example.
"""
import time
import momi
from matplotlib import pyplot as plt
import numpy as np

def model_func(params):
    Nanc, N_1F, r_2, N_2F, T_1, T_2 = params
    model = momi.DemographicModel(N_e=1, gen_time=1,
                                  muts_per_gen=1e-8)
    model.add_size_param('Nanc')
    model.add_size_param("N_1F")
    model.add_growth_param('r_2')
    model.add_size_param("N_2F")
    model.add_time_param("T_1")
    model.add_time_param("T_2")

    model.add_leaf("YRI", N="N_1F")
    model.add_leaf("CEU", N="N_2F", g='r_2')
    model.add_leaf("CHB", N="N_2F", g=0)
    model.move_lineages("CHB", "CEU", t="T_1", N="N_2F", g="r_2")
    model.move_lineages("CEU", "YRI", t=lambda params: 2*params.T_1, N="N_1F")
    model.set_size("YRI", N='Nanc', g=0, t=lambda params: params.T_2 + 2 * params.T_1)
    
    model.set_params({
        'Nanc': Nanc,
        'N_1F': N_1F,
        'r_2': r_2,
        'N_2F': N_2F,
        'T_1': T_1,
        'T_2': T_2,
    })
    return model

