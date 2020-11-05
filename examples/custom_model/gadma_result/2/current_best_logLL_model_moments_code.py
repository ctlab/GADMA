import moments
import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location('module', 'demographic_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model_func = module.model_func


data = moments.Spectrum.from_file('2pop_e_gillettii_all_snp.fs')
ns = data.sample_sizes

p0 = [3.6536796762530233, 0.5756088036977836, 4.989089047598161, 0.33304022766326563, 0.6166099034251886]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = None


plot_ns = [4 for _ in ns]  # small sizes for fast drawing
gen_mod = moments.ModelPlot.generate_model(model_func,
                                           p0, plot_ns)
moments.ModelPlot.plot_model(gen_mod,
                             save_file='model_from_GADMA.png',
                             fig_title='Demographic model from GADMA',
                             draw_scale=False,
                             pop_labels=['WY', 'CO'],
                             nref=None,
                             gen_time=1.0,
                             gen_time_units='generations',
                             reverse_timeline=True)