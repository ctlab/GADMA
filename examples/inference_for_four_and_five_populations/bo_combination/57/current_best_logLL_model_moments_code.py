import moments
import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location('module', '/mnt/tank/scratch/enoskova/bayes/demographic_inference_data/4_DivMig_11_Sim/demographic_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model_func = module.model_func


data = moments.Spectrum.from_file('/mnt/tank/scratch/enoskova/bayes/demographic_inference_data/4_DivMig_11_Sim/fs_data.fs')
data.pop_ids = ['Pop 1', 'Pop 2', 'Pop 3', 'Pop 4']
ns = data.sample_sizes

p0 = [1.4647122554543466, 0.1731035593604139, 1.2077822863545993, 0.512501596564237, 0.2991429287451106, 0.3784626868549182, 0.06941338259787184, 2.1608935474758577, 0.018570867126364823, 0.16127026328208538, 0.06771122384596226]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))

# As no theta0 or mut. rate + seq. length are not set
theta0 = 1.0
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))


plot_ns = [4 for _ in ns]  # small sizes for fast drawing
gen_mod = moments.ModelPlot.generate_model(model_func,
                                           p0, plot_ns)
moments.ModelPlot.plot_model(gen_mod,
                             save_file='model_from_GADMA.png',
                             fig_title='Demographic model from GADMA',
                             draw_scale=False,
                             pop_labels=['Pop 1', 'Pop 2', 'Pop 3', 'Pop 4'],
                             nref=None,
                             gen_time=1.0,
                             gen_time_units='generations',
                             reverse_timeline=True)