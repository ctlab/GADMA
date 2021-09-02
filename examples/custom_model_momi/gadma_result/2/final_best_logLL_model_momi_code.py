import momi
import importlib.util

spec = importlib.util.spec_from_file_location('module', '/home/katenos/Workspace/popgen/temp/GADMA/examples/custom_model_momi/demographic_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model_func = module.model_func


# Momi does not supports downsizing of the SFS so in this code there is no downsizing.
data = momi.sfs_from_dadi('/home/katenos/Workspace/popgen/temp/GADMA/examples/custom_model_momi/YRI_CEU.fs')
data = data.subset_populations(['YRI', 'CEU'])

params = [102726.06351515892, 59543.758923395275, 0.001, 3.986264525061683, 7929.578722814034, 5272.715387420992]
model = model_func(params)
model.gen_time = 1
model.muts_per_gen = 2.35e-08
model.set_data(data, length=4040000)
ll_model = model.log_likelihood()
print(f'Value of log-likelihood: {ll_model}')
from matplotlib import pyplot as plt
momi.DemographyPlot(
	model,
	pop_x_positions=data.sampled_pops,
	figsize=(6,8),
	linthreshy=None,
	pulse_color_bounds=(0,.25)
)
plt.savefig('model_from_GADMA.png')
