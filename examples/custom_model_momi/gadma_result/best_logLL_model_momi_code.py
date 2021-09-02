import momi
import importlib.util

spec = importlib.util.spec_from_file_location('module', '/home/katenos/Workspace/popgen/temp/GADMA/examples/custom_model_momi/demographic_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model_func = module.model_func


# Momi does not supports downsizing of the SFS so in this code there is no downsizing.
data = momi.sfs_from_dadi('/home/katenos/Workspace/popgen/temp/GADMA/examples/custom_model_momi/YRI_CEU.fs')
data = data.subset_populations(['YRI', 'CEU'])

params = [109.36454428269073, 256330.26580715246, 0.0008948579139255634, 4.7288093065137735, 1402947.4995631143, 159.1072306368505]
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
