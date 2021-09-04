import momi
import importlib.util

spec = importlib.util.spec_from_file_location('module', '../demographic_model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model_func = module.model_func


# Momi does not supports downsizing of the SFS so in this code there is no downsizing.
ind2pop = {'tsk_0': 'YRI', 'tsk_1': 'YRI', 'tsk_2': 'CEU', 'tsk_3': 'CEU', 'tsk_4': 'CHB', 'tsk_5': 'CHB'}
data = momi.SnpAlleleCounts.read_vcf('../out_of_africa_chr22_sim.vcf', ind2pop=ind2pop).extract_sfs(n_blocks=100)
data = data.subset_populations(['YRI', 'CEU', 'CHB'])

params = [15004.596619670217, 7223.225653867972, 270.24291209230444, 0.0690831942210091, 0.0009024496699107315, 1820.3282780957186, -0.000874306156876506, 360037.9623043344, 2424.316004053876, 36.219376382869015]
model = model_func(params)
model.gen_time = 1
model.muts_per_gen = 1.29e-08
model.set_data(data, length=51304566)
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
