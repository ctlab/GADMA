import momi
import numpy as np

def model_func(params):
	nu21, nu22, t2, nu11, Nanc, t1 = params

	model = momi.DemographicModel(
		N_e=1,
		gen_time=1,
		muts_per_gen=2.35e-08
	)

	model.add_size_param('Nanc', lower=1.0, upper=100000000.0)
	model.add_time_param('t1', lower=0.2, upper=10000000.0)
	model.add_size_param('nu11', lower=1.0, upper=100000000.0)
	model.add_pulse_param('s1', lower=0.001, upper=0.999)
	model.add_time_param('t2', lower=0.2, upper=10000000.0)
	model.add_size_param('nu21', lower=1.0, upper=100000000.0)
	model.add_size_param('nu22', lower=1.0, upper=100000000.0)

	model.add_leaf(
		pop_name='YRI',
		t=0,
		N='nu21',
		g=0,
	)
	model.add_leaf(
		pop_name='CEU',
		t=0,
		N='nu22',
		g=0,
	)
	model.move_lineages(
		pop_from='CEU',
		pop_to='YRI',
		t='t2',
		p=1,
		N='nu11',
		g=lambda params: np.log(params.nu11 / params.Nanc) / params.t1,
	)
	model.set_size(
		pop_name='YRI',
		t=lambda params: params.t2 + params.t1,
		N='Nanc',
		g=0,
	)

	model.set_params({
		'nu21': nu21,
		'nu22': nu22,
		't2': t2,
		'nu11': nu11,
		'Nanc': Nanc,
		't1': t1,
	})
	return model


# Momi does not supports downsizing of the SFS so in this code there is no downsizing.
ind2pop = {'tsk_0': 'YRI', 'tsk_1': 'YRI', 'tsk_2': 'YRI', 'tsk_3': 'YRI', 'tsk_4': 'YRI', 'tsk_5': 'YRI', 'tsk_6': 'YRI', 'tsk_7': 'YRI', 'tsk_8': 'YRI', 'tsk_9': 'YRI', 'tsk_10': 'CEU', 'tsk_11': 'CEU', 'tsk_12': 'CEU', 'tsk_13': 'CEU', 'tsk_14': 'CEU', 'tsk_15': 'CEU', 'tsk_16': 'CEU', 'tsk_17': 'CEU', 'tsk_18': 'CEU', 'tsk_19': 'CEU'}
data = momi.SnpAlleleCounts.read_vcf('/home/stas/git/gadma_moments/data_for_documentation/data/YRI_CEU_sim_data.vcf', ind2pop=ind2pop).extract_sfs(n_blocks=100)
data = data.subset_populations(['YRI', 'CEU'])

params = [15732.236100953842, 3193.99783133937, 3666.04009293891, 1535.8389737576153, 7985.380946886594, 2844.140716204959]
model = model_func(params)
model.set_data(data, length=1000000)
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
