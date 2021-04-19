import moments
import numpy as np

def model_func(params, ns):
	t1, nu11, t2, nu21, s1, t3, nu31, nu32, m3_12, m3_21 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	fs.integrate(tf=t1, Npop=[nu11], dt_fac=0.01)
	fs.integrate(tf=t2, Npop=[nu21], dt_fac=0.01)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu1_func = lambda t: (s1 * nu21) * (nu31 / (s1 * nu21)) ** (t / t3)
	migs = np.array([[0, m3_12], [m3_21, 0]])
	fs.integrate(tf=t3, Npop=lambda t: [nu1_func(t), nu32], m=migs, dt_fac=0.01)
	return fs

data = moments.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
ns = data.sample_sizes

p0 = [0.40091112473565726, 1.2991083662683578, 0.40091112473565726, 1.2991083662683578, 0.46626163289360073, 0.15438272830486668, 10.436235792300675, 0.2758858272665412, 0.0, 3.0802710775319566]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = 7310.141567530054  # moments was not used for inference


plot_ns = [4 for _ in ns]  # small sizes for fast drawing
gen_mod = moments.ModelPlot.generate_model(model_func,
                                           p0, plot_ns)
moments.ModelPlot.plot_model(gen_mod,
                             save_file='model_from_GADMA.png',
                             fig_title='Demographic model from GADMA',
                             draw_scale=True,
                             pop_labels=['YRI', 'CEU'],
                             nref=7310,
                             gen_time=1.0,
                             gen_time_units='generations',
                             reverse_timeline=True)