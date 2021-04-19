import moments
import numpy as np

def model_func(params, ns):
	s1, t1, nu11, nu12, m1_12, m1_21 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu1_func = lambda t: (s1 * 1.0) * (nu11 / (s1 * 1.0)) ** (t / t1)
	migs = np.array([[0, m1_12], [m1_21, 0]])
	fs.integrate(tf=t1, Npop=lambda t: [nu1_func(t), nu12], m=migs, dt_fac=0.01)
	return fs

data = moments.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
ns = data.sample_sizes

p0 = [0.35134079394370255, 0.22018639448111757, 2.216873182572613, 0.1999881700478657, 0.0, 3.232386970089608]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
theta0 = 0.37976
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))


plot_ns = [4 for _ in ns]  # small sizes for fast drawing
gen_mod = moments.ModelPlot.generate_model(model_func,
                                           p0, plot_ns)
moments.ModelPlot.plot_model(gen_mod,
                             save_file='model_from_GADMA.png',
                             fig_title='Demographic model from GADMA',
                             draw_scale=False,
                             pop_labels=['YRI', 'CEU'],
                             nref=None,
                             gen_time=1.0,
                             gen_time_units='None',
                             reverse_timeline=True)