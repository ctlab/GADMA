import moments
import numpy as np

def model_func(params, ns):
	t1, nu11, s1, t2, nu21, nu22, m2_12, m2_21 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	nu1_func = lambda t: 1.0 + (nu11 - 1.0) * (t / t1)
	fs.integrate(tf=t1, Npop=lambda t: [nu1_func(t)], dt_fac=0.01)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu2_func = lambda t: ((1 - s1) * nu11) * (nu22 / ((1 - s1) * nu11)) ** (t / t2)
	migs = np.array([[0, m2_12], [m2_21, 0]])
	fs.integrate(tf=t2, Npop=lambda t: [nu21, nu2_func(t)], m=migs, dt_fac=0.01)
	return fs

data = moments.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
ns = data.sample_sizes

p0 = [0.5116078590137683, 2.1904303576611808, 0.9655841896336654, 0.11392625625343339, 1.8148323026787092, 1.7524479743527377, 1.071321983150697, 0.8690522677357615]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = 7202.306804003384  # moments was not used for inference


plot_ns = [4 for _ in ns]  # small sizes for fast drawing
gen_mod = moments.ModelPlot.generate_model(model_func,
                                           p0, plot_ns)
moments.ModelPlot.plot_model(gen_mod,
                             save_file='model_from_GADMA.png',
                             fig_title='Demographic model from GADMA',
                             draw_scale=True,
                             pop_labels=['YRI', 'CEU'],
                             nref=7202,
                             gen_time=1.0,
                             gen_time_units='generations',
                             reverse_timeline=True)