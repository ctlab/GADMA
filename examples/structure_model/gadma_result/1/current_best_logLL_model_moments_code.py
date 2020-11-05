import moments
import numpy as np

def model_func(params, ns):
	t1, nu11, nu11_1, nu11_2, t2, nu21, nu22, m2_12, m2_21 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	fs.integrate(tf=t1, Npop=[nu11], dt_fac=0.01)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu1_func = lambda t: nu11_1 + (nu21 - nu11_1) * (t / t2)
	nu2_func = lambda t: nu11_2 * (nu22 / nu11_2) ** (t / t2)
	migs = np.array([[0, m2_12], [m2_21, 0]])
	fs.integrate(tf=t2, Npop=lambda t: [nu1_func(t), nu2_func(t)], m=migs, dt_fac=0.01)
	return fs

dd = moments.Misc.make_data_dict('dadi_2pops_CVLN_CVLS_snps.txt')
data = moments.Spectrum.from_data_dict(dd, ['CVLN', 'CVLS'], [10, 10], polarized=False)
ns = data.sample_sizes

p0 = [0.4650886021003231, 1.0, 0.02748197143994576, 0.4475852503882414, 1.0273860138296458, 6.156423614837759, 1.316658888151156, 0, 1.1556919215878465]
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
                             pop_labels=['CVLN', 'CVLS'],
                             nref=None,
                             gen_time=1.0,
                             gen_time_units='generations',
                             reverse_timeline=True)