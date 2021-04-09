import moments
import numpy as np

def model_func(params, ns):
	t1, nu11, t2, nu21, s1, t3, nu31, nu32, m3_12, m3_21 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	nu1_func = lambda t: 1.0 * (nu11 / 1.0) ** (t / t1)
	fs.integrate(tf=t1, Npop=lambda t: [nu1_func(t)], dt_fac=0.01)
	nu1_func = lambda t: nu11 * (nu21 / nu11) ** (t / t2)
	fs.integrate(tf=t2, Npop=lambda t: [nu1_func(t)], dt_fac=0.01)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu1_func = lambda t: (s1 * nu21) * (nu31 / (s1 * nu21)) ** (t / t3)
	migs = np.array([[0, m3_12], [m3_21, 0]])
	fs.integrate(tf=t3, Npop=lambda t: [nu1_func(t), nu32], m=migs, dt_fac=0.01)
	return fs

data = moments.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
ns = data.sample_sizes

p0 = [1.5533553013298524, 3.0263119402045597, 2.3382325588120585, 7.791940792879351, 0.7584966757708813, 3.584781881560943, 3.2807692902311305, 1.3579509748057264, 0.9058230420701527, 0.4010589938216259]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = 2557.1851516217007  # moments was not used for inference


plot_ns = [4 for _ in ns]  # small sizes for fast drawing
gen_mod = moments.ModelPlot.generate_model(model_func,
                                           p0, plot_ns)
moments.ModelPlot.plot_model(gen_mod,
                             save_file='model_from_GADMA.png',
                             fig_title='Demographic model from GADMA',
                             draw_scale=True,
                             pop_labels=['YRI', 'CEU'],
                             nref=2557,
                             gen_time=1.0,
                             gen_time_units='generations',
                             reverse_timeline=True)