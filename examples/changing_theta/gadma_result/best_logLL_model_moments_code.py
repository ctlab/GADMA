import moments
import numpy as np

def model_func(params, ns):
	t1, nu11, s1, t2, nu21, nu22, m2_12 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	fs.integrate(tf=t1, Npop=[nu11], dt_fac=0.01)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu1_func = lambda t: (s1 * nu11) * (nu21 / (s1 * nu11)) ** (t / t2)
	migs = np.array([[0, m2_12], [m2_12, 0]])
	fs.integrate(tf=t2, Npop=lambda t: [nu1_func(t), nu22], m=migs, dt_fac=0.01)
	return fs

data = moments.Spectrum.from_file('/home/katenos/Workspace/popgen/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['Pop 1', 'Pop 2']
ns = data.sample_sizes

p0 = [5.0, 1.6017267183354083, 0.10231452390998837, 1.9450055497773657, 2.817456387415982, 0.559720261493847, 1.3258648052909567]
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
                             pop_labels=['Pop 1', 'Pop 2'],
                             nref=None,
                             gen_time=1.0,
                             gen_time_units='generations',
                             reverse_timeline=True)