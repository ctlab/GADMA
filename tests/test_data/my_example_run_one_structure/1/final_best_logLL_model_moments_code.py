import moments
import numpy as np

def model_func(params, ns):
	t1, nu11, s1, t2, nu21, nu22, m2_12, m2_21 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	fs.integrate(tf=t1, Npop=[nu11], dt_fac=0.01)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu1_func = lambda t: (s1 * nu11) + (nu21 - (s1 * nu11)) * (t / t2)
	nu2_func = lambda t: ((1 - s1) * nu11) * (nu22 / ((1 - s1) * nu11)) ** (t / t2)
	migs = np.array([[0, m2_12], [m2_21, 0]])
	fs.integrate(tf=t2, Npop=lambda t: [nu1_func(t), nu2_func(t)], m=migs, dt_fac=0.01)
	return fs

data = moments.Spectrum.from_file('/home/katenos/Workspace/popgen/GADMA/fs_examples/YRI_CEU.fs')
ns = data.sample_sizes

p0 = [0.35429717440738756, 1.8867740900794858, 0.9596278258202917, 0.11419294254800488, 1.8681238930793145, 1.7366002336440358, 1.0401276126368213, 0.8645012452373468]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
theta0 = 0.37976
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))
