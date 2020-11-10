import moments
import numpy as np

def model_func(params, ns):
	s1, t1, nu11, nu12, m1_12, m1_21 = params
	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
	fs = moments.Spectrum(sts)
	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
	nu2_func = lambda t: ((1 - s1) * 1.0) * (nu12 / ((1 - s1) * 1.0)) ** (t / t1)
	migs = np.array([[0, m1_12], [m1_21, 0]])
	fs.integrate(tf=t1, Npop=lambda t: [nu11, nu2_func(t)], m=migs, dt_fac=0.01)
	return fs

data = moments.Spectrum.from_file('/home/katenos/Workspace/popgen/GADMA/fs_examples/YRI_CEU.fs')
ns = data.sample_sizes

p0 = [0.9008754692131705, 0.27216732523595943, 1.5540619589423879, 1.3542114682523927, 0, 1.1434963765342256]
model = model_func(p0, ns)
ll_model = moments.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = moments.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
theta0 = 0.37976
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))
