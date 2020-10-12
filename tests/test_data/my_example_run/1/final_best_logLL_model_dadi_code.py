import dadi
import numpy as np

def model_func(params, ns, pts):
	t1, nu11, s1, t2, nu21, nu22, m2_12, m2_21 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.Integration.one_pop(phi, xx, T=t1, nu=nu11)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu1_func = lambda t: (s1 * nu11) + (nu21 - (s1 * nu11)) * (t / t2)
	nu2_func = lambda t: ((1 - s1) * nu11) * (nu22 / ((1 - s1) * nu11)) ** (t / t2)
	phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu1_func, nu2=nu2_func, m12=m2_12, m21=m2_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/katenos/Workspace/popgen/GADMA/fs_examples/YRI_CEU.fs')
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [0.35429717440738756, 1.8867740900794858, 0.9596278258202917, 0.11419294254800488, 1.8681238930793145, 1.7366002336440358, 1.0401276126368213, 0.8645012452373468]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
theta0 = 0.37976
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))
