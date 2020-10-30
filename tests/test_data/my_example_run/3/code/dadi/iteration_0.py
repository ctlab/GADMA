import dadi
import numpy as np

def model_func(params, ns, pts):
	s1, t1, nu11, nu12, m1_12, m1_21 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu2_func = lambda t: ((1 - s1) * 1.0) + (nu12 - ((1 - s1) * 1.0)) * (t / t1)
	phi = dadi.Integration.two_pops(phi, xx, T=t1, nu1=nu11, nu2=nu2_func, m12=m1_12, m21=m1_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/katenos/Workspace/popgen/GADMA/fs_examples/YRI_CEU.fs')
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [0.09200137889929888, 0.30099757687879214, 1.1640124094525566, 4.181058071544814, 0, 1.2390776942547495]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
theta0 = 0.37976
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))
