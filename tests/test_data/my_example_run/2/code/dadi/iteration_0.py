import dadi
import numpy as np

def model_func(params, ns, pts):
	s1, t1, nu11, nu12, m1_12, m1_21 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu1_func = lambda t: (s1 * 1.0) * (nu11 / (s1 * 1.0)) ** (t / t1)
	phi = dadi.Integration.two_pops(phi, xx, T=t1, nu1=nu1_func, nu2=nu12, m12=m1_12, m21=m1_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [0.460711025328101, 0.3868964003925894, 2.5398842587324215, 0.29155088973655907, 2.507654693052538, 0.0]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
theta0 = 0.37976
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))
