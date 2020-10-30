import dadi
import numpy as np

def model_func(params, ns, pts):
	t1, nu11, s1, t2, nu21, nu22, m2_12 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.Integration.one_pop(phi, xx, T=t1, nu=nu11)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu1_func = lambda t: (s1 * nu11) * (nu21 / (s1 * nu11)) ** (t / t2)
	phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu1_func, nu2=nu22, m12=m2_12, m21=m2_12)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/katenos/Workspace/popgen/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['Pop 1', 'Pop 2']
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [5.0, 1.6017267183354083, 0.10231452390998837, 1.9450055497773657, 2.817456387415982, 0.559720261493847, 1.3258648052909567]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = None
