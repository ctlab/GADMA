import dadi
import numpy as np

def model_func(params, ns, pts):
	t1, nu11, nu11_1, nu11_2, t2, nu21, nu22, m2_12, m2_21 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.Integration.one_pop(phi, xx, T=t1, nu=nu11)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu1_func = lambda t: nu11_1 + (nu21 - nu11_1) * (t / t2)
	nu2_func = lambda t: nu11_2 * (nu22 / nu11_2) ** (t / t2)
	phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu1_func, nu2=nu2_func, m12=m2_12, m21=m2_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

dd = dadi.Misc.make_data_dict('dadi_2pops_CVLN_CVLS_snps.txt')
data = dadi.Spectrum.from_data_dict(dd, ['CVLN', 'CVLS'], [10, 10], polarized=False)
pts = [30, 40, 50]
ns = data.sample_sizes

p0 = [0.4650886021003231, 1.0, 0.02748197143994576, 0.4475852503882414, 1.0273860138296458, 6.156423614837759, 1.316658888151156, 0, 1.1556919215878465]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = None
