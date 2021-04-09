import dadi
import numpy as np

def model_func(params, ns, pts):
	t1, nu11, t2, nu21, s1, t3, nu31, nu32, m3_12, m3_21 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	nu1_func = lambda t: 1.0 + (nu11 - 1.0) * (t / t1)
	phi = dadi.Integration.one_pop(phi, xx, T=t1, nu=nu1_func)
	nu1_func = lambda t: nu11 * (nu21 / nu11) ** (t / t2)
	phi = dadi.Integration.one_pop(phi, xx, T=t2, nu=nu1_func)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu2_func = lambda t: ((1 - s1) * nu21) + (nu32 - ((1 - s1) * nu21)) * (t / t3)
	phi = dadi.Integration.two_pops(phi, xx, T=t3, nu1=nu31, nu2=nu2_func, m12=m3_12, m21=m3_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [1.1760789519903059, 1.2583197603211462, 1.6980490886604096, 1.2950706036135067, 0.15339939989687218, 0.2840682042555172, 2.463755800837137, 0.356804237476812, 0.0, 1.253787552971805]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = 6016.872817447757  # dadi was not used for inference
