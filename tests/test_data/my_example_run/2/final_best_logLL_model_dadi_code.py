import dadi
import numpy as np

def model_func(params, ns, pts):
	t1, nu11, t2, nu21, s1, t3, nu31, nu32, m3_12, m3_21 = params
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.Integration.one_pop(phi, xx, T=t1, nu=nu11)
	phi = dadi.Integration.one_pop(phi, xx, T=t2, nu=nu21)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu1_func = lambda t: (s1 * nu21) * (nu31 / (s1 * nu21)) ** (t / t3)
	phi = dadi.Integration.two_pops(phi, xx, T=t3, nu1=nu1_func, nu2=nu32, m12=m3_12, m21=m3_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/katenos/Workspace/popgen/temp/GADMA/examples/changing_theta/YRI_CEU.fs')
data.pop_ids = ['YRI', 'CEU']
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [0.40091112473565726, 1.2991083662683578, 0.40091112473565726, 1.2991083662683578, 0.46626163289360073, 0.15438272830486668, 10.436235792300675, 0.2758858272665412, 0.0, 3.0802710775319566]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))
Nanc = 7310.141567530054  # dadi was not used for inference
