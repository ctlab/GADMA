import dadi
import numpy as np

def model_func(params, ns, pts):
	t1, nu11, s1, t2, nu21, nu22, m2_12, m2_21 = params
	_Nanc_size = 1.0  # This value can be used in splits with fractions
	xx = dadi.Numerics.default_grid(pts)
	phi = dadi.PhiManip.phi_1D(xx)
	phi = dadi.Integration.one_pop(phi, xx, T=t1, nu=nu11)
	phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
	nu1_func = lambda t: (s1 * nu11) + (nu21 - (s1 * nu11)) * (t / t2)
	nu2_func = lambda t: ((1 - s1) * nu11) * (nu22 / ((1 - s1) * nu11)) ** (t / t2)
	phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu1_func, nu2=nu2_func, m12=m2_12, m21=m2_21)
	sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
	return sfs

data = dadi.Spectrum.from_file('/home/enoskova/Workspace/noscode_GADMA/ML_tests/data/2_BotDivMig_8_Sim/fs_data.fs')
data.pop_ids = ['Pop 1', 'Pop 2']
pts = [20, 30, 40]
ns = data.sample_sizes

p0 = [3.9900000000000295, 0.2433385564519036, 0.36407236209241545, 0.7932321097551149, 1.9176901031961817, 3.0618185008033563, 1.1765791914075407, 0.0010002450196909402]
func_ex = dadi.Numerics.make_extrap_log_func(model_func)
model = func_ex(p0, ns, pts)
ll_model = dadi.Inference.ll_multinom(model, data)
print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

theta = dadi.Inference.optimal_sfs_scaling(model, data)
print('Optimal value of theta: {0}'.format(theta))

# As no theta0 or mut. rate + seq. length are not set
theta0 = 1.0
Nanc = int(theta / theta0)
print('Size of ancestral population: {0}'.format(Nanc))
