import moments.LD
import numpy as np
import os


def model_func(params, rho=None, theta=0.001):
	Nanc = 1.0
	s1, t1, nu11, nu12, m1_12, m1_21, s2, t2, nu21, nu22, nu23, m2_12, m2_13, m2_21, m2_23, m2_31, m2_32 = params
	Y = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
	Y = moments.LD.LDstats(Y, num_pops=1, pop_ids=['YRI', 'CEU', 'CHB'])
	Y = Y.split(0)
	nu1_func = lambda t: (s1 * Nanc) + (nu11 - (s1 * Nanc)) * (t / t1)
	migs = np.array([[0, m1_12], [m1_21, 0]])
	Y.integrate(tf=t1, nu=lambda t: [nu1_func(t), nu12], m=migs, rho=rho, theta=theta)
	Y = Y.split(1)
	nu1_func = lambda t: nu11 + (nu21 - nu11) * (t / t2)
	nu2_func = lambda t: (s2 * nu12) + (nu22 - (s2 * nu12)) * (t / t2)
	nu3_func = lambda t: ((1 - s2) * nu12) + (nu23 - ((1 - s2) * nu12)) * (t / t2)
	migs = np.array([[0, m2_12, m2_13], [m2_21, 0, m2_23], [m2_31, m2_32, 0]])
	Y.integrate(tf=t2, nu=lambda t: [nu1_func(t), nu2_func(t), nu3_func(t)], m=migs, rho=rho, theta=theta)
	return Y


rep_data_file = os.path.join(os.path.dirname(__file__), "preprocessed_data.bp")
opt_params = [0.2733400868162437, 0.14723719861251347, 1.645245624986071, 0.3301751304310405, 0.0, 2.05546966983028, 0.6167215133544224, 0.06665585004803394, 1.9120526315350221, 2.16933447355378, 1.448935418735779, 0.0, 1.4815737474560864, 0.1556756223072201, 0, 0.07768307959823452, 0.0, 6321]
rs = [0.0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001]
param_names = ['s1', 't1', 'nu11', 'nu12', 'm1_12', 'm1_21', 's2', 't2', 'nu21', 'nu22', 'nu23', 'm2_12', 'm2_13', 'm2_21', 'm2_23', 'm2_31', 'm2_32', 'Nanc']
