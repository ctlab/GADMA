import moments.LD
import numpy as np
import os

def model_func(params, rho=None, theta=0.001):
	t1, nu11, nu12, m1_12 = params
	Y = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
	Y = moments.LD.LDstats(Y, num_pops=1, pop_ids=['deme0', 'deme1'])
	Y = Y.split(0)
	migs = np.array([[0, m1_12], [m1_12, 0]])
	Y.integrate(tf=t1, nu=[nu11, nu12], m=migs, rho=rho, theta=theta)
	return Y


rep_data_file = os.path.join(os.path.dirname(__file__), "preprocessed_data.bp")
opt_params = [0.07224855484916312, 1.0536252967417892, 2.2336162670631126, 0.9291254115918008, 10773]
rs = [0.0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001]
param_names = ['t1', 'nu11', 'nu12', 'm1_12', 'Nanc']
