import moments.LD
import numpy as np

def model_func(params, rho=None, theta=0.001):
    t1, nu11, s1, t2, nu21, nu22, m2_12 = params
    Nanc = 1.0 #This value can be used in splits with fraction variable
    Y = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
    Y = moments.LD.LDstats(Y, num_pops=1, pop_ids=['YRI', 'CEU'])
    nu1_func = lambda t: Nanc * (nu11 / Nanc) ** (t / t1)
    Y.integrate(tf=t1, nu=lambda t: [nu1_func(t)], rho=rho, theta=theta)
    Y = Y.split(0)
    nu2_func = lambda t: ((1 - s1) * nu11) + (nu22 - ((1 - s1) * nu11)) * (t / t2)
    migs = np.array([[0, m2_12], [m2_12, 0]])
    Y.integrate(tf=t2, nu=lambda t: [nu21, nu2_func(t)], m=migs, rho=rho, theta=theta)
    return Y

rep_data_file = "/home/stas/git/gadma_moments/data_for_documentation/data/preprocessed_data.bp"
opt_params = [0.18154684643488384, 0.3552140446773699, 0.7544814289189229, 0.14035487862181226, 1.9894358260758085, 0.715539980127665, 1.0355376622898436, 7729]
rs = [0.0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001]
param_names = ['t1', 'nu11', 's1', 't2', 'nu21', 'nu22', 'm2_12', 'Nanc']