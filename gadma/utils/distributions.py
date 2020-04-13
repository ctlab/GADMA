import numpy as np

def trunc_normal(mean, sigma, lower, upper):
    from scipy.stats import truncnorm
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    return truncnorm.rvs(a, b, loc=mean, scale=sigma)


def trunc_lognormal(mean, sigma, lower, upper):
    return np.exp(trunc_normal(np.log(mean), np.log(sigma), np.log(lower), np.log(upper)))


def trunc_normal_3_sigma_rule(mean, lower, upper):
    '''Our truncated normal distribution with sigma equal to 3 sigma rule'''
    sigma = max(mean - lower, upper - mean) / 3
    return trunc_normal(mean, sigma, lower, upper)


def trunc_lognormal_3_sigma_rule(mean, lower, upper):
    return np.exp(trunc_normal_3_sigma_rule(np.log(mean), np.log(lower), np.log(upper)))

