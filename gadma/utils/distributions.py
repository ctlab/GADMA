import numpy as np


def trunc_normal(mean, sigma, lower, upper):
    """
    Truncated normal distribution.
    """
    from scipy.stats import truncnorm
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    return truncnorm.rvs(a, b, loc=mean, scale=sigma)


def trunc_lognormal(mean, sigma, lower, upper):
    """
    Truncated log-normal distribution.
    """
    return np.exp(trunc_normal(np.log(mean), np.log(sigma),
                               np.log(lower), np.log(upper)))


def trunc_normal_3_sigma_rule(mean, lower, upper):
    """
    Truncated normal distribution with sigma according by
    three sigma rule.
    """
    sigma = max(mean - lower, upper - mean) / 3
    return trunc_normal(mean, sigma, lower, upper)


def trunc_lognormal_3_sigma_rule(mean, lower, upper):
    """
    Truncated log-normal distribution with sigma according by
    three sigma rule.
    """
    return np.exp(trunc_normal_3_sigma_rule(np.log(mean), np.log(lower),
                                            np.log(upper)))


# Random generators
def uniform_generator(domain):
    return np.random.uniform(domain[0], domain[1])


def trunc_lognormal_sigma_generator(domain):
    return trunc_lognormal_3_sigma_rule(1, domain[0], domain[1])


def trunc_normal_sigma_generator(domain):
    return trunc_normal_3_sigma_rule(1, domain[0], domain[1])


def custom_generator(variables):
    from .variables import PopulationSizeVariable, ContinuousVariable
    N_A = PopulationSizeVariable("__name").resample()
    values = list()
    for var in variables:
        x = var.resample()
        values.append(var.translate_units(x, N_A))
        if isinstance(var, ContinuousVariable):
            values[-1] = max(values[-1], var.domain[0])
            values[-1] = min(values[-1], var.domain[1])
    return np.array(values, dtype=object)

def multiply_generator(gen1, domain1, gen2, domain2):
    def generator(domain):
        return gen1(domain1) * gen2(domain2)
    return generator
