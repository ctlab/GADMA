import numpy as np
from .utils import get_correct_dtype


def trunc_normal(mean, sigma, lower, upper):
    """
    Truncated normal distribution.
    """
    from scipy.stats import truncnorm
    if sigma == 0:
        sigma = 1e-15
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    result = truncnorm.rvs(a, b, loc=mean, scale=sigma)
    # fix additional list wrapper in scipy v1.5.0
    if isinstance(result, (list, np.ndarray)):
        assert len(result) == 1
        result = result[0]
    return result


def trunc_lognormal(mean, sigma, lower, upper):
    """
    Truncated log-normal distribution.
    """
    mean_plus_sigma = mean + sigma
    transf_mean = np.log(mean)
    transf_sigma = np.log(mean_plus_sigma) - transf_mean
    return np.exp(trunc_normal(transf_sigma, transf_sigma,
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
    """
    Uniform generator. Runs ``numpy.random.uniform`` on ``domain``.
    """
    return np.random.uniform(domain[0], domain[1])


def trunc_lognormal_sigma_generator(domain):
    """
    Generator for :func:`trunc_lognormal_3_sigma_rule`
    """
    lower = domain[0]
    if domain[0] == 0:
        lower = 1e-15
    mean = 1
    if mean < domain[0]:
        mean = domain[0]
    if mean > domain[1]:
        mean = domain[1]
    return trunc_lognormal_3_sigma_rule(mean, lower, domain[1])


def trunc_normal_sigma_generator(domain):
    """
    Generator for :func:`trunc_normal_3_sigma_rule`
    """
    mean = 1
    if mean < domain[0]:
        mean = domain[0]
    if mean > domain[1]:
        mean = domain[1]
    return trunc_normal_3_sigma_rule(mean, domain[0], domain[1])


def custom_generator(variables):
    """
    Custom generator for demographic model variables.
    """
    from .variables import PopulationSizeVariable, ContinuousVariable
    N_A = PopulationSizeVariable("__name").resample()
    values = list()
    for var in variables:
        x = var.resample()
        if var.log_transformed:
            x = np.exp(x)
        x = var._transform_value_from_gen_to_phys(value=x, Nanc=N_A)
        if var.log_transformed:
            x = np.log(x)
        values.append(x)
        if isinstance(var, ContinuousVariable):
            values[-1] = max(values[-1], var.domain[0])
            values[-1] = min(values[-1], var.domain[1])
    return np.array(values, dtype=get_correct_dtype(values))


def generator_for_Nanc(N_mean, domain):
    if N_mean is None:
        return np.random.uniform(*domain)
    return N_mean * trunc_lognormal_sigma_generator(
        [domain[0]/N_mean, domain[1]/N_mean]
    )


class DemographicGenerator:
    def __init__(self, var_cls, Nanc_domain, Nanc_mean,
                 combined_generator=True):
        self.var_cls = var_cls
        self.Nanc_domain = Nanc_domain
        self.Nanc_mean = Nanc_mean
        self.combined_generator = combined_generator

    def __call__(self, domain, *args, **kwargs):
        Nanc = generator_for_Nanc(self.Nanc_mean, self.Nanc_domain)
        if not self.combined_generator:
            return min(domain[1], max(domain[0], Nanc))
        value = self.var_cls.default_rand_gen.__call__(
            self.var_cls.default_domain
        )

        tr_value = self.var_cls._transform_value_from_gen_to_phys(value, Nanc)
        return min(domain[1], max(domain[0], tr_value))


def rescale_generator(generator, rescale_function):
    def wrap_generator(domain, *args, **kwargs):
        domain = [rescale_function(x, reverse=True) for x in domain]
        res = generator(np.array(domain), *args, **kwargs)
        return rescale_function(res, reverse=False)
    return wrap_generator

# def multiply_generator(gen1, domain1, gen2, domain2):
#     def generator(domain):
#         return gen1(domain1) * gen2(domain2)
#     return generator
