import numpy as np
from .distributions import uniform_generator, trunc_normal_sigma_generator,\
                           trunc_lognormal_sigma_generator
from functools import partial
from keyword import iskeyword


class Variable(object):
    '''
    Abstract class for keeping parameters of events in demographic model.
    Any new class for variable must be an instance of this class.

    New class should have :attr:`default_domain`, :attr:`default_rand_gen`
    class attributes and :func:`get_bounds`, :func:`get_possible_values`
    methods implemented.

    :cvar Variable.default_domain: default domain of the variable.
    :cvar Variable.default_rand_gen: default random generator of the variable.
                                     Is used in :func:`resample` method.

    :param name: unique name of the variable.
    :type name: str
    :param var_type: type of the variable (usually 'continuous'
                     or 'discrete').
    :type var_type: str
    :param domain: domain of the variable.
    :param rand_gen: random generator of the variable, should be a function
                     that takes domain as argument and returns sampled value.
    '''
    def __init__(self, name, var_type, domain, rand_gen):
        self.is_active = True
        self.name = name
        # TODO The following assert does not work for python 2
        assert(name.isidentifier() and not iskeyword(name))
        self.var_type = var_type
        self.domain = domain
        self.rand_gen = rand_gen

    def __copy__(self):
        return self

    def resample(self, *args, **kwargs):
        """
        Returns sampled value of the variable by calling `rand_gen` on
        the `domain`.

        :param `*args`: arguments to pass in `rand_gen`.
        :param `**kwargs`: kwargs to pass in `rand_gen`.
        """
        return self.rand_gen.__call__(self.domain, *args, **kwargs)

    def __str__(self):
        return "%s %s" % (self.__class__.__name__, self.name)

    def __repr__(self):
        return "%s %s" % (self.__class__.__name__, self.name)

    def get_bounds(self):
        """
        Returns bounds of the variable domain.
        """
        raise NotImplementedError

    def get_possible_values(self):
        """
        Returns list of all possible values of the variable.
        """
        raise NotImplementedError

    def correct_value(self, value):
        """
        Check that value is correct for this variable.
        """
        raise NotImplementedError


class ContinuousVariable(Variable):
    """
    Class of the continuous variable.

    :param domain: domain of the variable, if `None` then
                   :attr:`default_domain` will be taken.
    :param rand_gen: random generator for the variable, if `None` then
                     :attr:`default_rand_gen` will be taken.

    * :attr:`default_domain` = array([-inf, inf])

    * :attr:`default_rand_gen` = uniform distribution over domain.
    """
    default_domain = np.array([-np.inf, np.inf])
    default_rand_gen = uniform_generator

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if rand_gen is None:
            rand_gen = self.__class__.default_rand_gen
        if domain[0] > domain[1]:
            raise ValueError("The lower bound of variable's domain must be "
                             "greater than upper bound. Got domain: "
                             f"{domain}.")
        super(ContinuousVariable, self).__init__(name, 'continuous',
                                                 domain, rand_gen)

    def get_bounds(self):
        """
        Returns the domain of the variable.
        """
        return self.domain

    def get_possible_values(self):
        """
        :raises AttributeError: it is impossible to get possible values\
        for continuous variable.
        """
        raise AttributeError("Impossible to produce a list of values for"
                             " continuous variable " + self.name)

    def correct_value(self, value):
        """
        Check that value is correct for this variable.
        """
        return value >= self.domain[0] and value <= self.domain[1]


class DiscreteVariable(Variable):
    """
    Class of the discrete variable.

    :param domain: domain of the variable, if `None` then
                   :attr:`default_domain` will be taken.
    :param rand_gen: random generator for the variable, if `None` then
                     :attr:`default_rand_gen` will be taken.

    """
    default_domain = np.array([])

    def default_rand_gen(a, size=None, replace=True, p=None):
        """
        See documentation of ``numpy.random.choice`` for more information.
        """
        return np.random.choice(a, size, replace, p)

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if rand_gen is None:
            rand_gen = self.__class__.default_rand_gen
        super(DiscreteVariable, self).__init__(
                    name, 'discrete', domain, rand_gen)

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        self._domain = np.array(domain, dtype=object)

    def get_bounds(self):
        """
        Returns bounds - minimum and maximum over domain of the variable.
        """
        return [min(self.domain), max(self.domain)]

    def get_possible_values(self):
        """
        Returns domain of the variable.
        """
        return self.domain

    def correct_value(self, value):
        """
        Check that value is correct for this variable.
        """
        return value in self.domain


class PopulationSizeVariable(ContinuousVariable):
    """
    Variable for keepeing size of population in demographic model.

    * :attr:`default_domain` = array([1e-2, 100])

    * :attr:`default_rand_gen` = truncated log normal distribution over\
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([1e-2, 100])
    default_rand_gen = trunc_lognormal_sigma_generator

    @staticmethod
    def translate_units(value, Nanc):
        return type(Nanc)(value * Nanc)


def migration_generator(domain):
    """
    Generates random value of the migration. With probability of 0.5 generates
    zero, otherwise runs
    :func:`gadma.utils.distributions.trunc_normal_sigma_generator`.
    """
    if 0 == domain[0] and np.random.choice([False, True]):
        return 0
    return trunc_normal_sigma_generator(domain)


class MigrationVariable(ContinuousVariable):
    """
    Variable for keepeing migration parameter of the demographic model.

    * :attr:`default_domain` = array([0, 10])

    * :attr:`default_rand_gen` = truncated log normal distribution over\
      domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([0, 10])
    default_rand_gen = migration_generator

    @staticmethod
    def translate_units(value, Nanc):
        return value / (2 * Nanc)


class TimeVariable(ContinuousVariable):
    """
    Variable for keepeing time parameter of the demographic model.

    * :attr:`default_domain` = array([0, 5])

    * :attr:`default_rand_gen` = truncated log normal distribution over\
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([1e-15, 5])
    default_rand_gen = trunc_normal_sigma_generator

    @staticmethod
    def translate_units(value, Nanc):
        return type(Nanc)(value * (2 * Nanc))


class SelectionVariable(ContinuousVariable):
    """
    Variable for keepeing selection parameter of the demographic model.

    * :attr:`default_domain` = array([0, 10])

    * :attr:`default_rand_gen` = truncated log normal distribution over\
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([1e-15, 10])
    default_rand_gen = trunc_normal_sigma_generator

    @staticmethod
    def translate_units(value, Nanc):
        return value / (2 * Nanc)


class FractionVariable(ContinuousVariable):
    """
    Variable for keepeing fraction parameter of the demographic model.

    * :attr:`default_domain` = array([0, 1])

    * :attr:`default_rand_gen` = random uniform distribution over domain.
    """
    default_domain = np.array([1e-3, 1-1e-3])
    default_rand_gen = uniform_generator

    @staticmethod
    def translate_units(value, Nanc):
        return value


class Dynamic(object):
    """
    Abstract class for Dynamic value.
    New class should be instance of this class, should have :attr:`format_str`
    attribute and implement :func:`_inner_func` (staticmethod),
    :func:`__str__` and :func:`func_str` methods.

    :cvar Dynamic.format_str: format string for string representation of the
                              dynamic.
    """
    format_str = ''

    @staticmethod
    def _inner_func(y1, y2, x_diff):
        """
        Inner function of the dynamic.

        :param y1: the initial value of the dynamic.
        :param y2: the final value of the dynamic.
        :param x_diff: the x-difference between initial and final values.
        """
        raise NotImplementedError

    def __str__(self):
        """
        Returns string representation of the dynamic.
        """
        raise NotImplementedError

    @classmethod
    def func_str(cls, y1, y2, x_diff):
        """
        Returns string representation of the dynamic via formating
        :attr:`format_str` with argumets.
        """
        return cls.format_str.format(str(y1), str(y2), str(x_diff))


def linear(y1, y2, x_diff, x):
    y = y1 + (y2 - y1) * (x / x_diff)
    return y


def exponent(y1, y2, x_diff, x):
    y = y1 * (y2 / y1) ** (x / x_diff)
    return y


def constant(y1, y2, x_diff, x):
    return y2


class Exp(Dynamic):
    """
    Exponential dynamic.

    * :attr:`format_str` = "lambda t: {0} * ({1} / {0}) ** (t / {2})"
    """
    format_str = "lambda t: {0} * ({1} / {0}) ** (t / {2})"

    @staticmethod
    def _inner_func(y1, y2, x_diff):
        """
        Returns lambda function of x: y1 * (y2 / y1) ** (x / x_diff).
        """
        return partial(exponent, y1, y2, x_diff)

    def __str__(self):
        """
        Returns "Exp".
        """
        return "Exp"


class Lin(Dynamic):
    """
    Linear dynamic.

    * :attr:`format_str` = "lambda t: {0} + ({1} - {0}) * (t / {2})"
    """
    format_str = "lambda t: {0} + ({1} - {0}) * (t / {2})"

    @staticmethod
    def _inner_func(y1, y2, x_diff):
        """
        Returns lambda function of x: y1 + (y2 - y1) * (x / x_diff).
        """
        return partial(linear, y1, y2, x_diff)

    def __str__(self):
        """
        Returns "Lin".
        """
        return "Lin"


class Sud(Dynamic):
    """
    Sudden (constant) dynamic.

    * :attr:`format_str` = "{1}"
    """
    format_str = '{1}'

    @staticmethod
    def _inner_func(y1, y2, x_diff):
        """
        Returns y2.
        """
        return partial(constant, y1, y2, x_diff)

    def __str__(self):
        """
        Returns "Sud".
        """
        return "Sud"


def dynamic_generator(domain):
    p = []
    for x in domain:
        if x == 'Sud' or x == 0:
            p.append(len(domain) + 1)
        elif x == 'Lin' or x == 1:
            p.append(len(domain) - 1)
        else:
            p.append(len(domain))
    p = np.array(p) / np.sum(p)
    return np.random.choice(domain, p=p)


class DynamicVariable(DiscreteVariable):
    """
    Variable for keepeing selection parameter of the demographic model.

    * :attr:`default_domain` = array([\
        :class:`gadma.utils.variables.Exp`,\
        :class:`gadma.utils.variables.Lin`,\
        :class:`gadma.utils.variables.Sud`])

    * :attr:`default_rand_gen` = random choice over domain.
    """

    _help_dict = {'Sud': Sud, 'Lin': Lin, 'Exp': Exp, 0: Sud, 1: Lin, 2: Exp}
    default_domain = np.array(['Sud', 'Lin', 'Exp'])
    default_rand_gen = dynamic_generator

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if not all(dom in self.__class__._help_dict for dom in domain):
            raise ValueError("Domain of DynamicVariable must be a subset "
                             "of the following general domain: "
                             f"{self._help_dict.keys()}")
        super(DynamicVariable, self).__init__(name, domain, rand_gen)

    @staticmethod
    def get_func_from_value(value):
        """
        Returns :func:`gadma.utils.variables.Dynamic._inner_func` function
        from the value.

        :param value: value of the variable.
        :type value: :class:`gadma.utils.variables.Dynamic`
        """
        if value not in DynamicVariable._help_dict:
            raise ValueError(f"Value {value} not in domain:"
                             f"{DynamicVariable._help_dict.keys()}.")
        return DynamicVariable._help_dict[value]._inner_func

    def get_bounds(self):
        """
        :raises AttributeError: Dynamic variable has incomparative values.
        """
        raise AttributeError("DynamicVariable domain has incomparative values")

    @staticmethod
    def translate_units(value, Nanc):
        return value
