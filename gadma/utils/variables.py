import numpy as np
from .distributions import *
from .utils import extract_args
from functools import partial
from keyword import iskeyword


class Variable(object):
    '''
    Abstract class for keeping parameters of events in demographic model.
    Any new class for variable must be an instance of this class.

    New class should have :attr:`default_domain`, :attr:`default_rand_gen`
    class attributes and :func:`get_bounds`, :func:`get_possible_values`
    methods implemented.

    :cvar default_domain: default domain of the variable.
    :cvar default_rand_gen: default random generator of the variable. Is
                            used in :func:`resample` method.

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
        Retrurns list of all possible values of the variable.
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
    default_rand_gen = extract_args(np.random.uniform)

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if rand_gen is None:
            rand_gen = self.__class__.default_rand_gen
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


class DiscreteVariable(Variable):
    """
    Class of the discrete variable.

    :param domain: domain of the variable, if `None` then
                   :attr:`default_domain` will be taken.
    :param rand_gen: random generator for the variable, if `None` then
                     :attr:`default_rand_gen` will be taken.

    * :attr:`default_domain` = array([])

    * :attr:`default_rand_gen` = random choice over domain.
    """
    default_domain = np.array([])
    default_rand_gen = np.random.choice

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if rand_gen is None:
            rand_gen = self.__class__.default_rand_gen
        super(DiscreteVariable, self).__init__(
                    name, 'discrete', domain, rand_gen)

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


class PopulationSizeVariable(ContinuousVariable):
    """
    Variable for keepeing size of population in demographic model.

    * :attr:`default_domain` = array([1e-2, 100])

    * :attr:`default_rand_gen` = truncated log normal distribution over
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([1e-2, 100])
    default_rand_gen = extract_args(partial(trunc_lognormal_3_sigma_rule,
                                            1))


class MigrationVariable(ContinuousVariable):
    """
    Variable for keepeing migration parameter of the demographic model.

    * :attr:`default_domain` = array([0, 10])

    * :attr:`default_rand_gen` = truncated log normal distribution over
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([0, 10])
    default_rand_gen = extract_args(partial(trunc_normal_3_sigma_rule,
                                            1))


class TimeVariable(ContinuousVariable):
    """
    Variable for keepeing time parameter of the demographic model.

    * :attr:`default_domain` = array([0, 5])

    * :attr:`default_rand_gen` = truncated log normal distribution over
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([0, 5])
    default_rand_gen = extract_args(partial(trunc_normal_3_sigma_rule,
                                            1))


class SelectionVariable(ContinuousVariable):
    """
    Variable for keepeing selection parameter of the demographic model.

    * :attr:`default_domain` = array([0, 10])

    * :attr:`default_rand_gen` = truncated log normal distribution over
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([0, 10])
    default_rand_gen = extract_args(partial(trunc_normal_3_sigma_rule,
                                            1))


class PercentVariable(ContinuousVariable):
    """
    Variable for keepeing percent parameter of the demographic model.

    * :attr:`default_domain` = array([0, 1])

    * :attr:`default_rand_gen` = random uniform distribution over domain.
    """
    default_domain = np.array([0, 1])
    default_rand_gen = extract_args(np.random.uniform)


class Dynamic(object):
    """
    Abstract class for Dynamic value.
    New class should be instance of this class, should have :attr:`format_str`
    attribute and implement :func:`_inner_func` (static), :func:`__str__` and
    :func:`func_str` methods.

    :cvar format_str: format string for string representation of the dynamic.
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

    def func_str(self, y1, y2, x_diff):
        """
        Returns string representation of the dynamic via formating
        :attr:`format_str` with argumets.
        """
        return self.format_str % (str(y1), str(y2), str(x_diff))


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
        return lambda x: y1 * (y2 / y1) ** (x / x_diff)

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
        return lambda x: y1 + (y2 - y1) * (x / x_diff)

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
        return y2

    def __str__(self):
        """
        Returns "Sud".
        """
        return "Sud"


class DynamicVariable(DiscreteVariable):
    """
    Variable for keepeing selection parameter of the demographic model.

    * :attr:`default_domain` = array([\
        :class:`gadma.utils.variables.Exp`,\
        :class:`gadma.utils.variables.Lin`,\
        :class:`gadma.utils.variables.Sud`])

    * :attr:`default_rand_gen` = random choice over domain.
    """

    _help_dict = {'Sud': Sud, 'Lin': Lin, 'Exp': Exp}
    default_domain = np.array([Sud, Lin, Exp])

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if not all(dom in self.__class__.default_domain for dom in domain):
            raise Exception("Domain of DynamicVariable must be a subset "
                            "of the following general domain: %s"
                            % ", ".join([x.__name__
                                         for x in self.default_domain]))
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
            raise Exception("Value %s not in domain" % (str(value)))
        return DynamicVariable._help_dict[value]._inner_func

    def get_bounds(self):
        """
        :raises AttributeError: Dynamic variable has incomparative values.
        """
        raise AttributeError("DynamicVariable domain has incomparative values")
