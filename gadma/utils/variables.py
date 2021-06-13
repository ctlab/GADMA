import numpy as np
from .distributions import uniform_generator, trunc_normal_sigma_generator, \
    trunc_lognormal_sigma_generator, \
    DemographicGenerator, rescale_generator
from functools import partial
from keyword import iskeyword
import copy


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
    default_domain = None
    default_rand_gen = None

    def __init__(self, name, var_type, domain, rand_gen):
        self.is_active = True
        self.name = name
        # TODO The following assert does not work for python 2
        assert (name.isidentifier() and not iskeyword(name))
        if domain is None:
            domain = copy.deepcopy(self.__class__.default_domain)
        if rand_gen is None:
            rand_gen = copy.deepcopy(self.__class__.default_rand_gen)
        self.var_type = var_type
        self.domain = domain
        self.rand_gen = rand_gen
        self._log_transformed = False

    @property
    def log_transformed(self):
        return self._log_transformed

    @log_transformed.setter
    def log_transformed(self, new_value):
        if self._log_transformed == new_value:
            return
        assert isinstance(new_value, bool)
        if new_value:
            self.apply_logarithm()
        else:
            self.apply_logarithm(back=True)
        self._log_transformed = new_value

    def apply_logarithm(self, back=False):
        raise NotImplementedError

    def __copy__(self):
        return self

    def resample(self, *args, **kwargs):
        """
        Returns sampled value of the variable by calling `rand_gen` on
        the `domain`.

        :param `*args`: arguments to pass in `rand_gen`.
        :param `**kwargs`: kwargs to pass in `rand_gen`.
        """
        if self.log_transformed:
            return np.log(self.rand_gen.__call__(np.exp(self.domain),
                                                 *args,
                                                 **kwargs))
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
        super(ContinuousVariable, self).__init__(name, 'continuous',
                                                 domain, rand_gen)
        if self.domain[0] > self.domain[1]:
            raise ValueError("The lower bound of variable's domain must be "
                             "greater than upper bound. Got domain: "
                             f"{self.domain}.")
        # for logarithm transform
        self._true_domain = self.domain

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
        return self.domain[0] <= value <= self.domain[1]

    def apply_logarithm(self, back=False):
        """
        Applies logarithm transform to the variable. Domain and random
        generator are changed.
        """
        if back:
            self.domain = self._true_domain
        else:
            self._true_domain = self.domain
            self.domain = np.log(self.domain).tolist()


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
            name, var_type='discrete', domain=domain, rand_gen=rand_gen)

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


class DemographicVariable(Variable):
    """
    Class for demographic variables. Keeps units of itself and translates them.
    Units could be "physical" (number of individuals, generations),
    "genetic" (scaled to size of ancestral population Nanc) or "universal"
    (different proportions, fractions or dynamics of pop size).

    :note: default_domain is set in genetic units. But it is translated if\
           needed
    """
    # domain of Nanc in physical units. It is needed to translate
    # default_domain to physical units
    default_domain_in_phys = np.array([100, 1e6])

    def __init__(self, name, units="genetic", domain=None, rand_gen=None):
        if units != "physical" and units != "genetic" and units != "universal":
            raise ValueError(f"Units {units} is incorrect")
        if units == "universal":
            self.units = units
        else:
            self.units = "genetic"
        if isinstance(self, (ContinuousVariable, DiscreteVariable)):
            super(DemographicVariable, self).__init__(name, domain, rand_gen)
        else:
            var_type = "unknown"
            super(DemographicVariable, self).__init__(name, var_type,
                                                      domain, rand_gen)

        self.translate_units_to(units, self.default_domain_in_phys)

    @staticmethod
    def _transform_value_from_gen_to_phys(value, Nanc):
        raise NotImplementedError

    @staticmethod
    def _transform_value_from_phys_to_gen(value, Nanc):
        raise NotImplementedError

    def translate_value_into(self, units, value, Nanc=None):
        if not self.correct_value(value):
            raise ValueError("Given value is not correct: "
                             f"value {value} not in domain {self.domain}.")
        if self.units == "universal":
            return value
        if units == self.units:
            return value
        if Nanc is None:
            raise ValueError("Set Nanc value for translation")
        if units == "physical":
            return self._transform_value_from_gen_to_phys(value, Nanc)
        elif units == "genetic":
            return self._transform_value_from_phys_to_gen(value, Nanc)
        else:
            raise ValueError(f"Units {units} is incorrect")

    def translate_units_to(self, units, Nanc_domain=None):
        if units == self.units:
            return
        if self.units == "universal":
            return
        if Nanc_domain is None:
            Nanc_domain = list(self.default_domain_in_phys)
        domain = []
        domain.append(
            self.translate_value_into(
                units,
                self.domain[0],
                Nanc_domain[0]
            )
        )
        domain.append(
            self.translate_value_into(
                units,
                self.domain[1],
                Nanc_domain[1],
            )
        )
        self.domain = domain
        self.units = units
        if units == "physical":
            self.rand_gen = DemographicGenerator(
                self.rand_gen,
                Nanc_domain,
            )
        elif units == "genetic":
            self.rand_gen = self.rand_gen.genetic_generator

    def rescale(self, Nref, reverse=False):
        """
        Rescales variable domain and rand_gen by factor of Nref.
        Rescaling is valid if units are physical and rescaling is done by
        translation to genetic units of domain.
        """
        if Nref is None:
            return
        if self.units == "genetic":
            return
        base_reverse = reverse

        def rescale_func(value, reverse=False):
            if base_reverse:
                reverse = not reverse
            return self.rescale_value(value=value,
                                      Nref=Nref,
                                      reverse=reverse)
        self.rand_gen = rescale_generator(self.rand_gen,
                                          rescale_function=rescale_func)
        self.domain[0] = self.rescale_value(self.domain[0], Nref,
                                            reverse=reverse)
        self.domain[1] = self.rescale_value(self.domain[1], Nref,
                                            reverse=reverse)

    def rescale_value(self, value, Nref, reverse=False):
        """
        Rescales value by factor of Nref.
        """
        if reverse:
            return self._transform_value_from_gen_to_phys(value=value,
                                                          Nanc=Nref)
        return self._transform_value_from_phys_to_gen(value=value,
                                                      Nanc=Nref)


class PopulationSizeVariable(DemographicVariable, ContinuousVariable):
    """
    Variable for keepeing size of population in demographic model.

    * :attr:`default_domain` = array([1e-2, 100])

    * :attr:`default_rand_gen` = truncated log normal distribution over\
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([1e-2, 100])  # in genetic units
    default_rand_gen = trunc_lognormal_sigma_generator

    @staticmethod
    def _transform_value_from_gen_to_phys(value, Nanc):
        return Nanc * value

    @staticmethod
    def _transform_value_from_phys_to_gen(value, Nanc):
        return value / Nanc


def migration_generator(domain):
    """
    Generates random value of the migration. With probability of 0.5 generates
    zero, otherwise runs
    :func:`gadma.utils.distributions.trunc_normal_sigma_generator`.
    """
    if 0 == domain[0] and np.random.choice([False, True]):
        return 0
    return trunc_normal_sigma_generator(domain)


class MigrationVariable(DemographicVariable, ContinuousVariable):
    """
    Variable for keepeing migration parameter of the demographic model.

    * :attr:`default_domain` = array([0, 10])

    * :attr:`default_rand_gen` = truncated log normal distribution over\
      domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([0, 10.0])
    default_rand_gen = migration_generator

    @staticmethod
    def _transform_value_from_gen_to_phys(value, Nanc):
        return float(value) / (2 * Nanc)

    @staticmethod
    def _transform_value_from_phys_to_gen(value, Nanc):
        return 2 * Nanc * value


class TimeVariable(DemographicVariable, ContinuousVariable):
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
    def _transform_value_from_gen_to_phys(value, Nanc):
        return 2 * Nanc * value

    @staticmethod
    def _transform_value_from_phys_to_gen(value, Nanc):
        return value / (2 * Nanc)


class SelectionVariable(DemographicVariable, ContinuousVariable):
    """
    Variable for keepeing selection parameter of the demographic model.

    * :attr:`default_domain` = array([0, 10])

    * :attr:`default_rand_gen` = truncated log normal distribution over\
        domain with mean equal to 1.

    :note: Values are assumed to be in genetic units.
    """
    default_domain = np.array([0, 10])
    default_rand_gen = trunc_normal_sigma_generator

    @staticmethod
    def _transform_value_from_gen_to_phys(value, Nanc):
        return value / (2 * Nanc)

    @staticmethod
    def _transform_value_from_phys_to_gen(value, Nanc):
        return 2 * Nanc * value


class FractionVariable(DemographicVariable, ContinuousVariable):
    """
    Variable for keepeing fraction parameter of the demographic model.

    * :attr:`default_domain` = array([0, 1])

    * :attr:`default_rand_gen` = random uniform distribution over domain.
    """
    default_domain = np.array([1e-3, 1 - 1e-3])
    default_rand_gen = uniform_generator
    units = "universal"

    def __init__(self, name, domain=None, rand_gen=None):
        super(FractionVariable, self).__init__(name,
                                               units="universal",
                                               domain=domain,
                                               rand_gen=rand_gen)

    @staticmethod
    def _transform_value_from_gen_to_phys(value, Nanc):
        return value

    @staticmethod
    def _transform_value_from_phys_to_gen(value, Nanc):
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


class DynamicVariable(DemographicVariable, DiscreteVariable):
    """
    Variable for keepeing selection parameter of the demographic model.

    * :attr:`default_domain` = array([\
        :class:`gadma.utils.variables.Exp`,\
        :class:`gadma.utils.variables.Lin`,\
        :class:`gadma.utils.variables.Sud`])

    * :attr:`default_rand_gen` = random choice over domain.
    """

    _help_dict = {'Sud': Sud, 'Lin': Lin, 'Exp': Exp, 0: Sud, 1: Lin, 2: Exp}
    default_domain = ['Sud', 'Lin', 'Exp']
    default_rand_gen = dynamic_generator

    def __init__(self, name, domain=None, rand_gen=None):
        super(DynamicVariable, self).__init__(name,
                                              units="universal",
                                              domain=domain,
                                              rand_gen=rand_gen)
        if not all(dom in self.__class__._help_dict for dom in self.domain):
            raise ValueError("Domain of DynamicVariable must be a subset "
                             "of the following general domain: "
                             f"{self._help_dict.keys()}. "
                             f"Got domain: {self.domain}")

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
    def _transform_value_from_gen_to_phys(value, Nanc):
        return value

    @staticmethod
    def _transform_value_from_phys_to_gen(value, Nanc):
        return value
