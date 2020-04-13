import numpy as np
from transform import Transform
from distributions import *
from functools import partial
import numpy as np
from keyword import iskeyword

class Unique(type):

    def __call__(cls, name, *args, **kwargs):
        if name not in cls._cache:
            self = cls.__new__(cls, name, *args, **kwargs)
            cls.__init__(self, name, *args, **kwargs)
            cls._cache[name] = weakref.ref(self)
        dead = set()
        for name, ref in cls._cache.items():
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        for name in dead:
            if name in cls._cache:
                del cls._cache[name]
        return cls._cache[args[0]]()

    def __init__(cls, name, bases, attributes):
        super().__init__(name, bases, attributes)
        cls._cache = {}

class Variable(metaclass=Unique):
    '''
    Variable class for keeping parameter of event.
    '''
    def __init__(self, name, var_type, domain, rand_gen=np.random.normal):
        self.is_active=True
        self.name = name
        #TODO The following assert does not work for python 2
        assert(name.isidentifier() and not iskeyword(name))
        self.var_type = var_type
        self.domain = domain
#        if not isinstance(transform, Transform):
#            raise ValueError('Can only init with gadma.Transform object')
#        self.transform = transform
        self.rand_gen = rand_gen

    def resample(self, *args, **kwargs):
        return self.rand_gen(self.domain, *args, **kwargs)

    def suppress(self):
        self.is_active = False

    def activate(self):
        self.is_active = True

    def copy(self):
        return self.__class__(
            name=self.name,
            var_type=self.var_type,
            domain=self.domain,
            rand_gen=self.rand_gen)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Variable):
            return self.name == other.name and self.type == other.type
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "%s %s" % (self.__class__.__name__, self.name)

    def get_bounds(self):
        raise NotImplementedError

    def get_possible_values(self):
        raise NotImplementedError


class ContinuousVariable(Variable):
    def __init__(self, name, domain, rand_gen=np.random.normal):
        super(ContinuousVariable, self).__init__(name, 'continuous', domain, rand_gen)

    def get_bounds(self):
        return self.domain

    def get_possible_values(self):
        raise AttributeError("Impossible to produce a list of values for continuous variable " + self.name)


class DiscreteVariable(Variable):
    def __init__(self, name, domain, rand_gen=np.random.normal):
        super(DiscreteVariable, self).__init__(name, 'discrete', domain, rand_gen)

    def get_bounds(self):
        return [min(self.domain), max(self.domain)]

    def get_possible_values(self):
        return self.domain


class PopulationSizeVariable(ContinuousVariable):
    '''
    Parameter class for keeping size of population.
    '''
    default_domain   = np.array([1e-2, 100])
    default_rand_gen = lambda domain: trunc_lognormal_3_sigma_rule(1, domain[0], domain[1])

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.default_domain
        if rand_gen is None:
            rand_gen = self.default_rand_gen
        super(PopulationSizeVariable, self).__init__(name, domain, rand_gen)


class MigrationVariable(ContinuousVariable):
    '''
    Parameter class for keeping migration.
    '''
    default_domain   = np.array([0, 10])
    default_rand_gen = lambda domain: trunc_normal_3_sigma_rule(1, domain[0], domain[1])

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.default_domain
        if rand_gen is None:
            rand_gen = self.default_rand_gen
        super(MigrationVariable, self).__init__(name, domain, rand_gen)


class TimeVariable(ContinuousVariable):
    '''
    Parameter class for keeping time.
    '''
    default_domain   = np.array([0, 5])
    default_rand_gen = lambda domain: trunc_normal_3_sigma_rule(1, domain[0], domain[1])

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.default_domain
        if rand_gen is None:
            rand_gen = self.default_rand_gen
        super(TimeVariable, self).__init__(name, domain, rand_gen)

class SelectionVariable(ContinuousVariable):
    '''
    Parameter class for keeping size of population.
    '''
    default_domain   = np.array([0, 10])
    default_rand_gen = lambda domain: trunc_normal_3_sigma_rule(1, domain[0], domain[1])

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.default_domain
        if rand_gen is None:
            rand_gen = self.default_rand_gen
        super(SelectionVariable, self).__init__(name, domain, rand_gen)

class PercentVariable(ContinuousVariable):
    '''
    Parameter class for keeping time.
    '''
    default_domain   = np.array([0, 1])
    default_rand_gen = np.random.normal

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.default_domain
        if rand_gen is None:
            rand_gen = self.default_rand_gen
        super(PercentVariable, self).__init__(name, domain, rand_gen)


class Dynamic(object):
    form_str = ''
    def __init__(self, size_from, size_to):
        self.size_from = size_from
        self.size_to   = size_to
#        self.time = time

    @staticmethod
    def _inner_func(y1, y2, x_diff):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def func_str(self, time):
        return self.form_str % (str(self.size_from), str(self.size_to), str(time))


class Exp(Dynamic):
    form_str = "lambda t: {0} * ({1} / {0}) ** (t / {2})"
    @staticmethod
    def _inner_func(y1, y2, x_diff):
        return lambda x: y1 * (y2 / y1) ** (x / x_diff)

    def __str__(self):
        return "Exp"


class Lin(Dynamic):
    form_str = "lambda t: {0} + ({1} - {0}) * (t / {2})"
    @staticmethod
    def _inner_func(y1, y2, x_diff):
        return lambda x: y1 + (y2 - y1) * (x / x_diff)

    def __str__(self):
        return "Lin"


class Sud(Dynamic):
    form_str = '{1}'
    @staticmethod
    def _inner_func(y1, y2, x_diff):
        return y2

    def __str__(self):
        return "Sud"


class DynamicVariable(DiscreteVariable):
    '''
    Parameter class for keeping size of population.
    '''
    _help_dict = {'Sud': Sud, 'Lin': Lin, 'Exp': Exp}
    default_domain   = np.array([Sud, Lin, Exp])
    default_rand_gen = lambda domain: np.random.choice(domain)

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.default_domain
        if not all(av_value in self.default_domain for av_value in domain):
            raise Exception
        if rand_gen is None:
            rand_gen = self.default_rand_gen
        super(DynamicVariable, self).__init__(name, domain, rand_gen)

    @staticmethod
    def get_func_from_value(value):
        if value not in DynamicVariable._help_dict:
            raise Exception("Value %s not in domain" % (str(value)))
        return DynamicVariable._help_dict[value]._inner_func
