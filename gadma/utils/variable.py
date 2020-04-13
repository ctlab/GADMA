import numpy as np
from .distributions import *
from functools import partial
from keyword import iskeyword

class UniqueMeta(type):
    """
    metaclass for creating variables with unique names
    taking from https://stackoverflow.com/questions/34818622/ensure-uniqueness-of-instance-attribute-in-python
    """
    class DescName(object):
        def __init__(self, cls):
            self.cache = set()
            self.obj2name = {None: self}

        def __get__(self, obj, cls=None):
            return self.obj2name[obj]

        def __set__(self, obj, value):
            """
            Name cannot be changed
            """
            Warning("Name of the variable cannot be changed")

        def setname(self, obj, name):
            if name in self.cache:
                raise AttributeError('The variable with the same name (%s) already exists' % name)

            self.cache.discard(self.obj2name.get(obj, None))
            self.cache.add(name)
            self.obj2name[obj] = name

    def __new__(meta, name, bases, dct):
        cls = super(UniqueMeta, meta).__new__(meta, name, bases, dct)
        cls.name = meta.DescName(cls)  # add the name class attribute
        return cls

    @classmethod
    def as_metaclass(meta, *bases):
        class metaclass(meta):
            def __new__(cls, name, this_bases, d):
                # subclass to ensure super works with our methods
                return meta(name, bases, d)
        return type.__new__(metaclass, str('tmpcls'), (), {})

    def __call__(cls, name, *args, **kwargs):
        # Instead of relying on type we do the new and init calls
        obj = cls.__new__(cls, *args, **kwargs)
        cls.name.setname(obj, name)
        obj.__init__(name, *args, **kwargs)
        return obj


class Variable(UniqueMeta.as_metaclass()):
    '''
    Variable class for keeping parameter of event.
    '''
    def __init__(self, name, var_type, domain, rand_gen=np.random.normal):
        self.is_active=True
#        self.name = name # is set in metaclass
        #TODO The following assert does not work for python 2
        assert(name.isidentifier() and not iskeyword(name))
        self.var_type = var_type
        self.domain = domain
#        if not isinstance(transform, Transform):
#            raise ValueError('Can only init with gadma.Transform object')
#        self.transform = transform
        self.rand_gen = rand_gen

    def resample(self, *args, **kwargs):
        return self.rand_gen.__call__(self.domain, *args, **kwargs)

    def __str__(self):
        return "%s %s" % (self.__class__.__name__, self.name)

    def __repr__(self):
        return "%s %s" % (self.__class__.__name__, self.name)

    def get_bounds(self):
        raise NotImplementedError

    def get_possible_values(self):
        raise NotImplementedError


class ContinuousVariable(Variable):
    default_domain   = np.array([-np.inf, np.inf])
    default_rand_gen = lambda domain: np.random.uniform(domain[0], domain[1])

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if rand_gen is None:
            rand_gen = self.__class__.default_rand_gen
        super(ContinuousVariable, self).__init__(name, 'continuous', domain, rand_gen)

    def get_bounds(self):
        return self.domain

    def get_possible_values(self):
        raise AttributeError("Impossible to produce a list of values for continuous variable " + self.name)


class DiscreteVariable(Variable):
    default_domain   = np.array([])
    default_rand_gen = lambda domain: np.random.choice(domain)

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if rand_gen is None:
            rand_gen = self.__class__.default_rand_gen
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


class MigrationVariable(ContinuousVariable):
    '''
    Parameter class for keeping migration.
    '''
    default_domain   = np.array([0, 10])
    default_rand_gen = lambda domain: trunc_normal_3_sigma_rule(1, domain[0], domain[1])


class TimeVariable(ContinuousVariable):
    '''
    Parameter class for keeping time.
    '''
    default_domain   = np.array([0, 5])
    default_rand_gen = lambda domain: trunc_normal_3_sigma_rule(1, domain[0], domain[1])


class SelectionVariable(ContinuousVariable):
    '''
    Parameter class for keeping size of population.
    '''
    default_domain   = np.array([0, 10])
    default_rand_gen = lambda domain: trunc_normal_3_sigma_rule(1, domain[0], domain[1])


class PercentVariable(ContinuousVariable):
    '''
    Parameter class for keeping time.
    '''
    default_domain   = np.array([0, 1])
    default_rand_gen = lambda domain: np.random.uniform(domain[0], domain[1])


class Dynamic(object):
    form_str = ''

    @staticmethod
    def _inner_func(y1, y2, x_diff):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def func_str(self, y1, y2, time):
        return self.form_str % (str(y1), str(y2), str(time))


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

    def __init__(self, name, domain=None, rand_gen=None):
        if domain is None:
            domain = self.__class__.default_domain
        if not all(dom in self.__class__.default_domain for dom in domain):
            raise Exception("Domain of DynamicVariable must be a subset of the following general domain: %s" % ", ".join([x.__name__ for x in self.default_domain]))
        super(DynamicVariable, self).__init__(name, domain=domain, rand_gen=rand_gen)

    @staticmethod
    def get_func_from_value(value):
        if value not in DynamicVariable._help_dict:
            raise Exception("Value %s not in domain" % (str(value)))
        return DynamicVariable._help_dict[value]._inner_func

    def get_bounds(self):
        raise Exception("DynamicVariable domain has incomparative values")
