from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import VariablePool, variables_values_repr
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable
from .model import Model
from collections import OrderedDict
import copy
import numpy as np
import importlib.util


class CustomDemographicModel(Model):
    def __init__(self, function, variables,
                 gen_time=None, theta0=None, mu=None):
        self.function = function
        self.gen_time = None
        self.Nref = 1.0
        self.theta0 = theta0  # mutation flux = 4 * mu * length
        self.mu = mu  # mutation rate per base per generation
        super(CustomDemographicModel, self).__init__(raise_excep=False)
        self.variables = variables

    def as_custom_string(self, values):
        if isinstance(values, dict):
            values = [val for var, val in self.var2value(values).keys()]
        return variables_values_repr(self.variables, values)
