from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import VariablePool, variables_values_repr
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable
from .demographic_model import DemographicModel
from collections import OrderedDict
import copy
import numpy as np
import importlib.util


class CustomDemographicModel(DemographicModel):
    def __init__(self, function, variables,
                 gen_time=None, theta0=None, mu=None):
        self.function = function
        super(CustomDemographicModel, self).__init__(gen_time, theta0, mu)
        self.variables = variables
