from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import VariablePool, variables_values_repr
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable
from .event import Model, SetSize, MoveLineages, Leaf
from . import DemographicModel
from collections import OrderedDict
import copy
import numpy as np


class CoalescentDemographicModel(DemographicModel):

    default_p = 1
    default_init_g = None
    default_size_g = 0
    default_pop_size = None

    def __init__(self, N_e, N_a, gen_time=None, sequence_length=None, mu=None, rec_rate=None,
                 linear_constrain=None):
        self.events = list()
        self.N_e = N_e
        self.N_a = N_a
        self.add_variable(N_a)
        self.has_Na = True
        self.rec_rate = rec_rate
        self.sequence_length = sequence_length
        super(CoalescentDemographicModel, self).__init__(gen_time, 4 * mu * sequence_length, mu,
                                                         linear_constrain)

    def name2value(self, values):
        var2value = self.var2value(values)
        name2value = dict()
        for var, value in var2value.items():
            name2value[var.name] = value
        return name2value

    def change_pop_size(self, pop, t, size_pop=None, g=0):
        new_set_size = SetSize(pop=pop,
                               t=t,
                               size_pop=size_pop,
                               g=g)
        self.events.append(new_set_size)
        self.add_variables(new_set_size.variables)

    def move_lineages(self, pop_from, pop_to, t, p=1, size_pop_to=None, g_pop_to=None):
        new_move_lineages = MoveLineages(pop_from=pop_from,
                                         pop_to=pop_to,
                                         t=t,
                                         p=p,
                                         size_pop_to=size_pop_to,
                                         g_pop_to=g_pop_to)
        self.events.append(new_move_lineages)
        self.add_variables(new_move_lineages.variables)

    def add_leaf(self, pop, t=0, size_pop=None, g=None):
        new_leaf = Leaf(pop=pop,
                        t=t,
                        size_pop=size_pop,
                        g=g)
        self.events.append(new_leaf)
        self.add_variables(new_leaf.variables)
