from ..utils import Variable
from .event import Epoch, Split
from collections import OrderedDict
import copy

class DemographicModel(object):
    def __init__(self, N_A=None):
        self.events = list()
        self._variables = list()
        self.N_A = N_A

    def add_variable(self, variable):
        if not isinstance(variable, Variable):
            return
        if variable not in self._variables:
            self._variables.append(variable)

    def add_variables(self, variables):
        for variable in variables:
            self.add_variable(variable)

    def add_variables_2d(self, variables):
        for list_of_vars in variables:
            for variable in list_of_vars:
                self.add_variable(variable)

    def _get_current_pop_sizes(self):
        if len(self.events) == 0:
            return [1.0]
        return copy.deepcopy(self.events[-1].size_args)         

    def add_epoch(self, time_arg, size_args, mig_args=None, dyn_args=None, sel_args=None):
        sizes = self._get_current_pop_sizes()
        print(sizes, size_args)
        new_epoch = Epoch(time_arg, sizes, size_args, mig_args, dyn_args, sel_args) 
        self.events.append(new_epoch)
        self.add_variable(time_arg)
        self.add_variables(size_args)
        if mig_args is not None:
            self.add_variables_2d(mig_args)
        if dyn_args is not None:
            self.add_variables(dyn_args)
        if sel_args is not None:
            self.add_variables(sel_args)

    def add_split(self, pop_to_div, size_args):
        sizes = self._get_current_pop_sizes()
        sizes[pop_to_div] = size_args[0]
        sizes.append(size_args[1])
        new_split = Split(pop_to_div, sizes)
        self.events.append(new_split)
        self.add_variables(sizes)

    @property
    def varname2value(self):
        return _varname2value

    @property
    def variables(self):
        return self._variables

    def var2value(self, values):
        var2value = OrderedDict()
        for variable, value in zip(self.variables, values):
            var2value[variable] = value
        return var2value
