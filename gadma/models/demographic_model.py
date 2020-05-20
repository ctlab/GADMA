from ..utils import Variable
from . import Model, Epoch, Split
from collections import OrderedDict
import copy

class DemographicModel(Model):
    """
    Class for demographic model of epoch type.
    This type is common for :py:mod:`dadi` and :py:mod:`moments`.

    :param Nref: reference population size (usually ancestral population
                 size).
    :type Nref: float
    """
    def __init__(self, Nref=1.0):
        self.events = list()
        self.Nref = Nref
        self.fixed_vars = {}
        super(DemographicModel, self).__init__(raise_excep=False)

    @staticmethod
    def from_structure(self, structure, create_migs=True, create_sels=False,
                       create_dyns=True, sym_migs=False, Nref=1.0):
        dm = DemographicModel(Nref)
        for n_pop in range(1, len(structure) + 1):
            n_int = structure[n_pop]
            if n_pop == 1:
                n_int -= 1
            for i_int in range(1, n_int + 1):
                time_var = TimeVariable('t%d' % (i_int))
                size_vars = list()
                for i_pop in range(n_pop):
                    var = PopulationSizeVariable('nu%d%d' % (i_int, n_pop))
                    size_vars.append(var)
                mig_vars = None
                if create_migs:
                    mig_vars = np.array(shape=(n_pop, n_pop), dtype=object)
                    for i in range(n_pop):
                        for j in range(n_pop):
                            if i == j:
                                continue
                            if sym_mig and j < i:
                                continue
                            var = MigrationVariable('m%d_%d%d' % 
                                                    (i_int, i, j))
                            mig_vars[i][j] = var
                            if sym_migs:
                                mig_vars[j][i] = var
                sel_vars = None
                if create_sels:
                    sel_vars = list()
                    for i in range(n_pop):
                        var = SelectionVariable('g%d%d' % (i_int, i))
                        sel_vars.append(var)
                dyn_vars = None
                if create_dyns:
                    dyn_vars = list()
                    for i in range(n_pop):
                        var = DynamicVariable('dyn%d%d' % (i_int, i))
                        dyn_vars.append(var)
                dm.add_epoch(time_var, size_vars, mig_vars, dyn_vars, sel_vars)
            var_1 = PopulationSizeVariable(size_vars[-1].name + '_1')
            var_2 = PopulationSizeVariable(size_vars[-1].name + '_2')
            dm.add_split(n_pop - 1, [var_1, var_2])

    def _get_current_pop_sizes(self):
        """
        Returns the populations sizes after the last epoch.
        """
        if len(self.events) == 0:
            return [self.Nref]
        return copy.copy(self.events[-1].size_args)

    def add_epoch(self, time_arg, size_args, mig_args=None,
                  dyn_args=None, sel_args=None):
        """
        Adds new epoch to the demographic model events.

        :param time_arg: time of the epoch.
        :param size_args: population sizes at the end of the epoch.
        :param mig_args: migrations between populations during the epoch.
        :param dyn_args: dynamics of the populations during the epoch.
        :param sel_args: selection coefficients of the populations during
                         the epoch.

        :note: all arguments could contain variables of :class:`Variable`\
               class as well as different constants/values.
        """
        sizes = self._get_current_pop_sizes()
        new_epoch = Epoch(time_arg, sizes, size_args,
                          mig_args, dyn_args, sel_args)
        self.events.append(new_epoch)
        self.add_variables(new_epoch.variables)

    def add_split(self, pop_to_div, size_args):
        """
        Adds new split to the demographic model events.

        :param pop_to_div: population to divide.
        :param size_args: population sizes of two subpopulations after the
                          split.
        """
        sizes = self._get_current_pop_sizes()
        sizes[pop_to_div] = size_args[0]
        sizes.append(size_args[1])
        new_split = Split(pop_to_div, sizes)
        self.events.append(new_split)
        self.add_variables(new_split.variables)

    def get_value(self, item):
        """
        Returns value of the item if it is variable, otherwise returns item
        as it is.

        :param item: object to get its value.
        """
        if isinstance(item, Variable):
            return super(DemographicModel, self).get_value(item)
        return item

    def fix_dynamics(self, var2value):
        assert isinstance(var2value, dict)
        for event in self.events:
            if isinstance(event, Epoch) and event.dyn_args is not None:
                for dyn_arg in event.dyn_args:
                    if isinstance(dyn_arg, Variable):
                        value = var2value[dyn_arg]
                        self.fix_variable(dyn_arg, value)
                        self.event.fix_variable(dyn_arg, value)

    def unfix_dynamics(self):
        for event in self.events:
            if isinstance(event, Epoch) and event.dyn_args is not None:
                for dyn_arg in event.dyn_args:
                    if isinstance(dyn_arg, Variable):
                        self.unfix_variable(dyn_arg)
                        self.event.unfix_variable(dyn_arg)

