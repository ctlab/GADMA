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
        super(DemographicModel, self).__init__(raise_excep=False)

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
