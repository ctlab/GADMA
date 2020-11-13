from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import VariablePool, variables_values_repr
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable
from . import Model, Epoch, Split
from collections import OrderedDict
import copy
import numpy as np


class DemographicModel(Model):
    """
    Base class for demographic model.

    :param gen_time: Time of one generation.
    :type gen_time: float
    :param theta0: Mutation flux (4 * mu * L, where L - length of sequence).
    :type theta0: float
    :param mu: Mutation rate per base per generation.
    :type mu: float
    :param linear_constrain: linear constrain on parameters.
    :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
    """
    def __init__(self, gen_time=None, theta0=None, mu=None,
                 linear_constrain=None):
        self.gen_time = None
        self.Nref = 1.0
        self.theta0 = theta0  # mutation flux = 4 * mu * length
        self.mu = mu  # mutation rate per base per generation
        self.fixed_vars = {}
        self.linear_constrain = linear_constrain
        super(DemographicModel, self).__init__(raise_excep=False)

    def get_number_of_parameters(self, values):
        """
        Returns number of parameters of the model.
        """
        var2value = self.var2value(values)
        return len(var2value.keys())

    def as_custom_string(self, values):
        """
        Returns string representation of the demographic model with
        parameters as a list with variables and their values.

        :param values: Values of the demographic model.
        """
        if isinstance(values, dict):
            values = [val for var, val in self.var2value(values).items()]
        return variables_values_repr(self.variables, values)

    def translate_units(self, values, Nanc, Tg=1):
        """
        Translates values from genetic units to physical.

        :param values: Values of parameters.
        :param Nanc: Size of ancestral population.
        :param Tg: Time per generation.
        """
        var2value = self.var2value(values)
        translated_values = list()
        for var, val in var2value.items():
            if not hasattr(var, 'translate_units'):
                raise ValueError("Values cannot be translated for demographic"
                                 " model as there is at least one variable "
                                 "that is not specified as demographic.")
            translated_values.append(var.translate_units(val, Nanc))
            if isinstance(var, TimeVariable):
                translated_values[-1] *= Tg
        return translated_values


class EpochDemographicModel(DemographicModel):
    """
    Class for demographic model of epoch type.
    This type is common for :py:mod:`dadi` and :py:mod:`moments`.

    See :class:`gadma.models.demographic_model.DemographicModel` for
    constructor docs.
    """
    def __init__(self, gen_time=None, theta0=None, mu=None,
                 linear_constrain=None):
        self.events = list()
        super(EpochDemographicModel, self).__init__(gen_time, theta0, mu,
                                                    linear_constrain)

    def _get_current_pop_sizes(self):
        """
        Returns the populations sizes after the last epoch.
        """
        if len(self.events) == 0:
            return [self.Nref]
        return copy.copy(self.events[-1].size_args)

    def number_of_populations(self):
        """
        Returns number of populations in the model.
        """
        return len(self._get_current_pop_sizes())

    def add_epoch(self, time_arg, size_args, mig_args=None,
                  dyn_args=None, sel_args=None, dom_args=None):
        """
        Adds new epoch to the demographic model events.

        :param time_arg: time of the epoch.
        :param size_args: population sizes at the end of the epoch.
        :param mig_args: migrations between populations during the epoch.
        :param dyn_args: dynamics of the populations during the epoch.
        :param sel_args: selection coefficients of the populations during
                         the epoch.
        :param dom_args: dominance coefficients.

        :note: all arguments could contain variables of :class:`Variable`\
               class as well as different constants/values including\
               :class:`gadma.models.BinaryOperation` instances.
        """
        sizes = self._get_current_pop_sizes()
        new_epoch = Epoch(time_arg, sizes, size_args,
                          mig_args, dyn_args, sel_args, dom_args)
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

    def fix_variable(self, variable, value):
        """
        Fixes variable in the model to the value.

        :param variable: Variable to fix.
        :param value: Value of the variable to fix.
        """
        for event in self.events:
            if variable in event.variables:
                event.fix_variable(variable, value)
                super(DemographicModel, self).fix_variable(variable, value)
                return
        raise ValueError(f"There is no such unfixed variable {variable} in"
                         " demographic model.")

    def unfix_variable(self, variable):
        """
        Unfixes the variable of the model.
        """
        for event in self.events:
            if variable in event._variables:
                event.unfix_variable(variable)
                super(DemographicModel, self).unfix_variable(variable)
                return
        raise ValueError(f"There is no such fixed variable {variable} in"
                         " demographic model.")

    def fix_dynamics(self, values):
        """
        Makes all dynamics in events fixed.

        :param values: Values of parameters to take for fixation.
        :type values: list or dict
        """
        var2value = self.var2value(values)
        assert isinstance(var2value, dict)
        for event in self.events:
            if isinstance(event, Epoch) and event.dyn_args is not None:
                for dyn_arg in event.dyn_args:
                    if isinstance(dyn_arg, Variable):
                        value = var2value[dyn_arg]
                        super(DemographicModel, self).fix_variable(dyn_arg,
                                                                   value)
                        event.fix_variable(dyn_arg, value)

    def unfix_dynamics(self):
        """
        Makes all dynamics in events unfixed.
        """
        for event in self.events:
            if isinstance(event, Epoch) and event.dyn_args is not None:
                for dyn_arg in event.dyn_args:
                    if isinstance(dyn_arg, Variable):
                        super(DemographicModel, self).unfix_if_fixed(dyn_arg)
                        event.unfix_if_fixed(dyn_arg)

    def get_number_of_parameters(self, values):
        """
        Returns number of parameters of the model. Be careful as this number
        could be different to the number of variables. According to the
        population size dynamics in `values` some variables could have no
        influence on the model and in this case they are not counted.

        :param values: Values of the parameters. Dynamics makes the most
                       sense.
        :type values: list or dict
        """
        var2value = self.var2value(values)
        all_variables = VariablePool()
        for event in self.events:
            if isinstance(event, Split):
                continue
            help_event = Epoch(event.time_arg,
                               [None for _ in event.init_size_args],
                               event.size_args,
                               event.mig_args,
                               event.dyn_args,
                               event.sel_args,
                               event.dom_args)
            variables = help_event.variables
            if event.dyn_args is not None:
                for dyn in event.dyn_args:
                    if isinstance(dyn, Variable):
                        value = var2value[dyn]
                    else:
                        value = dyn
                    if value != 'Sud':
                        for init_size in event.init_size_args:
                            if (isinstance(init_size, Variable) and
                                    init_size not in variables):
                                variables.append(init_size)
            for var in variables:
                if var not in all_variables and var not in self.fixed_values:
                    all_variables.append(var)
        return len(all_variables)

    def as_custom_string(self, values):
        """
        Returns string representation of the demographic model with
        parameters.

        :param values: Values of the demographic model.
        """
        strings = []
        values = {var.name: val
                  for var, val in self.var2value(values).items()}
        for event in self.events:
            strings.append(event.as_custom_string(values))
        return "[ " + ",\t".join(strings) + " ]"

    def get_involved_for_split_time_vars(self, n_split):
        """
        Returns list of ints and bias. If value > 0 then this variable is
        involved in sum of times for split.

        It will return A, b: Ax + b = time of `n_split` split.
        """
        var2value = self.var2value(np.zeros(len(self.variables)))
        n_sp = 0
        b = 0
        total_n_split = 0
        for event in self.events:
            if isinstance(event, Split):
                total_n_split += 1
        for event in reversed(self.events):
            if isinstance(event, Split):
                n_sp += 1
                if n_sp == total_n_split - n_split + 1:
                    return list(var2value.values()), b
            else:
                time_arg = event.time_arg
                if isinstance(time_arg, Variable):
                    var2value[time_arg] += 1
                else:
                    b += time_arg
        return [], 0
