from ..utils import Variable, TimeVariable, DemographicVariable
from ..utils import PopulationSizeVariable, VariablePool, variables_values_repr
from . import Model, Epoch, Split
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
    :param Nref: rescaling factor of the parameters values.
    :type Nref: float
    :param Nanc_variable: If not None then demographic model has parameter of
                          the ancestral population size. Usually it is some
                          variable but for dadi and moments this parameter
                          could be missed.
    :type Nanc_variable: :class:`gadma.utils.Variable`
    :param linear_constrain: linear constrain on parameters.
    :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
    """
    def __init__(self, gen_time=None, theta0=None, mu=None, Nref=None,
                 Nanc_variable=None, linear_constrain=None):
        super(DemographicModel, self).__init__(raise_excep=False)
        self.gen_time = gen_time
        self.Nref = Nref  # rescaling factor
        self.theta0 = theta0  # mutation flux = 4 * mu * length
        self.mu = mu  # mutation rate per base per generation
        self.Nanc_variable = Nanc_variable
        if self.has_anc_size:
            if not isinstance(self.Nanc_variable, PopulationSizeVariable):
                raise ValueError("Nanc_variable must be instance of "
                                 "PopulationSizeVariable, got: "
                                 f"{self.Nanc_variable.__class__}.")
            if self.Nanc_variable.units != "physical":
                raise ValueError("Nanc_variable must be in physical units, "
                                 f"got: {self.Nanc_variable.units}.")
            # If we have rescaling we have to rescale Nanc, it will be done
            # automatically in add_variable
            self.add_variable(self.Nanc_variable)
            assert len(self.variables) == 1
            # as there was rescaling then variable has changed:
            self.Nanc_variable = self.variables[0]
        else:
            self.Nanc_variable = None
        self.fixed_vars = {}
        self.linear_constrain = linear_constrain

    @property
    def has_anc_size(self):
        return self.Nanc_variable is not None

    def add_variable(self, variable):
        """
        Overrides :meth:`Model.add_variable` method. Rescales added variables
        if they are in physical units.

        :param variable: variable to add.
        :type variable: :class:`Variable`
        """
        if self.Nref is not None and not isinstance(variable,
                                                    DemographicVariable):
            raise ValueError("Demographic model has rescaling factor (Nref), "
                             "it is not possible to add not-demographic "
                             f"variables in it. Got variable: {variable}.")
        super(DemographicModel, self).add_variable(variable)

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

    def translate_values(self, units, values, Nanc=None,
                         time_in_generations=False, rescale_back=False):
        """
        Translates values from current units to new. If some variables do not
        have units they are not translated.

        :param units: Units to translate to. Could be "physical" or "genetic".
        :param values: Values of parameters.
        :param Nanc: Size of ancestral population if it is not a parameter.
                     E.g. has_anc_size could be False.
        :param time_in_generations: If False then time is translated to
                                    years according to gen_time of model.
                                    Valid only if `units`=="physical".
        :param rescale_back: If True then values are rescaled back according
                             to Nref factor of the model.
        """
        if Nanc is None and not self.has_anc_size:
            for var in self.variables:
                if (isinstance(var, DemographicVariable) and
                        var.units == "physical"):
                    raise ValueError("It is not possible to translate "
                                     "parameter values of model without Nanc "
                                     "as parameter (has_anc_size==False).")
        var2value = self.var2value(values)
        if Nanc is None and self.has_anc_size:
            Nanc = var2value[self.Nanc_variable]
        Tg = self.gen_time
        if Tg is None or time_in_generations:
            Tg = 1
        translated_values = list()
        for var in self.variables:
            val = var2value[var]
            if not isinstance(var, DemographicVariable):
                # ignore translation if it is not possible to do it
                tr_value = val
            else:
                tr_value = var.translate_value_into(units=units,
                                                    value=val,
                                                    Nanc=Nanc)
            if isinstance(var, TimeVariable) and units == "physical":
                tr_value *= Tg
            # If our physical units are scaled then we should rescale them back
            if rescale_back:
                if (self.Nref is not None and
                        isinstance(var, DemographicModel) and
                        var.units == "physical"):
                    tr_value = var.rescale_value(tr_value, reverse=True)
            translated_values.append(tr_value)
        return translated_values


class EpochDemographicModel(DemographicModel):
    """
    Class for demographic model of epoch type.
    This type is common for :py:mod:`dadi` and :py:mod:`moments`.

    See :class:`gadma.models.demographic_model.DemographicModel` for
    constructor docs.
    """
    def __init__(self, gen_time=None, theta0=None, mu=None, Nref=None,
                 Nanc_variable=None, linear_constrain=None):
        self.events = list()
        super(EpochDemographicModel, self).__init__(
            gen_time=gen_time,
            theta0=theta0,
            mu=mu,
            Nref=Nref,
            Nanc_variable=Nanc_variable,
            linear_constrain=linear_constrain
        )

    def _get_current_pop_sizes(self):
        """
        Returns the populations sizes after the last epoch.
        """
        if len(self.events) == 0:
            if self.has_anc_size:
                return copy.copy([self.Nanc_variable])
#            if self.Nref is not None:
#                # rescaling Nanc = 1.0:
#                size_cls = PopulationSizeVariable
#                return size_cls._transform_value_from_phys_to_gen(1.0,
#                                                                  self.Nref)
            return [1.0]
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
