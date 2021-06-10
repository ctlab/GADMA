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
    :param has_anc_size: If True then model does not have size of ancestral
                         population. It is the case for dadi and moments as
                         they have multinom mode when this size is generated
                         automatically from the rest of the parameters.
    :type has_anc_size: bool
    :param linear_constrain: linear constrain on parameters.
    :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
    """

    def __init__(self, gen_time=None, theta0=None, mu=None, Nref=None,
                 has_anc_size=False, linear_constrain=None):
        super(DemographicModel, self).__init__(raise_excep=False)
        self.gen_time = gen_time
        self.Nref = Nref  # rescaling factor
        self.theta0 = theta0  # mutation flux = 4 * mu * length
        self.mu = mu  # mutation rate per base per generation
        self.has_anc_size = has_anc_size
        self.fixed_vars = {}
        self.linear_constrain = linear_constrain

    def add_variable(self, variable):
        """
        Overrides :meth:`Model.add_variable` method. Checks that if model
        does not have ancestral size then no variables in physical units are
        added.

        :param variable: variable to add.
        :type variable: :class:`Variable`
        """
        if (not self.has_anc_size and
                isinstance(variable, DemographicVariable) and
                variable.units == "physical"):
            raise ValueError("Demographic model has no ancestral size: only "
                             "variables in genetic units are accepted."
                             "It will be impossible to translate physical "
                             f"units of a given variable: {variable}.")
        super(DemographicModel, self).add_variable(variable)

    def _get_Nanc_size(self, values):
        """
        Method to override. Is used in :meth:`.DemographicModel.get_Nanc_size`.
        """
        raise NotImplementedError

    def get_Nanc_size(self, values=None):
        """
        Returns Nanc size as variable (if it is) or constant. If model does not
        has_anc_size then ValueError is raised.

        :param values: Values of model variables. Could be required to
                       understand what variable or constant is actual Nanc
                       size.
        """
        if self.has_anc_size:
            return self._get_Nanc_size(values)
        else:
            raise ValueError("Model does not have ancestral size so it is not"
                             " available.")

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
                                    Valid only if `units` == "physical".
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
            Nanc = self.get_value_from_var2value(var2value, self.Nanc_size)
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
                        isinstance(var, DemographicVariable) and
                        var.units == "physical"):
                    tr_value = var.rescale_value(tr_value, reverse=True)
            translated_values.append(tr_value)
        return translated_values


class EpochDemographicModel(DemographicModel):
    """
    Class for demographic model of epoch type.
    This type is common for :py:mod:`dadi` and :py:mod:`moments`.

    See :class:`gadma.models.demographic_model.DemographicModel` for
    more information.

    By default (when None value) has_anc_size is True if Nanc_size is set and
    False otherwise (when Nanc_size is None). If Nanc_size is not set (i.e.
    None) then it is taken as 1.0.

    :param Nanc_size: Constant or variable for demographic model.
                      Usually it is some variable but for dadi and moments this
                      parameter could be equal to 1. In that case they will use
                      multinom inference and get best Nanc_size for the model.
    :type Nanc_size: float or :class:`gadma.utils.PopulationSizeVariable`
    """

    def __init__(self, gen_time=None, theta0=None, mu=None, Nref=None,
                 has_anc_size=None, Nanc_size=None, linear_constrain=None,
                 inbreeding_args=None):
        if has_anc_size is None:
            has_anc_size = Nanc_size is not None
        if Nanc_size is None:
            Nanc_size = 1.0
        self.inbreeding_args = inbreeding_args
        self.events = list()
        super(EpochDemographicModel, self).__init__(
            gen_time=gen_time,
            theta0=theta0,
            mu=mu,
            Nref=Nref,
            has_anc_size=has_anc_size,
            linear_constrain=linear_constrain
        )
        self.Nanc_size = Nanc_size
        if isinstance(self.Nanc_size, Variable):
            if not isinstance(self.Nanc_size, PopulationSizeVariable):
                raise ValueError("Nanc_size must be instance of "
                                 "PopulationSizeVariable, got: "
                                 f"{self.Nanc_size.__class__}.")
            if self.Nanc_size.units != "physical":
                raise ValueError("Nanc_size must be in physical units, "
                                 f"got: {self.Nanc_size.units}.")
            self.add_variable(self.Nanc_size)
            assert len(self.variables) == 1

    def _get_Nanc_size(self, values=None):
        """
        Method of parent class. Is used in
        :meth:`.DemographicModel.get_Nanc_size`.
        `values` are not used in this method.
        """
        return self.Nanc_size

    def _get_current_pop_sizes(self):
        """
        Returns the populations sizes after the last epoch.
        """
        if len(self.events) == 0:
            return copy.copy([self._get_Nanc_size()])
        return copy.copy(self.events[-1].size_args)

    def number_of_populations(self):
        """
        Returns number of populations in the model.
        """
        return len(self._get_current_pop_sizes())

    @property
    def has_inbreeding(self):
        return self.inbreeding_args is not None

    def add_inbreeding(self, inbr_args=None):
        self.inbreeding_args = inbr_args
        self.add_variables(inbr_args)

    def add_epoch(self, time_arg, size_args, mig_args=None,
                  dyn_args=None, sel_args=None,
                  dom_args=None):
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
        if self.has_inbreeding:
            raise ValueError("Model already has inbreeding."
                             " You can't add new Epoch")

        sizes = self._get_current_pop_sizes()
        new_epoch = Epoch(time_arg, sizes, size_args, mig_args,
                          dyn_args, sel_args, dom_args)
        self.events.append(new_epoch)
        self.add_variable(new_epoch)

    def add_split(self, pop_to_div, size_args):
        """
        Adds new split to the demographic model events.

        :param pop_to_div: population to divide.
        :param size_args: population sizes of two subpopulations after the
                          split.
        """
        if self.has_inbreeding:
            raise ValueError("Model already has inbreeding. "
                             " Split is impossible.")
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

        if self.has_inbreeding:
            inbr_coefficients = []
            for inbreeding in self.inbreeding_args:
                inbr_value = round(values[inbreeding.name], 3)
                inbr_coefficients.append(f"{inbr_value} ({inbreeding.name})")

            inbr_string = ", ".join(inbr_coefficients)

            strings.append(f"[inbr: {inbr_string}]")

        return "[ " + ",\t".join(strings) + " ]"

    def get_involved_for_split_time_vars(self, n_split):
        """
        Returns list of ints and bias. If value > 0 then this variable is
        involved in sum of times for split.

        It will return A, b: Ax + b = time of `n_split` split.
        """
        var2value = self.var2value(np.zeros(len(self.variables)))
        var2value = {var: value for var, value in var2value.items()
                     if var not in self.fixed_values}
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
