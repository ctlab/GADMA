import numpy as np

from . import Model
from ..utils import Variable, FractionVariable, float_repr
import copy


class Event(Model):
    """
    Base class for some event.
    """
    def __init__(self):
        super(Event, self).__init__(raise_excep=False)

#    def set_value(self, variable, value):
#        """
#        Fixes variable `variable` to the value of `value`. This variable is
#        no longer available after this operation.
#
#        :param variable: Variable of the event to fix.
#        :param value: New constant value of the variable.
#        """
#        raise NotImplementedError()

    def as_custom_string(self, values):
        """
        Returns string representation of the event.
        """
        raise NotImplementedError()


class Epoch(Event):
    """
    Epoch for demographic model. All arguments could be both values and some
    variables. Additionally values could be combinations of the variables
    (see :class:`gadma.models.VariablesCombinations`).

    :param time_arg: Time of the epoch.
    :type time_arg: float or :class:`gadma.TimeVariable`
    :param init_size_args: Sizes of populations at the beginning of the epoch.
    :type init_size_args: list of values and/or
                          :class:`gadma.PopulationSizeVariable`
    :param size_args: Sizes of populations at the end of the epoch.
    :type size_args: list of values and/or
                     :class:`gadma.PopulationSizeVariable`
    :param mig_args: Migration rates between populations.
    :type mig_args: 2d list of values and/or :class:`gadma.MigrationVariable`
    :param dyn_args: Dynamics of population size changes during the epoch.
    :type dyn_args: list of values and/or  :class:`gadma.DynamicVariable`
    :param sel_args: Selection rates for each population during the epoch.
    :type sel_args: list of values and/or :class:`gadma.SelectionVariable`
    """
    def __init__(self, time_arg, init_size_args, size_args,
                 mig_args=None, dyn_args=None, sel_args=None, dom_args=None):
        # Simple checks
        assert(len(init_size_args) == len(size_args))
        if mig_args is not None:
            mig_args = np.array(mig_args)
            assert(mig_args.ndim == 2)
            assert(len(mig_args) == len(size_args))
            for x in mig_args:
                assert(len(x) == len(size_args))
        if dyn_args is not None:
            assert (len(dyn_args) == len(size_args))
        if sel_args is not None:
            assert (len(sel_args) == len(size_args))
        if dom_args is not None:
            assert (len(dom_args) == len(size_args))

        self.n_pop = len(init_size_args)
        self.time_arg = time_arg
        self.init_size_args = init_size_args
        self.size_args = size_args
        self.sel_args = sel_args
        self.dom_args = dom_args
        if self.dom_args is not None and self.sel_args is None:
            raise ValueError("Dominance coefficients could be set only with"
                             " selection coefficients.")
        self.mig_args = mig_args
        self.dyn_args = dyn_args
        super(Epoch, self).__init__()

        self.add_variable(time_arg)
        self.add_variables(init_size_args)
        self.add_variables(size_args)
        self.add_variables(sel_args)
        self.add_variables(dom_args)
        self.add_variables(dyn_args)
        self.add_variables(mig_args)

#    def set_value(self, variable, value):
#        # check dynamics first as they are more probable in our situation
#        for i, dyn_arg in enumerate(self.dyn_args):
#            if variable is dyn_arg:
#                self.dyn_args[i] = value
#                return
#        if variable is self.time_arg:
#            self.time_arg = value
#        for i, migs in enumerate(self.mig_args):
#            for j, mig_arg in enumerate(migs):
#                if variable is mig_arg:
#                    self.mig_args[i][j] = value
#                return
#        for i, sel_arg in enumerate(self.sel_args):
#            if variable is sel_arg:
#                self.sel_args[i] = value
#                return
#        raise ValueError(f"Event has such variable {variable}. "
#                         f"Available variables: {self.variables}")
#
    def as_custom_string(self, values):
        def _help_f(x, y):
            return f"{y}" if x == "" else f"{y}({x})"

        def help_f(x):
            return _help_f(*self._arg_val_repr(x, values))

        all_repr = [help_f(self.time_arg)]
        sizes_repr = [help_f(arg) for arg in self.size_args]
        sizes_repr = f"[{', '.join(sizes_repr)}]"
        all_repr.append(sizes_repr)
        migs_repr = "[no migs]"
        if self.mig_args is not None:
            migs_repr = [[help_f(mig)for mig in migs]
                         for migs in self.mig_args]
            migs_str = []
            for migs in migs_repr:
                migs_str.append(f"[{', '.join(migs)}]")
            migs_repr = f"[{', '.join(migs_str)}]"
        if self.n_pop > 1:
            all_repr.append(migs_repr)
        sels_repr = ""
        if self.sel_args is not None:
            sels_repr = [help_f(arg) for arg in self.sel_args]
            if self.dom_args is not None:
                dom_repr = [help_f(arg) for arg in self.dom_args]
                together = [f'{x}({y})' for x, y in zip(self.sel_args,
                                                        self.dom_args)]
                sels_repr = f"[{', '.join(together)}]"
            else:
                sels_repr = f"[{', '.join(sels_repr)}]"
            all_repr.append(sels_repr)
        if self.dyn_args is not None:
            dyns_repr = [help_f(arg) for arg in self.dyn_args]
            dyns_repr = f"[{', '.join(dyns_repr)}]"
        else:
            dyns_repr = f"[{', '.join(['Sud' for _ in self.size_args])}]"
        all_repr.append(dyns_repr)
        return f"[ {', '.join(all_repr)} ]"

    def get_vars_not_in_init_args(self):
        variables = copy.copy(self.variables)
        for arg in self.init_size_args:
            if isinstance(arg, Variable):
                if arg in variables:
                    variables.remove(arg)
            elif isinstance(arg, Model):
                rem_vars = []
                for var in variables:
                    if var in arg.variables:
                        rem_vars.append(var)
                for var in rem_vars:
                    variables.remove(var)
        return variables


class Split(Event):
    """
    Class for split demographic event.

    :param pop_to_div: Population index that splits.
    :param size_args: Sizes of populations after split.
    """
    def __init__(self, pop_to_div, size_args=None):
        # Simple checks
        if size_args is not None:
            assert pop_to_div < len(size_args)

        self.n_pop = len(size_args) - 1
        self.pop_to_div = pop_to_div
        self.size_args = size_args
        super(Split, self).__init__()
        self.add_variable(pop_to_div)
        self.add_variables(size_args)

    def as_custom_string(self, values):
        def _help_f(x, y):
            return f"{y}" if x == "" else f"{y}({x.replace(' ', '')})"

        def help_f(x):
            return _help_f(*self._arg_val_repr(x, values))

        frac_str = ""
        for var in self.variables:
            if isinstance(var, FractionVariable):
                val = self.var2value(values)[var]
                frac_str = f" {100 * val : .2f}% ({var.name})"
        sizes_repr = [help_f(arg) for arg in self.size_args]
        sizes_repr = f"[{', '.join(sizes_repr)}]"

        return f"[ {self.pop_to_div + 1} pop split {frac_str} {sizes_repr} ]"
