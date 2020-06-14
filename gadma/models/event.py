import numpy as np

from . import Model
from ..utils import Variable


class Event(Model):
    def __init__(self):
        super(Event, self).__init__(raise_excep=False)

    def set_value(self, variable, value):
        raise NotImplementedError()

    def as_custom_string(self, values):
        raise NotImplementedError()

class Epoch(Event):
    def __init__(self, time_arg, init_size_args, size_args,
                 mig_args=None, dyn_args=None, sel_args=None):
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

        self.n_pop = len(init_size_args)
        self.time_arg = time_arg
        self.init_size_args = init_size_args
        self.size_args = size_args
        self.sel_args = sel_args
        self.mig_args = mig_args
        self.dyn_args = dyn_args
        super(Epoch, self).__init__()

        self.add_variable(time_arg)
        self.add_variables(init_size_args)
        self.add_variables(size_args)
        self.add_variables(sel_args)
        self.add_variables(dyn_args)
        self.add_variables(mig_args)

    def set_value(self, variable, value):
        # check dynamics first as they are more probable in our situation
        for i, dyn_arg in enumerate(self.dyn_args):
            if variable is dyn_arg:
                self.dyn_args[i] = value
                return
        if variable is self.time_arg:
            self.time_arg = value
        for i, migs in enumerate(self.mig_args):
            for j, mig_arg in enumerate(migs):
                if variable is mig_arg:
                    self.mig_args[i][j] = value
                return
        for i, sel_arg in enumerate(self.sel_args):
            if variable is sel_arg:
                self.sel_args[i] = value
                return

    def as_custom_string(self, values):
        var2value = self.var2value(values)
        val_str = lambda val: f"{val:5.3f}" if isinstance(val, float)\
                              else f"{val}"
        repr_f = lambda var: f"{val_str(var2value[var])}({var.name})"\
                             if isinstance(var, Variable)\
                             else f"{val_str(var)}"
        all_repr = [repr_f(self.time_arg)]
        sizes_repr = [repr_f(arg) for arg in self.size_args]
        sizes_repr = f"[{', '.join(sizes_repr)}]"
        all_repr.append(sizes_repr)
        migs_repr = "[no migs]" 
        if self.mig_args is not None:
            migs_repr = [[repr_f(mig )for mig in migs]
                         for migs in self.mig_args]
            migs_str = []
            for migs in migs_repr:
                migs_str.append(f"[{', '.join(migs)}]")
            migs_repr = f"[{', '.join(migs_str)}]"
        if self.n_pop > 1:
            all_repr.append(migs_repr)
        sels_repr = ""
        if self.sel_args is not None:
            sels_repr = [repr_f(arg) for arg in self.sel_args]
            sels_repr = f"[{', '.join(sels_repr)}]"
            all_repr.append(sels_repr)
        if self.dyn_args is not None:
            dyns_repr = [repr_f(arg) for arg in self.dyn_args]
            dyns_repr = f"[{', '.join(dyns_repr)}]"
        else:
            dyns_repr = "[{', '.join(['Sud' for _ in self.size_args])}]"
        all_repr.append(dyns_repr)
        return f"[ {', '.join(all_repr)} ]"


class Split(Event):
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
        var2value = self.var2value(values)
        val_str = lambda val: f"{val:5.3f}" if isinstance(val, float)\
                              else f"{val}"
        repr_f = lambda var: f"{val_str(var2value[var])}({var.name})"\
                             if isinstance(var, Variable)\
                             else f"{val_str(var)}"
        sizes_repr = [repr_f(arg) for arg in self.size_args]
        return f"[ {self.pop_to_div + 1} pop split [{', '.join(sizes_repr)}] ]"

