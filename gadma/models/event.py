import numpy as np

from . import Model
from ..utils import Variable


class Event(Model):
    def __init__(self):
        super(Event, self).__init__(raise_excep=False)

    def set_value(self, variable, value):
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
