import numpy as np

from . import Model
from ..utils import Variable

class Event(Model):
    def __init__(self):
        super(Event, self).__init__(raise_excep=False)


class Epoch(Event):
    def __init__(self, time_arg, init_size_args, size_args, mig_args=None, dyn_args=None, sel_args=None):
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


class Split(Event):
    def __init__(self, pop_to_div, size_args=None):
        self.n_pop = len(size_args) - 1
        self.pop_to_div = pop_to_div
        self.size_args = size_args
        super(Split, self).__init__()
        self.add_variable(pop_to_div)
        self.add_variables(size_args)
