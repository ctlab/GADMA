import numpy as np

class Event(object):
    def get_value(self, some_arg, varname2value):
        if isinstance(some_arg, Variable):
            return varname2value[some_arg.name]
        return some_arg

class Epoch(Event):
    def __init__(self, time_arg, init_size_args, size_args, mig_args=None, dyn_args=None, sel_args=None):
        self.n_pop = len(init_size_args)
        self.time_arg = time_arg
        self.init_size_args = init_size_args
        self.size_args = size_args
        self.sel_args = sel_args
        self.mig_args = mig_args
        self.dyn_args = dyn_args

class Split(Event):
    def __init__(self, pop_to_div, size_args=None):
        self.n_pop = len(size_args) - 1
        self.pop_to_div = pop_to_div
        self.size_args = size_args
