from .. import BinaryOperation
from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import VariablePool, variables_values_repr
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable
from .event import Model, SetSize, MoveLineages, Leaf
from . import DemographicModel, EpochDemographicModel
import numpy as np


class CoalescentDemographicModel(DemographicModel):
    default_p = 1
    default_init_g = None
    default_size_g = 0
    default_pop_size = None

    def __init__(self, N_e, mu, sequence_length=None, N_a=None, gen_time=None, rec_rate=None,
                 linear_constrain=None):
        self.events = list()
        self.N_e = N_e
        self.N_a = N_a
        self.has_Na = True
        self.rec_rate = rec_rate
        self.gen_time = gen_time
        self.sequence_length = sequence_length
        theta0 = None
        if sequence_length is not None:
            theta0 = 4 * mu * sequence_length

        super(CoalescentDemographicModel, self).__init__(gen_time=gen_time,
                                                         theta0=theta0,
                                                         mu=mu,
                                                         linear_constrain=linear_constrain)
        self.add_variable(N_a)

    def name2value(self, values):
        var2value = self.var2value(values)
        name2value = dict()
        for var, value in var2value.items():
            name2value[var.name] = value
        return name2value

    def _get_size_pop(self, pop, time, var2value):
        def f(var):
            if isinstance(var, BinaryOperation):
                return var.get_value(var2value)
            return var2value.get(var, var)

        after_event = None
        after_time = time

        for event in self.events:
            if event.pop == pop:
                if after_time < f(event.t) < time or \
                        after_event is None and f(event.t) < time:
                    after_time = f(event.t)
                    after_event = event

        # incorrect time
        if after_event is None:
            for event in self.events:
                if event.pop == pop:
                    if f(event.t) < after_time or \
                            after_event is None:
                        after_time = f(event.t)
                        after_event = event
        assert (after_event is not None)
        delta_time = time - f(after_event.t)
        size = f(after_event.size_pop)
        dyn = f(after_event.dyn)
        g = f(after_event.g)
        end2dur = self._get_epoch_duration(var2value=var2value)
        duration = end2dur[f(after_event.t)]
        # p = last_size + g * t
        if dyn == "Sud":
            return size
        if dyn == "Lin":
            raise NotImplementedError("Cannot work with linear function.")
        if dyn == "Exp":
            # P = P_0 * e^{g*t}
            start_size = size / np.exp(g * duration)
            dt = duration - delta_time
            # TODO add groth rate
            # add_epoch(, [start_size * np.exp(g * dt), ]
            return start_size * np.exp(g * dt)

    def _get_epoch_duration(self, var2value):
        def f(var):
            if isinstance(var, BinaryOperation):
                return var.get_value(var2value)
            return var2value.get(var, var)

        times = []
        for event in self.events:
            times.append(f(event.t))
        start2dur = dict()
        end2dur = dict()
        times.sort(reverse=True)
        cur_time = times[0]
        end2dur[cur_time] = 0
        for t in times:
            if t != cur_time:
                start2dur[cur_time] = cur_time - t
                end2dur[t] = cur_time - t
                cur_time = t
        return end2dur

    def _processing_epoch(self, epoch_model, epoch_events, pop2pos, size_args, var2value, epoch_begin, epoch_end):
        for epoch_event in epoch_events:
            if isinstance(epoch_event, MoveLineages):
                pop2pos[epoch_event.pop_from] = len(size_args)
                size_args.append(self._get_size_pop(pop=epoch_event.pop_from,
                                                    time=epoch_begin,
                                                    var2value=var2value))

                size_args[pop2pos[epoch_event.pop]] = self._get_size_pop(pop=epoch_event.pop,
                                                                         time=epoch_begin,
                                                                         var2value=var2value)
                epoch_model.add_split(pop_to_div=epoch_event.pop,
                                      size_args=[size_args[pop2pos[epoch_event.pop]],
                                                 size_args[pop2pos[epoch_event.pop_from]]])
        for pop, pos in pop2pos.items():
            size_args[pos] = self._get_size_pop(pop=pop,
                                                time=epoch_end,
                                                var2value=var2value)
        epoch_model.add_epoch(time_arg=epoch_begin - epoch_end,
                              size_args=size_args)

    def translate_into(self, ModelClass, values):
        def f(variable):
            if isinstance(variable, BinaryOperation):
                return variable.get_value(values)
            return var2value.get(variable, variable)

        if ModelClass is EpochDemographicModel:
            if self.N_a is None:
                raise ValueError("Set N_a value for translation")

            epoch_model = EpochDemographicModel(gen_time=self.gen_time,
                                                theta0=self.theta0,
                                                mu=self.mu,
                                                linear_constrain=self.linear_constrain)
            var2value = self.var2value(values)

            # set all variables in genetic
            for var in self.variables:
                var2value[var] = var.translate_value_into(units="genetic",
                                                          value=var2value[var],
                                                          N_A=f(self.N_a),
                                                          gen_time=self.gen_time)

            sorted_events = sorted(self.events,
                                   key=lambda x: f(x.t),
                                   reverse=True)
            num_of_event = len(sorted_events)
            if num_of_event == 0:
                return epoch_model

            pop2pos = dict()
            pop2pos[sorted_events[0].pop] = 0
            size_args = [f(sorted_events[0].size_pop)]
            cur_epoch_time = f(sorted_events[0].t)
            epoch_events = []
            for event in sorted_events:
                if cur_epoch_time == f(event.t):
                    epoch_events.append(event)
                else:
                    print(epoch_events)
                    self._processing_epoch(epoch_model=epoch_model,
                                           epoch_events=epoch_events,
                                           pop2pos=pop2pos,
                                           size_args=size_args,
                                           var2value=var2value,
                                           epoch_begin=cur_epoch_time,
                                           epoch_end=f(event.t))
                    cur_epoch_time = f(event.t)
                    epoch_events = [event]
            return epoch_model
        raise ValueError(f"Can not translate coalescent demographic model into {ModelClass}")

    def change_pop_size(self, pop, t, size_pop=None, dyn='Sud', g=0):
        new_set_size = SetSize(pop=pop,
                               t=t,
                               dyn=dyn,
                               size_pop=size_pop,
                               g=g)
        self.events.append(new_set_size)
        self.add_variables(new_set_size.variables)

    def move_lineages(self, pop_from, pop, t, p=1, dyn='Sud',
                      size_pop=None, g=None):
        new_move_lineages = MoveLineages(pop_from=pop_from,
                                         pop=pop,
                                         t=t,
                                         p=p,
                                         dyn=dyn,
                                         size_pop=size_pop,
                                         g=g)
        self.events.append(new_move_lineages)
        self.add_variables(new_move_lineages.variables)

    def add_leaf(self, pop, t=0, dyn='Sud', size_pop=None, g=None):
        new_leaf = Leaf(pop=pop,
                        t=t,
                        dyn=dyn,
                        size_pop=size_pop,
                        g=g)
        self.events.append(new_leaf)
        self.add_variables(new_leaf.variables)
