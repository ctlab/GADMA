from .variables_combinations import BinaryOperation, Subtraction, operation_creation
from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import VariablePool, variables_values_repr
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable
from .event import SetSize, MoveLineages, Leaf
from . import DemographicModel, EpochDemographicModel
import numpy as np


class CoalescentDemographicModel(DemographicModel):
    default_p = 1
    default_init_g = None
    default_size_g = 0
    default_pop_size = None

    def __init__(self, mu, N_e=None, sequence_length=None, gen_time=None, rec_rate=None,
                 linear_constrain=None):
        self.events = list()
        self.N_e = N_e
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

    @staticmethod
    def get_value_from_var2value(var2value, entity):
        if isinstance(entity, Variable):
            return var2value[entity]
        if isinstance(entity, BinaryOperation):
            return entity.get_value(var2value)
        return entity

    @property
    def has_anc_size(self):
        if len(self.events) == 0:
            return False
        return True

    def get_Nanc_variable(self, var2value):
        event = max(self.events, key=lambda x: self.get_value_from_var2value(var2value, x.t))
        return event.size_pop

    def name2value(self, values):
        var2value = self.var2value(values)
        name2value = dict()
        for var, value in var2value.items():
            name2value[var.name] = value
        return name2value

    def _get_size_pop(self, pop, time, var2value, inclusive=True):

        """
        :param inclusive: -- true if only last event can be at time
        """
        time = self.get_value_from_var2value(var2value=var2value,
                                             entity=time)
        after_event = None
        after_time = None

        for event in self.events:
            if event.pop == pop and event.size_pop is not None:
                event_time = self.get_value_from_var2value(var2value=var2value,
                                                           entity=event.t)
                if after_event is None and time > event_time or \
                        time > event_time > after_time:
                    after_time = event_time
                    after_event = event
                if inclusive and np.isclose(time, event_time):
                    after_time = event_time
                    after_event = event

        assert (after_event is not None)

        delta_time = time - self.get_value_from_var2value(var2value=var2value,
                                                          entity=after_event.t)
        size = after_event.size_pop
        dyn = after_event.dyn
        g = after_event.g
        # p = last_size + g * t
        if dyn == "Sud":
            return size
        if dyn == "Lin":
            raise NotImplementedError("Cannot work with linear function.")
        if dyn == "Exp":
            # P = P_0 * e^{g*t}
            raise NotImplementedError("Cannot work with exponent function.")
        # start_size = size / np.exp(g * duration)
        # dt = duration - delta_time
        # TODO add groth rate
        # add_epoch(, [start_size * np.exp(g * dt), ]
        # return start_size * np.exp(g * dt)

    def _processing_epoch(self, epoch_model, epoch_events, pop2pos,
                          size_args, var2value, epoch_begin, epoch_end):

        for epoch_event in epoch_events:
            if isinstance(epoch_event, MoveLineages):
                pop2pos[epoch_event.pop_from] = len(size_args)
                size_args.append(self._get_size_pop(pop=epoch_event.pop_from,
                                                    time=epoch_begin,
                                                    var2value=var2value,
                                                    inclusive=False))

                size_args[pop2pos[epoch_event.pop]] = self._get_size_pop(pop=epoch_event.pop,
                                                                         time=epoch_begin,
                                                                         var2value=var2value,
                                                                         inclusive=False)
                epoch_model.add_split(pop_to_div=epoch_event.pop,
                                      size_args=[size_args[pop2pos[epoch_event.pop]],
                                                 size_args[pop2pos[epoch_event.pop_from]]])
        for pop, pos in pop2pos.items():
            size_args[pos] = self._get_size_pop(pop=pop,
                                                time=epoch_end,
                                                var2value=var2value)
        epoch_model.add_epoch(time_arg=operation_creation(epoch_begin, epoch_end, Subtraction),
                              size_args=list.copy(size_args))

    def translate_into(self, ModelClass, values):
        if ModelClass is EpochDemographicModel:

            def is_from_one_epoch(x, y):
                return np.isclose(self.get_value_from_var2value(var2value=var2value,
                                                                entity=x),
                                  self.get_value_from_var2value(var2value=var2value,
                                                                entity=y))

            if not self.has_anc_size:
                raise ValueError("There should be at least one event in model")

            var2value = self.var2value(values)

            Nanc_var = self.get_Nanc_variable(var2value)
            Nanc = self.get_value_from_var2value(var2value=var2value,
                                                 entity=Nanc_var)

            epoch_model = EpochDemographicModel(gen_time=self.gen_time,
                                                theta0=self.theta0,
                                                mu=self.mu,
                                                Nanc_variable=Nanc_var,
                                                linear_constrain=self.linear_constrain)

            if Nanc_var.units != "physical":
                raise ValueError("Nanc variable must be in physical units for translation")

            values = self.translate_values(units="genetic",
                                           values=values,
                                           Nanc=Nanc)

            var2value = self.var2value(values)

            sorted_events = sorted(self.events,
                                   key=lambda x: self.get_value_from_var2value(var2value=var2value,
                                                                               entity=x.t),
                                   reverse=True)

            pop2pos = dict()
            pop2pos[sorted_events[0].pop] = 0
            size_args = [Nanc_var]
            dyn_args = ['Sud']
            if sorted_events[0].g is not None:
                dyn_args = ['Exp']

            cur_epoch_time = sorted_events[0].t
            epoch_events = []

            for event in sorted_events:
                if is_from_one_epoch(cur_epoch_time, event.t):
                    epoch_events.append(event)
                else:
                    self._processing_epoch(epoch_model=epoch_model,
                                           epoch_events=epoch_events,
                                           pop2pos=pop2pos,
                                           size_args=size_args,
                                           var2value=var2value,
                                           epoch_begin=cur_epoch_time,
                                           epoch_end=event.t)
                    cur_epoch_time = event.t
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
