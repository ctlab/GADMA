from .variables_combinations import BinaryOperation, \
    Subtraction, \
    Division, \
    Multiplication, \
    operation_creation, \
    Exp
from ..utils import Variable
from .event import SetSize, MoveLineages, Leaf
from . import DemographicModel, EpochDemographicModel
import numpy as np


class CoalescentDemographicModel(DemographicModel):
    r"""
    Special class for coalescent demographic model.

    :param mu: Mutation rate per base per generation.
    :type mu: float
    :param gen_time: Time of one generation.
    :type gen_time: float
    :param sequence_length: Length of sequence.
    :type sequence_length: float
    :param rec_rate: Recombination rate per generation per base
    :type rec_rate: float
    :param linear_constrain: linear constrain on parameters.
    :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`

    :note: Model works properly provided that you have only one ancestral
     population, otherwise model behavior is undefined.
     Also leaf should be created at the same time.
    """

    def __init__(self,
                 mu=None,
                 gen_time=None,
                 sequence_length=None,
                 rec_rate=None,
                 linear_constrain=None):
        self.events = list()
        self.rec_rate = rec_rate
        self.gen_time = gen_time
        self.sequence_length = sequence_length
        theta0 = None
        if sequence_length is not None:
            theta0 = 4 * mu * sequence_length

        super(CoalescentDemographicModel, self).__init__(
            gen_time=gen_time,
            theta0=theta0,
            mu=mu,
            linear_constrain=linear_constrain
        )

    @staticmethod
    def get_value_from_var2value(var2value, entity):
        if isinstance(entity, Variable):
            return var2value[entity]
        if isinstance(entity, BinaryOperation):
            return entity.get_value(var2value)
        return entity

    @property
    def has_anc_size(self):
        """
        If True than model has ancestral population.
        """
        if len(self.events) == 0:
            return False
        return True

    @has_anc_size.setter
    def has_anc_size(self, val):
        pass

    def _get_Nanc_size(self, values):
        var2value = self.var2value(values)
        Nanc_var = self.get_Nanc_variable(var2value)
        return self.get_value_from_var2value(var2value, Nanc_var)

    def get_Nanc_variable(self, values):
        var2value = self.var2value(values)
        """
        Return size variable of ancestral population.

        :param value: Values of the parameters.
        :type value: list or dict
        """
        event = max(
            self.events,
            key=lambda x: self.get_value_from_var2value(var2value, x.t)
        )
        Nanc_var, sud = self._get_size_pop(
            event.pop,
            time=event.t,
            var2value=var2value
        )
        return Nanc_var

    def _get_size_pop(self, pop, time, var2value, inclusive=True):

        """
        Return size of population `pop` in time `time`.

        :param pop: Name of population.
        :type pop: int
        :param time: Time at which to find out the size.
        :type time: :class:`Variable`, :class:`gadma.models.BinaryOperation`,
         float
        :param var2value: Dictionary from variable to its value.
        :type: dict
        :param inclusive: True if event, which occurred
         at a point in time `time`,affects the returned size.
        :type inclusive: bool
        """
        time_val = self.get_value_from_var2value(
            var2value=var2value,
            entity=time)
        after_event = None
        after_time = None

        for event in self.events:
            if event.pop == pop and event.size_pop is not None:
                event_time = self.get_value_from_var2value(
                    var2value=var2value,
                    entity=event.t
                )
                if time_val > event_time and (after_event is None or
                                              event_time > after_time):
                    after_time = event_time
                    after_event = event
                if inclusive and np.isclose(time_val, event_time):
                    after_time = event_time
                    after_event = event

        if after_event is None:
            raise ValueError("Incorrect time")

        size = after_event.size_pop
        dyn = after_event.dyn
        g = after_event.g
        if dyn == "Sud":
            return size, dyn
        if dyn == "Lin":
            raise NotImplementedError("Cannot work with linear function.")
        if dyn == "Exp":
            # P = P_0 * e^{g*t}
            before_event = None
            before_time = None
            for event in self.events:
                if event.pop == pop or \
                        (isinstance(event, MoveLineages)
                         and event.pop_from == pop):
                    event_time = self.get_value_from_var2value(
                        var2value=var2value,
                        entity=event.t
                    )
                    if event_time > after_time and (before_event is None or
                                                    before_time > event_time):
                        before_time = event_time
                        before_event = event
            if before_event is None:
                return size, dyn
            duration = operation_creation(
                Subtraction,
                before_time,
                after_time,
            )
            start_size = operation_creation(
                Division,
                size,
                operation_creation(
                    Exp,
                    operation_creation(
                        Multiplication,
                        g,
                        duration
                    )
                )
            )
            new_size = operation_creation(
                Multiplication,
                start_size,
                operation_creation(
                    Exp,
                    operation_creation(
                        Multiplication,
                        g,
                        operation_creation(
                            Subtraction,
                            before_event.t,
                            time
                        )
                    )
                )
            )
            return new_size, dyn

    def _processing_epoch(self,
                          epoch_model,
                          epoch_events,
                          pop2pos,
                          size_args,
                          dyn_args,
                          var2value,
                          epoch_begin,
                          epoch_end):

        for epoch_event in epoch_events:
            if isinstance(epoch_event, MoveLineages):
                pop2pos[epoch_event.pop_from] = len(size_args)
                size, dyn = self._get_size_pop(
                    pop=epoch_event.pop_from,
                    time=epoch_begin,
                    var2value=var2value,
                    inclusive=False
                )
                size_args.append(size)
                dyn_args.append(dyn)

                size_args[pop2pos[epoch_event.pop]] = self._get_size_pop(
                    pop=epoch_event.pop,
                    time=epoch_begin,
                    var2value=var2value,
                    inclusive=False
                )[0]
                epoch_model.add_split(
                    pop_to_div=epoch_event.pop,
                    size_args=[size_args[pop2pos[epoch_event.pop]],
                               size_args[pop2pos[epoch_event.pop_from]]]
                )
        for pop, pos in pop2pos.items():
            size, dyn = self._get_size_pop(pop=pop,
                                           time=epoch_end,
                                           var2value=var2value)
            size_args[pos] = size
            dyn_args[pos] = dyn

        epoch_model.add_epoch(
            time_arg=operation_creation(
                Subtraction,
                epoch_begin,
                epoch_end
            ),
            size_args=list.copy(size_args),
            dyn_args=list.copy(dyn_args)
        )

    def translate_into(self, ModelClass, values):

        """
        Translate this model into its representation in `ModelClass`.

        :param ModelClass: Representation of model in which transform
         current model.
        :type ModelClass: Name of model class in which current model
         must be transformed.
        :param values: Values of model variables.
        :type values: list or dict
        """

        if issubclass(ModelClass, EpochDemographicModel):

            def is_from_one_epoch(x, y):
                return np.isclose(
                    self.get_value_from_var2value(
                        var2value=var2value,
                        entity=x
                    ),
                    self.get_value_from_var2value(
                        var2value=var2value,
                        entity=y
                    )
                )

            if not self.has_anc_size:
                raise ValueError("There should be at least one event in model")

            var2value = self.var2value(values)

            Nanc_var = self.get_Nanc_variable(var2value)
            Nanc = self._get_Nanc_size(values)

            epoch_model = EpochDemographicModel(
                gen_time=self.gen_time,
                theta0=self.theta0,
                mu=self.mu,
                linear_constrain=self.linear_constrain
            )

            if Nanc_var.units != "physical":
                raise ValueError(
                    "Nanc variable must be in physical units for translation"
                )

            values = self.translate_values(
                units="genetic",
                values=values,
                Nanc=Nanc
            )

            var2value = self.var2value(values)

            sorted_events = sorted(
                self.events,
                key=lambda x: self.get_value_from_var2value(
                    var2value=var2value,
                    entity=x.t
                ),
                reverse=True
            )

            pop2pos = dict()
            pop2pos[sorted_events[0].pop] = 0
            size_args = [Nanc_var]
            dyn_args = [sorted_events[0].dyn]

            cur_epoch_time = sorted_events[0].t
            epoch_events = []

            for event in sorted_events:
                if is_from_one_epoch(cur_epoch_time, event.t):
                    epoch_events.append(event)
                else:
                    self._processing_epoch(
                        epoch_model=epoch_model,
                        epoch_events=epoch_events,
                        pop2pos=pop2pos,
                        size_args=size_args,
                        dyn_args=dyn_args,
                        var2value=var2value,
                        epoch_begin=cur_epoch_time,
                        epoch_end=event.t
                    )
                    cur_epoch_time = event.t
                    epoch_events = [event]
            return epoch_model
        raise ValueError(
            f"Can not translate coalescent demographic model into {ModelClass}"
        )

    def change_pop_size(self, pop, t, size_pop=None, dyn='Sud', g=0):
        """
        Change population size and/or growth rate at time t.

        :param pop: A population that is changing in size.
        :type pop: int
        :param t: Time of the event
        :param size_pop: New size of population
        :param dyn: Dynamic of growth rate
        :type dyn: str
        :param g: Growth rate
        :type g: float

        :note: `size_pop` and `t` could contain variables of :class:`Variable`\
               class as well as different constants/values including\
               :class:`gadma.models.BinaryOperation` instances.
        """
        new_set_size = SetSize(
            pop=pop,
            t=t,
            dyn=dyn,
            size_pop=size_pop,
            g=g
        )
        self.events.append(new_set_size)
        self.add_variables(new_set_size.variables)

    def move_lineages(self, pop_from, pop, t, dyn='Sud',
                      size_pop=None, g=None, p=1):
        """
        Move each lineage in pop_from to pop at time t with probability p.

        :param pop_from: Population lineages are moved from (backwards in time)
        :type pop_from: int
        :param pop: Population lineages are moved to (backwards in time)
        :type pop: int
        :param t: Time of the event
        :param size_pop: New size of population `pop`
        :param dyn: Dynamic of growth rate of population `pop`
        :type dyn: str
        :param g: Growth rate of population `pop`
        :type g: float
        :param p: Probability that lineage in `pop_from` moves to `pop`
        :type p: float

        :note: `size_pop` and `t` could contain variables of :class:`Variable`\
               class as well as different constants/values including\
               :class:`gadma.models.BinaryOperation` instances.
        """
        new_move_lineages = MoveLineages(
            pop_from=pop_from,
            pop=pop,
            t=t,
            p=p,
            dyn=dyn,
            size_pop=size_pop,
            g=g
        )
        self.events.append(new_move_lineages)
        self.add_variables(new_move_lineages.variables)

    def add_leaf(self, pop, t=0, dyn='Sud', size_pop=None, g=None):
        """
        Add a sampled leaf population to the model.

        :param pop: Leaf population.
        :type pop: int
        :param t: Time of the event
        :param size_pop: New size of population `pop`
        :param dyn: Dynamic of growth rate of population `pop`
        :type dyn: str
        :param g: Growth rate of population `pop`
        :type g: float

        :note: `size_pop` and `t` could contain variables of :class:`Variable`\
               class as well as different constants/values including\
               :class:`gadma.models.BinaryOperation` instances.
        """
        new_leaf = Leaf(
            pop=pop,
            t=t,
            dyn=dyn,
            size_pop=size_pop,
            g=g
        )
        self.events.append(new_leaf)
        self.add_variables(new_leaf.variables)
