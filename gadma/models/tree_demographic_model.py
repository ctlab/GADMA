from .variables_combinations import Subtraction, \
    Division, \
    Multiplication, \
    operation_creation, \
    Exp

from .event import PopulationSizeChange, LineageMovement, Leaf
from . import DemographicModel, EpochDemographicModel
import numpy as np

from ..utils import Variable


class TreeDemographicModel(DemographicModel):
    """
    Special class for tree demographic model. This demographic model is very
    close to momi2 models. It is back-in-time and has tree structure.

    :param mutation_rate: Mutation rate per base per generation.
    :type mutation_rate: float
    :param gen_time: Time of one generation.
    :type gen_time: float
    :param recombination_rate: Recombination rate per generation per base
    :type recombination_rate: float
    :param linear_constrain: linear constrain on parameters.
    :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`

    :note: Model works properly provided that you have only one ancestral
     population, otherwise model behavior is undefined.
     Also leaf should be created at the same time.
    """

    def __init__(self,
                 gen_time=None,
                 mutation_rate=None,
                 recombination_rate=None,
                 theta0=None,
                 linear_constrain=None):
        self.events = list()
        self.rec_rate = recombination_rate
        self.gen_time = gen_time

        super(TreeDemographicModel, self).__init__(
            gen_time=gen_time,
            theta0=theta0,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            linear_constrain=linear_constrain
        )

    def __eq__(self, other):
        """
        Checks that it is the same model with the same events.
        If structure is different but it is still the same model then use
        :meth:`equals`.
        """
        if self is other:
            return True
        if not isinstance(other, TreeDemographicModel):
            return False
        if len(self.events) != len(other.events):
            return False
        for event in self.events:
            has_same_event = False
            for other_event in other.events:
                if event == other_event:
                    has_same_event = True
                    break
            if not has_same_event:
                return False
        return True

    def equals(self, other, values):
        """
        Checks that two models are equal ignoring their structure.
        Just checks that all values in events are equal.
        """
        if self is other:
            return True
        if not isinstance(other, TreeDemographicModel):
            return False
        if len(self.events) != len(other.events):
            return False

        event_visited = [False for _ in self.events]

        for i, other_event in enumerate(other.events):
            for event in self.events:
                if event == other_event or event.equals(other_event, values):
                    event_visited[i] = True
        return event_visited == [True for _ in self.events]

    @classmethod
    def create_from(cls, model, values):
        if isinstance(model, EpochDemographicModel):
            return model.translate_to(cls, values)[0]
        raise ValueError(
            f"Cannot translate to {model.__class__}"
        )

    @property
    def has_anc_size(self):
        """
        Returns True if model is not empty. It is assumed that such models
        always have an ancestral population.
        """
        if len(self.events) == 0:
            return False
        return True

    @has_anc_size.setter
    def has_anc_size(self, val):
        """
        Does nothing as has_anc_size is always True.
        """
        pass

    def _get_Nanc_size(self, values):
        """
        Returns value of ancestral population size.
        """
        var2value = self.var2value(values)
        Nanc_var = self.get_Nanc_variable(var2value)
        return self.get_value_from_var2value(var2value, Nanc_var)

    def get_Nanc_variable(self, values):
        """
        Returns variable corresponding to the ancestral population size.

        :param value: Values of the parameters.
        :type value: list or dict
        """
        var2value = self.var2value(values)
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

    def number_of_populations(self):
        return sum([isinstance(event, Leaf) for event in self.events])

    def _get_size_pop(self, pop, time, var2value, inclusive=True):

        """
        Returns size of population `pop` at the moment of time `time`.

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
            entity=time
        )
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
        dyn_value = self.get_value_from_var2value(var2value, dyn)
        g = after_event.g
        if dyn_value == "Sud":
            return size, dyn
        if dyn_value == "Lin":
            raise NotImplementedError("Cannot work with linear function.")
        if dyn_value == "Exp":
            # P = P_0 * e^{g*t}
            before_event = None
            before_time = None
            for event in self.events:
                if event.pop == pop or \
                        (isinstance(event, LineageMovement)
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
            if isinstance(epoch_event, LineageMovement):
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
                    size_args=list.copy(size_args)
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

    def translate_to(self, ModelClass, values):

        """
        Translate this model into its representation in `ModelClass`.
        Supports :class:`gadma.EpochDemographicModel` only.

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
                Nanc_size=Nanc_var,
                mutation_rate=self.mutation_rate,
                recombination_rate=self.recombination_rate,
                linear_constrain=self.linear_constrain
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
            f"Can not translate tree demographic model into {ModelClass}"
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
        new_set_size = PopulationSizeChange(
            pop=pop,
            t=t,
            dyn=dyn,
            size_pop=size_pop,
            g=g
        )
        self.events.append(new_set_size)
        self.add_variables(new_set_size.variables)

    # TODO g = 0, explaining
    def move_lineages(self, pop_from, pop, t, dyn='Sud',
                      size_pop=None, g=0, p=1):
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
        new_move_lineages = LineageMovement(
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

    def add_leaf(self, pop, t=0, dyn='Sud', size_pop=None, g=0):
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
