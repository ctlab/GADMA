from . import Engine, register_engine, get_engine
from ..models import DemographicModel, StructureDemographicModel, \
    CustomDemographicModel, Epoch, Split
from ..models.coalescent_demographic_model import CoalescentDemographicModel
from ..models.event import Leaf, SetSize, MoveLineages
from ..utils import DynamicVariable, DiscreteVariable, Variable
from .. import SFSDataHolder, BinaryOperation
from .. import dadi_available, moments_available
from ..code_generator import id2printfunc

import warnings

momi_available = True


class MomiEngine(Engine):
    id = 'momi'
    if momi_available:
        import momi as base_module
        inner_data_type = base_module.Sfs
    supported_models = [CustomDemographicModel, CoalescentDemographicModel]  #:
    supported_data = [SFSDataHolder]  #:

    @classmethod
    def read_data(cls, data_holder):
        if data_holder.__class__ not in cls.supported_data:
            raise ValueError(f"Data class {data_holder.__class__.__name__}"
                             f" is not supported by {cls.id} engine.\nThe "
                             f"supported classes are: {cls.supported_data}"
                             f" and {cls.inner_data_type}")
        momi = cls.base_module
        data = momi.sfs_from_dadi(data_holder.filename)
        return data

    def _get_pop_name(self, index):
        if self.data is None:
            return "pop" + str(index)
        return self.data.population_label[index]

    def _inner_func(self, values):

        def _get_value(var):
            if isinstance(var, BinaryOperation):
                return var.get_value(values)
            return var2value.get(var, var)

        N_a = self.model.N_a

        var2value = self.model.var2value(values)
        for var in self.model.variables:
            var2value[var] = var.translate_value_into(units="physical",
                                                      value=var2value[var],
                                                      N_A=var2value.get(N_a, N_a),
                                                      gen_time=self.model.gen_time)
        model = None
        if isinstance(self.model, CustomDemographicModel):
            model = self.model.function(var2value)
        elif isinstance(self.model, CoalescentDemographicModel):
            model = self.base_module.DemographicModel(N_e=self.model.N_e,
                                                      gen_time=self.model.gen_time,
                                                      muts_per_gen=self.model.mu)
            for event in self.model.events:
                if isinstance(event, Leaf):
                    name = self._get_pop_name(event.pop)
                    model.add_leaf(pop_name=name,
                                   t=_get_value(event.t),
                                   N=_get_value(event.size_pop),
                                   g=_get_value(event.g))
                elif isinstance(event, MoveLineages):
                    name_pop_from = self._get_pop_name(event.pop_from)
                    name_pop_to = self._get_pop_name(event.pop)
                    model.move_lineages(pop_from=name_pop_from,
                                        pop_to=name_pop_to,
                                        t=_get_value(event.t),
                                        p=_get_value(event.p),
                                        N=_get_value(event.size_pop),
                                        g=_get_value(event.g))
                elif isinstance(event, SetSize):
                    name_pop = self._get_pop_name(event.pop)
                    model.set_size(pop_name=name_pop,
                                   t=_get_value(event.t),
                                   N=_get_value(event.size_pop),
                                   g=_get_value(event.g))
        return model

    def evaluate(self, values):
        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")
        model = self._inner_func(values)

        if self.model.sequence_length is None:
            raise ValueError("Please set sequence_length for the data")
        model.set_data(self.inner_data, length=self.model.sequence_length)
        return self.model.log_likelihood()

    def simulate(self, values, ns, num_replicates=1):
        if self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")

        sampled_n_dict = dict()
        numb_of_pop = len(ns)
        if self.data is None:
            warnings.warn("No name for population")

        for i in range(numb_of_pop):
            sampled_n_dict[self._get_pop_name(i)] = ns[i]

        model = self._inner_func(values)

        snp_counts = \
            model.simulate_data(length=self.model.sequence_length,
                                recoms_per_gen=self.model.rec_rate,
                                muts_per_gen=self.model.mu,
                                num_replicates=num_replicates,
                                sampled_n_dict=sampled_n_dict)
        return snp_counts.extract_sfs(n_blocks=100)

# if momi_available:
# register_engine(MomiEngine)
