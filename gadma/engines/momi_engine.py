from . import Engine
from ..models import CustomDemographicModel, BinaryOperation
from ..models import TreeDemographicModel, EpochDemographicModel
from ..models import Leaf, PopulationSizeChange, LineageMovement
from .. import SFSDataHolder, VCFDataHolder

import warnings

momi_available = False


class MomiEngine(Engine):
    id = 'momi'
    if momi_available:
        import momi as base_module
        inner_data_type = base_module.Sfs
    supported_models = [CustomDemographicModel,
                        TreeDemographicModel,
                        EpochDemographicModel]  #:
    supported_data = [VCFDataHolder, SFSDataHolder]  #:
    can_evaluate = True
    can_draw = True
    can_simulate = True

    @classmethod
    def _read_data(cls, data_holder):
        momi = cls.base_module
        if isinstance(data, SFSDataHolder):
            # it could be dadi format and in fsc format
            # TODO it is initial solution, it is bad
            if data_holder.filename.ends_with("fs"):
                data = momi.sfs_from_dadi(data_holder.filename)
            else:
                # fsc format
                #TODO
                pass
        elif isinstance(data, VCFDataHolder):
            data = momi.read_vcf()

        data.length = data_holder.sequence_length
        assert data.length is None, "Sequence length should be set"
        return data

    def _get_pop_name(self, index):
        if self.data_holder is None:
            return "pop" + str(index)
        return self.data_holder.population_label[index]

    def get_momi_model(self, values):
        """
        Returns momi's demographic model. Time is in generations.
        """
        if isinstance(self.model, EpochDemographicModel):
            gadma_model = self.model.translate_to(TreeDemographicModel, values)
        else:
            gadma_model = self.model

        var2value = gadma_model.var2value(values)

        Nanc_var = gadma_model.get_Nanc_variable(values)
        Nanc = gadma_model.get_value_from_var2value(var2value, Nanc_var)

        values = gadma_model.translate_values(
            units="genetic",
            values=values,
            Nanc=Nanc
        )

        var2value = gadma_model.var2value(values)
        model = None
        if isinstance(gadma_model, CustomDemographicModel):
            model = gadma_model.function(var2value)
        else:
            assert isinstance(gadma_model, TreeDemographicModel)
            # Create momi model
            # Time is in generations
            # N_e - default size of population if other is not set,
            # it is ignored
            model = self.base_module.DemographicModel(
                N_e=1e4,
                gen_time=1,
                muts_per_gen=gadma_model.mutation_rate
            )
            for event in gadma_model.events:
                if isinstance(event, Leaf):
                    name = self._get_pop_name(event.pop)
                    model.add_leaf(
                        pop_name=name,
                        t=gadma_model.get_value_from_var2value(
                            var2value,
                            event.t
                        ),
                        N=gadma_model.get_value_from_var2value(
                            var2value,
                            event.size_pop
                        ),
                        g=gadma_model.get_value_from_var2value(
                            var2value,
                            event.g
                        )
                    )
                elif isinstance(event, LineageMovement):
                    name_pop_from = self._get_pop_name(event.pop_from)
                    name_pop_to = self._get_pop_name(event.pop)
                    model.move_lineages(
                        pop_from=name_pop_from,
                        pop_to=name_pop_to,
                        t=gadma_model.get_value_from_var2value(
                            var2value,
                            event.t
                        ),
                        p=gadma_model.get_value_from_var2value(
                            var2value,
                            event.p
                        ),
                        N=gadma_model.get_value_from_var2value(
                            var2value,
                            event.size_pop
                        ),
                        g=gadma_model.get_value_from_var2value(
                            var2value,
                            event.g
                        )
                    )
                elif isinstance(event, PopulationSizeChange):
                    name_pop = self._get_pop_name(event.pop)
                    model.set_size(
                        pop_name=name_pop,
                        t=gadma_model.get_value_from_var2value(
                            var2value,
                            event.t
                        ),
                        N=gadma_model.get_value_from_var2value(
                            var2value,
                            event.size_pop
                        ),
                        g=gadma_model.get_value_from_var2value(
                            var2value,
                            event.g
                        )
                    )
        return model

    def evaluate(self, values):
        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")
        model = self.get_momi_model(values)

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
            model.simulate_data(
                length=self.model.sequence_length,
                recoms_per_gen=self.model.rec_rate,
                muts_per_gen=self.model.mu,
                num_replicates=num_replicates,
                sampled_n_dict=sampled_n_dict
            )
        return snp_counts.extract_sfs(n_blocks=100)

    def generate_code(self, values, filename=None, nanc=None,
                      gen_time=None, gen_time_units="years"):
        pass

# if momi_available:
# register_engine(MomiEngine)
