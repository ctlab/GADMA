from . import Engine, register_engine
from ..models import CustomDemographicModel, BinaryOperation
from ..models import TreeDemographicModel, EpochDemographicModel
from ..models import Leaf, PopulationSizeChange, LineageMovement
from ..data import check_and_return_projections_and_labels, read_popinfo
from .. import SFSDataHolder, VCFDataHolder
from .. import momi_available
from ..code_generator import id2printfunc

import warnings


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
        if (isinstance(data_holder, SFSDataHolder) and
                data_holder.filename.endswith("fs")):
            # it could be dadi format and in fsc format
            # TODO it is initial solution, it is bad
            data = momi.sfs_from_dadi(data_holder.filename)
        elif isinstance(data_holder, VCFDataHolder):
            # TODO check vcf pop labels and proj
            projections, populations = check_and_return_projections_and_labels(
                data_holder=data_holder
            )
            popmap, _ = read_popinfo(data_holder.popmap_file)
            data = momi.SnpAlleleCounts.read_vcf(
                vcf_file=data_holder.filename,
                ind2pop={sample: popmap[sample] for sample in popmap
                         if popmap[sample] in populations},
            ).extract_sfs(n_blocks=100)
            if data_holder.outgroup is False and not data.folded:
                data = data.fold()
        else:
            raise ValueError("Unknown type of data_holder: "
                             f"{data_holder.__class__}")
        return data

    def update_data_holder_with_inner_data(self):
        self.data_holder.projections = self.inner_data.sampled_n
        self.data_holder.population_labels = self.inner_data.sampled_pops
        self.data_holder.outgroup = not self.inner_data.folded

    def _get_pop_name(self, index, pop_labels=None):
        if pop_labels is not None:
            return pop_labels[index]
        if self.data_holder is None and self.inner_data is None:
            return "pop" + str(index)
        if self.inner_data is not None:
            return self.inner_data.sampled_pops[index]
        return self.data_holder.population_labels[index]

    def get_momi_model(self, values, pop_labels=None):
        """
        Returns momi's demographic model. Time is in generations.

        If pop_labels are None then labels from `_get_pop_name` are taken.
        """
        var2value = self.model.var2value(values)

        if isinstance(self.model, CustomDemographicModel):
            model = self.model.function(
                [var2value[var] for var in self.model.variables]
            )
            # This attributes may be set to different values so we change then
            model.muts_per_gen = self.model.mutation_rate
            model.gen_time = 1
            # model.set_params({var.name: var2value[var]
            #                   for var in self.model.variables})
            return model

        if isinstance(self.model, EpochDemographicModel):
            gadma_model, model_values = self.model.translate_to(
                TreeDemographicModel,
                values
            )
        else:
            gadma_model = self.model
            model_values = values

        Nanc_var = gadma_model.get_Nanc_variable(model_values)
        Nanc = gadma_model.get_value_from_var2value(var2value, Nanc_var)

        values = self.model.translate_values(
            units="physical",
            values=values,
            Nanc=Nanc
        )

        var2value = self.model.var2value(values)

        def get_value(entity):
            return self.model.get_value_from_var2value(var2value, entity)

        model = None

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
            # check for the dynamic
            dyn = self.model.get_value_from_var2value(var2value, event.dyn)
            if dyn == "Sud":
                g = 0
            else:
                assert dyn == "Exp", "Linear size change is not supported"
                g = get_value(event.g)

            N = get_value(event.size_pop)
            t = get_value(event.t)
            if isinstance(event, Leaf):
                name = self._get_pop_name(event.pop, pop_labels=pop_labels)
                model.add_leaf(
                    pop_name=name,
                    t=t,
                    N=N,
                    g=g,
                )
            elif isinstance(event, LineageMovement):
                name_pop_from = self._get_pop_name(
                    event.pop_from, pop_labels=pop_labels
                )
                name_pop_to = self._get_pop_name(
                    event.pop, pop_labels=pop_labels
                )
                p = get_value(event.p)
                model.move_lineages(
                    pop_from=name_pop_from,
                    pop_to=name_pop_to,
                    t=t,
                    p=p,
                    N=N,
                    g=g
                )
            elif isinstance(event, PopulationSizeChange):
                name_pop = self._get_pop_name(event.pop, pop_labels=pop_labels)
                model.set_size(
                    pop_name=name_pop,
                    t=t,
                    N=N,
                    g=g,
                )
        return model

    def get_N_ancestral(self, values):
        Nanc_var = self.model.get_Nanc_size(values)
        return self.get_value_from_var2value(
            entity=Nanc_var,
            var2value=self.model.var2value(values)
        )

    def evaluate(self, values):
        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")
        momi_model = self.get_momi_model(values)

        kwargs = {}
        if self.inner_data.length is None:
            assert self.data_holder is not None, ("Please set data holder with"
                                                  " sequence_length")
            kwargs['length'] = self.data_holder.sequence_length
        momi_model.set_data(self.inner_data, **kwargs)
        return momi_model.log_likelihood()

    def simulate(self, values, ns, sequence_length, population_labels,
                 num_replicates=1):
        if self.model is None:
            raise ValueError("Please set model for the engine or"
                             " use set_and_evaluate function instead.")
        if population_labels is None:
            population_labels = [self._get_pop_name(i) for i in range(len(ns))]

        assert len(ns) == len(population_labels)
        sampled_n_dict = dict(zip(population_labels, ns))

        model = self.get_momi_model(values, pop_labels=population_labels)

        snp_counts = model.simulate_data(
            length=sequence_length,
            recoms_per_gen=self.model.recombination_rate or 1e-8,
            muts_per_gen=self.model.mutation_rate,
            num_replicates=num_replicates,
            sampled_n_dict=sampled_n_dict
        )
        return snp_counts.extract_sfs(n_blocks=100)

    def generate_code(self, values, filename=None, nanc=None,
                      gen_time=None, gen_time_units="years"):
        """
        Generates code for momi2.
        """
        if self.data_holder is None:
            raise AttributeError("Engine was initialized with inner "
                                 "data. Need gadma.DataHolder for "
                                 "generation of code.")
        if filename is not None and not filename.endswith("py"):
            filename = filename + ".py"
        return id2printfunc[self.id](self, values,
                                     filename, nanc, gen_time,
                                     gen_time_units)


if momi_available:
    register_engine(MomiEngine)
