from . import Engine, register_engine, get_engine
from ..models import CustomDemographicModel, BinaryOperation
from ..models import TreeDemographicModel, EpochDemographicModel
from ..models import Leaf, PopulationSizeChange, LineageMovement
from ..data import check_and_return_projections_and_labels, read_popinfo
from ..data import get_list_of_names_from_vcf
from .. import SFSDataHolder, VCFDataHolder
from .. import momi_available, dadi_available, matplotlib_available
from ..code_generator import id2printfunc
import tempfile
import os
import copy
import warnings

if matplotlib_available:
    from matplotlib import pyplot as plt


class MomiEngine(Engine):
    id = 'momi2'
    if momi_available:
        import momi as base_module
        inner_data_type = base_module.Sfs
    supported_models = [CustomDemographicModel,
                        TreeDemographicModel,
                        EpochDemographicModel]  #:
    supported_data = [VCFDataHolder, SFSDataHolder]  #:
    can_evaluate = True
    can_draw_model = True
    can_draw_comp = False
    can_simulate = True

    @staticmethod
    def _get_ind2pop(vcf_data_holder, verbose=False):
        ind2pop = {}
        projections, populations = check_and_return_projections_and_labels(
            data_holder=vcf_data_holder,
            verbose=verbose,
        )
        popmap, _ = read_popinfo(vcf_data_holder.popmap_file)
        vcf_samples = get_list_of_names_from_vcf(vcf_data_holder.filename)
        for sample in vcf_samples:
            if sample not in popmap:
                continue
            if sample not in ind2pop and popmap[sample] in populations:
                ind2pop[sample] = popmap[sample]
        return ind2pop

    @classmethod
    def _read_data(cls, data_holder):
        momi = cls.base_module
        if isinstance(data_holder, SFSDataHolder):
            sfs_file = data_holder.filename
            if sfs_file.endswith(".fs") or sfs_file.endswith(".sfs"):
                data = momi.sfs_from_dadi(sfs_file)
            elif (sfs_file.endswith(".txt") or sfs_file.endswith(".obs")):
                if not dadi_available and not moments_available:
                    raise ValueError(
                        "Dadi or moments engine is required for reading SFS "
                        f"from file {sfs_file}"
                    )
                engine = get_engine("dadi" if dadi_available else "moments")
                readed_sfs = engine.read_data(data_holder)
                data = None
                try:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        filename = os.path.join(tmpdirname, "dadi_sfs.fs")
                        readed_sfs.to_file(filename)
                        data = momi.sfs_from_dadi(filename)
                except Exception as e:
                    raise ValueError(
                        f"Failed to read given file {data_holder.filename} "
                        f"using {engine.id} engine."
                    ) from e
            elif data_holder.filename.endswith(".gz"):
                data = momi.Sfs.load(data_holder.filename)
            else:
                raise SyntaxError(
                    "Input data filename extension is neither .fs (.sfs) or "
                    ".txt or .obs or .gz.")
        elif isinstance(data_holder, VCFDataHolder):
            projections, populations = check_and_return_projections_and_labels(
                data_holder=data_holder,
                verbose=True,
            )
            ind2pop = cls._get_ind2pop(data_holder)
            data = momi.SnpAlleleCounts.read_vcf(
                vcf_file=data_holder.filename,
                ind2pop=ind2pop,
            ).extract_sfs(n_blocks=100)
            data = data.subset_populations(populations)
        else:
            raise ValueError("Unknown type of data_holder: "
                             f"{data_holder.__class__}")
        if data_holder.outgroup is False and not data.folded:
            data = data.fold()
        if data_holder.population_labels is not None:
            data = data.subset_populations(data_holder.population_labels)
        if data_holder.projections is not None:
            if list(data_holder.projections) != list(data.sampled_n):
                warnings.warn("Momi cannot downsize SFS data.it is using data "
                              "with full size.")
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
        if isinstance(self.model, CustomDemographicModel):
            return None
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
        data_holder_exists = self.data_holder is not None
        if data_holder_exists:
            if self.data_holder.sequence_length is not None:
                kwargs['length'] = self.data_holder.get_total_sequence_length()
            sfs_data = isinstance(self.data_holder, SFSDataHolder)
            if sfs_data and self.data_holder.non_ascertained_pops is not None:
                pop_list = self.data_holder.non_ascertained_pops
                kwargs['non_ascertained_pops'] = pop_list
        # else:
        #     assert self.inner_data.length is None, ("Please set data holder"
        #                                             " with sequence_length")
        momi_model.set_data(self.inner_data, **kwargs)
        ll = None
        try:
            ll = momi_model.log_likelihood()
        except AssertionError as e:
            if not isinstance(self.model, CustomDemographicModel):
                raise e
        return ll

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

    def draw_schematic_model_plot(self, values, save_file=None,
                                  fig_title="Demographic Model from GADMA",
                                  nref=None, gen_time=1,
                                  gen_time_units="Generations"):
        original_model = self.model
        # if our model does not have ancestral size
        if not self.model.has_anc_size:
            assert nref is not None, ("nref should be set as model does not "
                                      "have ancestral size")
            # we copy our model and set ancestral size to nref
            self.model = copy.deepcopy(self.model)
            self.model.unfix_variable(self.model.Nanc_size)
            self.model.has_anc_size = True
            # and fix values to have new value for Nanc variable
            values = list(values)
            ind = self.model.variables.index(self.model.Nanc_size)
            values.insert(ind, nref)

        # get population labels
        if self.inner_data is not None:
            pop_labels = self.inner_data.sampled_pops
        else:
            assert self.data_holder is not None
            pop_labels = self.data_holder.population_labels

        # get momi model
        model = self.get_momi_model(values)
        model.gen_time = gen_time

        # draw plot
        self.base_module.DemographyPlot(
            model,
            pop_x_positions=pop_labels,
            figsize=(6, 8),
            linthreshy=None,
            pulse_color_bounds=(0, .25),
        )
        # save it or show
        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file)
        self.model = original_model

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
