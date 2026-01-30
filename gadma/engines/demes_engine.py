from . import Engine, register_engine
from ..models import EpochDemographicModel, StructureDemographicModel
from ..models import CustomDemographicModel, Split, Epoch
from collections import defaultdict
from .. import demes, demes_available
from .. import demesdraw, demesdraw_available
from .. import matplotlib


def get_demes_graph_from_dadi_func(
    model_func,
    params,
    Nanc,
    gen_time,
    pop_labels
):
    """
    Calls dadi to construct demes graph.
    :param model_func: Function with model for dadi
    :param params: Parameter values for dadi function
    :param Nanc: Size of ancestral population to do the rescaling
    :param gen_time: Time of one generation for rescaling
    :param pop_labels: List of MODERN populations' labels
    """
    import dadi
    # we need to run dadi function once to have the correct cash
    npop = len(pop_labels)
    ns, pts = [2 for _ in range(npop)], 10
    model_func(params, ns, pts)
    # we need to create a mapping for the last created populations,
    # otherwise they will have strange names
    # first we count "eras". This code is copied from dadi.
    # Dadi creates new labels and we just repeat after it to get
    # how many sets of new labels are going to be created
    era = 1
    for prev_i, e in enumerate(dadi.Demes.cache[1:]):
        if e.deme_ids is None:
            if isinstance(e, dadi.Demes.Split):
                era += 1
            elif isinstance(e, dadi.Demes.Remove):
                pass
            elif isinstance(e, dadi.Demes.Reorder):
                pass
            elif e.duration > 0 and dadi.Demes.cache[prev_i].duration > 0:
                era += 1
    mapping = {
        new_label: [f'd{era}_{ind+1}']
        for ind, new_label in enumerate(pop_labels)
    }
    try:
        g = dadi.Demes.output(
            Nref=Nanc,
            generation_time=gen_time,
            deme_mapping=mapping,
        )
    except Exception:
        # If we fail, maybe we can try without mapping
        g = dadi.Demes.output(
            Nref=Nanc,
            generation_time=gen_time,
            deme_mapping=None,
        )
    return g


class DemesEngine(Engine):
    """
    Engine for demes usage. It is not full engine. It supports any data and
    does nothing during its reading.

    The main purpose of this engine is to draw plots and generate
    code for models.
    """

    id = 'demes'  #:
    base_module = None  #:
    supported_models = [
        EpochDemographicModel,
        StructureDemographicModel,
        CustomDemographicModel
    ]  #:
    supported_data = None  # Any data is good, we do not need it #:
    inner_data_type = None  # base_module.Spectrum  #:

    can_evaluate = False
    can_simulate = False
    can_draw_model = True
    can_draw_comp = False

    @staticmethod
    def read_data(data_holder):
        pass

    def update_data_holder_with_inner_data(self):
        pass

    def build_demes_graph(self, values, nanc=None,
                          gen_time=None, gen_time_units="generations"):
        assert self.model is not None
        description = None

        # We will use labels of final populations to name their ancestors
        if (self.data_holder is not None and
                self.data_holder.population_labels is not None):
            pop_labels = list(self.data_holder.population_labels)
        else:
            n_pop = self.model.number_of_populations()
            pop_labels = [f"pop{i}" for i in range(n_pop)]
        pop_labels = [label.replace(" ", "_") for label in pop_labels]

        if not self.model.has_anc_size and nanc is None:
            # If model is given in genetic units, then we will still draw it
            # We will take Nanc as 1 and we need to translate time in genetic
            # units back, so we will need to divide it by 2 as they will be
            # ultiplied by 2 Nanc
            gen_time = 0.5
            gen_time_units = "genetic units"
            nanc = 1.0

            # if we have migration we might have a problem as demes does not
            # like when migrations are more than 1.0
            is_epoch_model = not isinstance(self.model, CustomDemographicModel)
            if is_epoch_model and len(pop_labels) >= 1:
                rate = 1
                var2value = self.model.var2value(values)
                for epoch in self.model.events:
                    if not isinstance(epoch, Epoch):
                        continue
                    if epoch.mig_args is not None:
                        for mig_row in epoch.mig_args:
                            mig_to_i = 0
                            for mig_el in mig_row:
                                if mig_el in var2value:
                                    mig_to_i += var2value[mig_el]
                            rate = max(rate, mig_to_i)
                if rate > 1:
                    rate = max(100, rate)
                    description = "WARNING: "\
                                  f"migration rates are divided by {rate}"
                    for epoch in self.model.events:
                        if not isinstance(epoch, Epoch):
                            continue
                        if epoch.mig_args is not None:
                            for mig_row in epoch.mig_args:
                                for mig_el in mig_row:
                                    if mig_el in var2value:
                                        var2value[mig_el] /= rate
                    values = [var2value[var] for var in self.model.variables]

        in_genetic_units = gen_time_units == "genetic units"

        # If we have a custom model we can work with it only if it is for dadi
        # If genetic units then Nanc should be None, dadi will scale migration
        if isinstance(self.model, CustomDemographicModel):
            graph = get_demes_graph_from_dadi_func(
                model_func=self.model.function,
                params=values,
                Nanc=None if in_genetic_units else nanc,
                gen_time=gen_time,
                pop_labels=pop_labels,
            )
            # we will change to genetic units for consistency
            if graph.time_units == "scaled":
                graph.time_units = "genetic units"
                graph.description = "WARNING: migration rates are normalized"
            return graph

        phys_values = self.model.translate_values(
            units="physical",
            values=values,
            Nanc=nanc,
            time_in_generations=True,
            rescale_back=False
        )
        var2value = self.model.var2value(phys_values)

        # useful function
        def get_value(entity):
            return self.model.get_value_from_var2value(var2value, entity)
        if self.model.has_anc_size:
            Nanc_size = get_value(self.model.Nanc_size)
        else:
            Nanc_size = nanc
        if (hasattr(self.model, "Nanc_size") and
                self.model.Nanc_size in var2value):
            var2value[self.model.Nanc_size] = Nanc_size
        whole_labels_list = list()
        # demes got time counted from nowdays so we will keep how deep we are
        last_time = 0
        # create demes builder
        builder = demes.Builder(
            description=description,
            time_units=gen_time_units,
            generation_time=gen_time,
            doi=None,
            defaults=None
        )
        if gen_time is None or in_genetic_units:
            gen_time = 1.0
        # We will create list of epochs dictionaries for each deme
        demes_epochs_dicts = defaultdict(list)
        # for each deme we keep its ancestor
        demes_ancestor = dict()
        # iterate by our events back in time and create epoch lists for demes
        for event in reversed(self.model.events):
            if isinstance(event, Split):
                new_label = pop_labels[event.pop_to_div] + "_" + pop_labels[-1]
                # put labels in labels list
                whole_labels_list.append(pop_labels[-1])
                whole_labels_list.append(pop_labels[event.pop_to_div])
                # remember ancestor
                demes_ancestor[pop_labels[-1]] = new_label
                demes_ancestor[pop_labels[event.pop_to_div]] = new_label
                # fix pop labels
                pop_labels[event.pop_to_div] = new_label
                pop_labels = pop_labels[:-1]
            else:
                event_time = self.model.get_value_from_var2value(
                    var2value,
                    event.time_arg
                )
                for i_pop in range(len(event.size_args)):
                    start_time = (last_time + event_time) * gen_time
                    end_time = last_time * gen_time
                    size_function = "constant"
                    start_size = get_value(event.init_size_args[i_pop])
                    end_size = get_value(event.size_args[i_pop])
                    if event.dyn_args is not None:
                        dynamic = get_value(event.dyn_args[i_pop])
                        if dynamic == "Sud":
                            # we should put valid values
                            start_size = end_size
                            size_function = "constant"
                        elif dynamic == "Lin":
                            size_function = "linear"
                        else:
                            assert dynamic == "Exp"
                            size_function = "exponential"
                    if size_function == "constant":
                        # we should put valid values
                        start_size = end_size
                    epoch_dict = {
                        "end_time": end_time,
                        "start_size": start_size,
                        "end_size": end_size,
                        "size_function": size_function,
                        "selfing_rate": 0,
                        "cloning_rate": 0
                    }
                    demes_epochs_dicts[pop_labels[i_pop]].append(epoch_dict)
                    if event.mig_args is not None:
                        for j_pop in range(len(event.size_args)):
                            if j_pop >= i_pop:
                                continue
                            mij = get_value(event.mig_args[i_pop][j_pop])
                            mji = get_value(event.mig_args[j_pop][i_pop])
                            if mij == mji and mij != 0:
                                pops = [pop_labels[i_pop], pop_labels[j_pop]]
                                builder.add_migration(
                                    rate=mij,
                                    demes=pops,
                                    source=None,
                                    dest=None,
                                    start_time=start_time,
                                    end_time=end_time
                                )
                            else:
                                if mij != 0:
                                    builder.add_migration(
                                        rate=mij,
                                        demes=None,
                                        source=pop_labels[j_pop],
                                        dest=pop_labels[i_pop],
                                        start_time=start_time,
                                        end_time=end_time
                                    )
                                if mji != 0:
                                    builder.add_migration(
                                        rate=mji,
                                        demes=None,
                                        source=pop_labels[i_pop],
                                        dest=pop_labels[j_pop],
                                        start_time=start_time,
                                        end_time=end_time
                                    )
                last_time += event_time
        # Now we add all demes including the first one
        Nanc_label = "ancestral"
        # pop_labels should change in prev loop and one population should
        # have been left
        if len(pop_labels) != 1:
            raise AssertionError(
                "Given model has several ancestral populations"
            )
        # We should check if ancestral population coincide with common
        # ancestor population
        if len(demes_epochs_dicts[pop_labels[-1]]) == 0:
            # we should put ancestral pop instead of last ancestor
            for pop_label in demes_ancestor:
                if demes_ancestor[pop_label] == pop_labels[-1]:
                    demes_ancestor[pop_label] = Nanc_label
        else:
            demes_ancestor[pop_labels[0]] = Nanc_label
            whole_labels_list.append(pop_labels[0])
        Nanc_epoch_dict = {
            "end_time": last_time * gen_time,
            "start_size": Nanc_size,
            "end_size": Nanc_size,
            "size_function": "constant",
            "selfing_rate": 0,
            "cloning_rate": 0
        }

        builder.add_deme(Nanc_label,
                         description=None,
                         ancestors=None,
                         proportions=None,
                         start_time=None,
                         epochs=[Nanc_epoch_dict],
                         defaults=None)
        for demes_name in reversed(whole_labels_list):
            # remove start_time from first epoch
            epochs = list(reversed(demes_epochs_dicts[demes_name]))
            builder.add_deme(demes_name,
                             description=None,
                             ancestors=[demes_ancestor[demes_name]],
                             proportions=None,
                             start_time=None,
                             epochs=epochs,
                             defaults=None)
        # resolve graph and get demes model
        graph = builder.resolve()
        return graph

    def generate_code(self, values, filename=None, nanc=None,
                      gen_time=None, gen_time_units="generations"):
        if filename is not None and not filename.endswith("yml"):
            filename = filename + ".yml"
        graph = self.build_demes_graph(
            values=values,
            nanc=nanc,
            gen_time=gen_time,
            gen_time_units=gen_time_units
        )
        demes_string = demes.dumps(graph)
        if filename is None:
            return demes_string
        with open(filename, 'w') as fl:
            fl.write(demes_string)

    def draw_schematic_model_plot(self, values, save_file=None,
                                  fig_title="Demographic Model from GADMA",
                                  nref=None, gen_time=1,
                                  gen_time_units="generations"):
        if not demesdraw_available:
            raise ValueError("Demesdraw is required to draw model plots with "
                             "demes engine.")
        graph = self.build_demes_graph(
            values=values,
            nanc=nref,
            gen_time=gen_time,
            gen_time_units=gen_time_units
        )

        fig, ax = demesdraw.utils.get_fig_axes(aspect=1, scale=0.8)
        demesdraw.tubes(
            graph,
            ax=ax,
            colours=None,
            log_time=False,
            title=fig_title,
            inf_ratio=0.2,
            positions=None,
            num_lines_per_migration=10,
            seed=None,
            labels='xticks-mid',
            fill=True,
            # We will draw scale bar only when units are translated
            scale_bar=(graph.time_units != "genetic units"),
        )
        fig.tight_layout()
        if save_file is None:
            matplotlib.pyplot.show()
        else:
            fig.savefig(save_file)
        return fig


if demes_available:
    register_engine(DemesEngine)
