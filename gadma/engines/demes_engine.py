from . import Engine, register_engine
from ..models import EpochDemographicModel, StructureDemographicModel
from ..models import Split
from collections import defaultdict
from .. import demes, demes_available
from .. import demesdraw, demesdraw_available
from .. import matplotlib


class DemesEngine(Engine):
    """
    Engine for demes usage. It is not full engine. It supports any data and
    does nothing during its reading.

    The main purpose of this engine is to draw plots and generate
    code for models.
    """

    id = 'demes'  #:
    base_module = None  #:
    supported_models = [EpochDemographicModel, StructureDemographicModel]  #:
    supported_data = None  # Any data is good, we do not need it #:
    inner_data_type = None  # base_module.Spectrum  #:

    can_evaluate = False
    can_simulate = False
    can_draw = True

    @staticmethod
    def read_data(data_holder):
        pass

    def build_demes_graph(self, values, nanc=None,
                          gen_time=None, gen_time_units="generations"):
        assert self.model is not None
        if not self.model.has_anc_size and nanc is None:
            raise ValueError("Demographic model has no ancestral size variable"
                             " and no value for it was given to the function. "
                             "Demes engine requires Nanc size value to work.")
        phys_values = self.model.translate_values(
            units="physical",
            values=values,
            Nanc=nanc,
            time_in_generations=False,
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
        # We will use labels of final populations to name their ancestors
        pop_labels = list(self.data_holder.population_labels)
        whole_labels_list = list()
        # demes got time counted from nowdays so we will keep how deep we are
        last_time = 0
        # create demes builder
        builder = demes.Builder(
            description=None,
            time_units=gen_time_units,
            generation_time=gen_time,
            doi=None,
            defaults=None
        )
        if gen_time is None:
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
        assert len(pop_labels) == 1
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

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
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
            optimisation_rounds=None,
            labels='xticks-mid',
            fill=True
        )
        if save_file is None:
            matplotlib.pyplot.show()
        else:
            fig.savefig(save_file)


if demes_available:
    register_engine(DemesEngine)
