import numpy as np
import collections
from . import register_engine
from . import Engine
from ..models import DemographicModel, StructureDemographicModel
from ..models import CustomDemographicModel, Epoch, Split
from .. import VCFDataHolder, moments_LD_available
from ..utils import DynamicVariable, get_correct_dtype
from ..utils import create_bed_files_and_extract_chromosomes
from ..code_generator import id2printfunc
from ..data import check_and_return_projections_and_labels
from os import listdir
import moments.LD


class MomentsLdEngine(Engine):
    """
    Engine for using :py:mod:`momentsLD` for demographic inference.

    Citation of :py:mod:`momentsLD`:

    Ragsdale Aaron P and Simon Gravel(2019)
    Models of archaic admixture and recent history from two-locus statistics
    PLoS Genetics, 15(6), e1008204.
    https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1008204

    Ragsdale Aaron P and Simon Gravel(2020)
    Unbiased estimation of linkage disequilibrium from unphased data
    Molecular Biology and Evolution 37.3 (2020): 923-932
    https://academic.oup.com/mbe/article/37/3/923/5614437
    """
    id = "momentsLD"

    if moments_LD_available:
        import moments.LD
        inner_data_type = dict

    can_draw = False
    can_evaluate = True
    can_simulate = True
    can_draw_comp = True

    supported_models = [DemographicModel, StructureDemographicModel,
                        CustomDemographicModel]
    supported_data = [VCFDataHolder, dict, moments.LD.LDstats_mod.LDstats]

    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    kwargs = {
        "r_bins": r_bins,
        "report": False,
        "bp_bins": np.array([ii for ii in range(0, 8275250, 1655050)]),
        "use_genotypes": True,
        "cM": True
    }

    @classmethod
    def _read_data(cls, data_holder):
        """
        Reads LD statistics data from `data_holder`.

        :param data_holder: holder of the data.
        :type data_holder: :class:`VCFDataHolder`
        """
        if isinstance(data_holder, VCFDataHolder):
            projections, pops = check_and_return_projections_and_labels(data_holder)
        else:
            raise ValueError("Wrong type of data_holder: "
                             f"{data_holder.__class__}")

        kwargs = cls.kwargs

        if data_holder.ld_kwargs:
            for key in data_holder.ld_kwargs:
                try:
                    kwargs[key] = eval(data_holder.ld_kwargs[key])
                except:  # NOQA
                    kwargs[key] = data_holder.ld_kwargs[key]

        chromosomes = create_bed_files_and_extract_chromosomes(data_holder)

        bed_files = data_holder.output_directory + "/bed_files/"
        rec_map = listdir(data_holder.recombination_maps)[0]
        extension = rec_map.split(".")[1]
        rec_map = rec_map.split(".")[0]
        rec_map = "_".join(rec_map.split('_')[:-1])
        reg_num = 0
        region_stats = {}
        if data_holder.recombination_maps is not None:
            if len(listdir(data_holder.recombination_maps)) == len(chromosomes):
                for chrom in chromosomes:
                    for num in range(1, chromosomes[chrom]):
                        region_stats.update({
                            f"{reg_num}":
                            moments.LD.Parsing.compute_ld_statistics(
                                data_holder.filename,
                                rec_map_file=f"{data_holder.recombination_maps}"
                                             f"/{rec_map}_{chrom}.{extension}",
                                pop_file=data_holder.popmap_file,
                                bed_file=f"{bed_files}/bed_file_{chrom}_{num}.bed",
                                pops=pops,
                                **kwargs
                            )
                        })
                        reg_num += 1

            elif len(listdir(data_holder.recombination_maps)) != len(chromosomes):
                rec_map = listdir(data_holder.recombination_maps)[0]
                for chrom in chromosomes:
                    for num in range(1, chromosomes[chrom]):
                        region_stats.update({
                            f"{reg_num}":
                            moments.LD.Parsing.compute_ld_statistics(
                                data_holder.filename,
                                rec_map_file=f"{data_holder.recombination_maps}"
                                             f"/{rec_map}",
                                map_name=f"{chrom}",
                                pop_file=data_holder.popmap_file,
                                bed_file=f"{bed_files}/bed_file_{chrom}_{num}.bed",
                                pops=pops,
                                **kwargs
                            )
                        })
                        reg_num += 1
        else:
            print("No recombination map provided, using physical distance")
            for chrom in chromosomes:
                for num in range(1, chromosomes[chrom]):
                    region_stats.update({
                        f"{reg_num}":
                        moments.LD.Parsing.compute_ld_statistics(
                            data_holder.filename,
                            pop_file=data_holder.popmap_file,
                            bed_file=f"{bed_files}/bed_file_{chrom}_{num}.bed",
                            pops=pops,
                            **kwargs
                        )
                    })
                    reg_num += 1
        data = moments.LD.Parsing.bootstrap_data(region_stats)
        return data

    @staticmethod
    def _get_kwargs(event, var2value):
        """
        Builds kwargs for moments.LD.Integration functions.

        :param event: build for this event
        :type event: event.Epoch
        :param var2value: dictionary {variable: value}, it is required because
            the dynamics values should be fixed.
        """
        ret_dict = {'tf': event.time_arg}

        if event.dyn_args is not None:
            ret_dict['nu'] = list()
            for i in range(event.n_pop):
                dyn_arg = event.dyn_args[i]
                dyn = MomentsLdEngine.get_value_from_var2value(var2value,
                                                               dyn_arg)

                if dyn == 'Sud':
                    ret_dict['nu'].append(event.size_args[i])
                else:
                    ret_dict['nu'].append(f'nu{i + 1}_func')
        else:
            ret_dict['nu'] = event.size_args

        if event.mig_args is not None:
            ret_dict['m'] = np.zeros(shape=(event.n_pop, event.n_pop),
                                     dtype=object)
            for i in range(event.n_pop):
                for j in range(event.n_pop):
                    if i == j:
                        continue
                    ret_dict['m'][i, j] = event.mig_args[i][j]
        return ret_dict

    def simulate(self, values):
        """
        Simulates expected LD statistics for proposed values of variables.

        :param values: Values of the model parameters, it could be list of
        values or dictionary {variable name: value}.
        :type values: list or dict
        """
        var2value = self.model.var2value(values)

        if isinstance(self.model, CustomDemographicModel):
            values_list = [var2value[var] for var in self.model.variables]
            return self.model.function(values_list)

        if self.data_holder.ld_kwargs:
            for key in self.data_holder.ld_kwargs:
                try:
                    self.kwargs[key] = eval(self.data_holder.ld_kwargs[key])
                except:  # NOQA
                    self.kwargs[key] = self.data_holder.ld_kwargs[key]
            self.r_bins = self.kwargs["r_bins"]

        Nref = self.model.get_value_from_var2value(
            var2value, self.model.Nanc_size)
        rhos = 4 * Nref * np.array(self.r_bins)
        theta = 4 * Nref * self.model.mutation_rate

        ld = moments.LD.Numerics.steady_state(rho=rhos, theta=theta)
        ld_stats = moments.LD.LDstats(ld, num_pops=1)

        addit_values = {}
        for ind, event in enumerate(self.model.events):
            if isinstance(event, Epoch):
                if event.dyn_args is not None:
                    for i in range(event.n_pop):
                        dyn_arg = event.dyn_args[i]
                        value = self.get_value_from_var2value(var2value,
                                                              dyn_arg)
                        if value != 'Sud':
                            func = DynamicVariable.get_func_from_value(value)
                            y1 = self.get_value_from_var2value(
                                var2value, event.init_size_args[i])
                            y2 = self.get_value_from_var2value(
                                var2value, event.size_args[i])
                            x_diff = self.get_value_from_var2value(
                                var2value, event.time_arg)
                            addit_values[f'nu{i + 1}_func'] = func(
                                y1=y1,
                                y2=y2,
                                x_diff=x_diff)
                kwargs_with_vars = self._get_kwargs(event, var2value)
                kwargs = {}
                for x, y in kwargs_with_vars.items():
                    if x == 'm' and isinstance(y, np.ndarray):
                        l_s = np.array(y, dtype=get_correct_dtype(y))
                        for i in range(y.shape[0]):
                            for j in range(y.shape[1]):
                                yel = y[i][j]
                                l_s[i][j] = self.get_value_from_var2value(
                                    var2value, yel)
                                l_s[i][j] = addit_values.get(l_s[i][j],
                                                             l_s[i][j])
                        kwargs[x] = l_s
                    elif x == 'nu':
                        n_pop_list = list()
                        all_sudden = True
                        for y1 in y:
                            n_pop_list.append(self.get_value_from_var2value(
                                var2value, y1))
                            if n_pop_list[-1] in addit_values:
                                all_sudden = False
                        if not all_sudden:
                            kwargs[x] = lambda t: [addit_values[el](t)
                                                   if el in addit_values
                                                   else el
                                                   for el in list(n_pop_list)]
                        else:
                            kwargs[x] = n_pop_list
                    else:
                        kwargs[x] = self.get_value_from_var2value(var2value, y)
                        kwargs[x] = addit_values.get(kwargs[x], kwargs[x])
                ld_stats.integrate(**kwargs, rho=rhos, theta=theta)

            elif isinstance(event, Split):
                ld_stats = ld_stats.split(event.pop_to_div)
        model = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(
                ld_stats[:-2], ld_stats[1:-1])]
            + [ld_stats[-1]],
            num_pops=ld_stats.num_pops,
            pop_ids=ld_stats.pop_ids,
        )
        model = moments.LD.Inference.sigmaD2(model)
        return model

    def evaluate(self, values, **options):
        """
        Simulates LD statistics from values and evaluate log likelihood between
        simulated LD statistics and observed LD statistics.

        :param values: Values of the model parameters, it could be list of
        values or dictionary {variable name: value}.
        :type values: list or dict
        """

        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")
        model = self.simulate(values)
        model = moments.LD.Inference.remove_normalized_lds(model)
        data = self.inner_data
        means, varcovs = moments.LD.Inference.remove_normalized_data(
            data['means'],
            data['varcovs'],
            num_pops=model.num_pops,
            normalization=0)
        ll_model = moments.LD.Inference.ll_over_bins(means, model, varcovs)
        return ll_model

    def draw_data_comp_plot(self, values, save_file, vmin=None):
        """
        Draw plots of LD curves for observed and simulated by model data.

        :param values: Values of the model parameters, it could be list of
               values or dictionary {variable name: value}.
        :type values: list or dict
        :param save_file: File to save picture. If None then picture will be
                  displayed to the screen.
        :type save_file: str
        """
        data = self.inner_data
        model = self.simulate(values)
        stats_to_plot = [
            [name] for name in model.names()[:-1][0] if name != 'pi2_0_0_0_0'
        ]

        labels_to_plot = []
        for stat in stats_to_plot:
            if stat[0].startswith('DD'):
                numbers = collections.Counter(stat[0].split("_")[1:])
                label = ""
                for num in numbers:
                    if numbers[num] > 1:
                        label += rf"$D_{num}^{numbers[num]}$ "
                    elif numbers[num] == 1:
                        label += rf"$D_{num}$ "
                labels_to_plot.append([label])
            elif stat[0].startswith('Dz'):
                core = stat[0].split("_")[0]
                numbers = ",".join(stat[0].split("_")[1:])
                numbers = "{" + f"{numbers}" + "}"
                label = [rf"${core}_{numbers}$"]
                labels_to_plot.append(label)
            elif stat[0].startswith('pi'):
                core = stat[0].split("_")[0][:2]
                numbers = ",".join(stat[0].split("_")[2:])
                pi_add = '2;'
                numbers = "{" + f"{pi_add}" + f"{numbers}" + "}"
                label = [rf"$\{core}_{numbers}$"]
                labels_to_plot.append(label)
        moments.LD.Plotting.plot_ld_curves_comp(
            model,
            data["means"][:-1],
            data["varcovs"][:-1],
            rs=np.array(self.r_bins),
            stats_to_plot=stats_to_plot,
            labels=labels_to_plot,
            fig_size=(len(stats_to_plot), 7),
            show=save_file is None,
            cols=round(len(stats_to_plot) / 3),
            output=save_file,
            plot_means=True,
            plot_vcs=True
        )

    def generate_code(self, values, filename, nanc=None,
                      gen_time=None, gen_time_units="years"):
        """
        Prints nice formated code in the format of engine to file or returns
        it as string if no file is set.

        :param values: values for the engine's model.
        :param filename: file to print code. If None then the string will
                         be returned.
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

    def get_N_ancestral(self, values):
        var2value = self.model.var2value(values)
        if self.model.Nanc_size:
            Nref = self.model.get_value_from_var2value(
                var2value, self.model.Nanc_size)
            return Nref
        else:
            raise ValueError(
                'Need ancestral population size as parameter'
            )

    def get_theta(self, values):
        var2value = self.model.var2value(values)
        if self.model.Nanc_size and self.model.mutation_rate:
            Nref = self.model.get_value_from_var2value(
                var2value, self.model.Nanc_size)
            theta = 4 * Nref * self.model.mutation_rate
        else:
            raise ValueError("Theta can't be calculated without"
                             "ancestral N size.")
        return theta

    def get_N_ancestral_from_theta(self, theta):
        nanc = theta / (4 * self.model.mutation_rate)
        return nanc

    def update_data_holder_with_inner_data(self):
        (self.data_holder.projections,
         self.data_holder.population_labels) = check_and_return_projections_and_labels(
            self.data_holder)
        self.data_holder.outgroup = None

if moments_LD_available:
    register_engine(MomentsLdEngine)
