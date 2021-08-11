import numpy as np

from . import register_engine
from . import Engine
from ..models import DemographicModel, StructureDemographicModel
from ..models import CustomDemographicModel, Epoch, Split
from .. import VCFDataHolder, moments_LD_available
from ..utils import DynamicVariable, get_correct_dtype
from os import listdir
import moments.LD
from moments.LD import LDstats


# This function will be transferred to utils very soon (after finishing)

def check_ld_data_holder(data_holder):
    """
    Check correctness of data for computing ld stored in data_holder,
    if find wrong data raises Errors or trying to fix.

    :param data_holder: holder of the data.
    :type data_holder: :class:`VCFDataHolder`
    """
    if data_holder.bed_file:
        extension = data_holder.bed_file[-4:]
        if extension != '.bed':
            raise FileExistsError("Check passed bed file. It doesn't have "
                                  ".bed extension.")

    if data_holder.bed_file and data_holder.bed_files_dir:
        raise ValueError('Single bed file and bed files directory '
                         'passed in the same time. Please, delete one of them '
                         'and try again.')

    if all(
        [
            data_holder.bed_files_dir,
            len(listdir(data_holder.bed_files_dir)) == 0
        ]
    ):
        raise ValueError('You passed empty bed files directory! '
                         'Check it.')

    elif len(listdir(data_holder.bed_files_dir)) == 1 and not data_holder.bed_file:
        data_holder.bed_file = (
            data_holder.bed_files_dir
            + "/"
            + listdir(data_holder.bed_files_dir)[0]
        )
        data_holder.bed_files_dir = None
        print("You've passed directory with single bed file.")

    elif data_holder.bed_files_dir:
        for bed_file in listdir(data_holder.bed_files_dir):
            if bed_file[-4:] != '.bed':
                raise FileExistsError('Bed files directory contains files with '
                                      'wrong extension. Please, delete unsupported files '
                                      'and try again.')

    if data_holder.ld_kwargs:
        if 'r_bins' in data_holder.ld_kwargs:
            data_holder.ld_kwargs['r_bins'] = eval(
                data_holder.ld_kwargs['r_bins']
            )
        elif ('r_bins' not in data_holder.ld_kwargs
              and data_holder.recombination_map is not None):
            raise ValueError("You didn't provided r_bins argument in "
                             "dictionary with arguments for computing LD stats "
                             "but provided recombination map! "
                             "Please check your param file and add r_bins.")

        if 'bp_bins' in data_holder.ld_kwargs:
            if isinstance(data_holder.ld_kwargs['bp_bins'], str):
                data_holder.ld_kwargs['bp_bins'] = eval(
                    data_holder.ld_kwargs['bp_bins']
                )

    if (data_holder.population_labels
            and 'pops' not in data_holder.ld_kwargs):
        data_holder.ld_kwargs.update({'pops': data_holder.population_labels})

    elif (data_holder.population_labels
          and 'pops' in data_holder.ld_kwargs):
        if data_holder.population_labels != data_holder.ld_kwargs['pops']:
            raise KeyError('Different population labels passed twice! '
                           'Check you param file and remove one of labels')

    return data_holder


class MomentsLD(Engine):
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
        import moments.LD as base_module
        inner_data_type = (base_module.LDstats, dict)

    can_draw = True
    can_evaluate = True
    can_simulate = True

    supported_models = [DemographicModel, StructureDemographicModel,
                        CustomDemographicModel]
    supported_data = [VCFDataHolder, LDstats, dict]

    # Genotype array?
    @classmethod
    def _read_data(cls, data_holder):
        """
        Reads LD statistics data from `data_holder`.

        :param data_holder: holder of the data.
        :type data_holder: :class:`VCFDataHolder`
        """
        data_holder = check_ld_data_holder(data_holder)

        assert isinstance(data_holder, VCFDataHolder)

        if data_holder.bed_files_dir:
            reg_num = 0
            region_stats = {}
            for file in listdir(data_holder.bed_files_dir):
                region_stats.update(
                    {
                        f"{reg_num}": moments.LD.Parsing.compute_ld_statistics(
                            data_holder.filename,
                            rec_map_file=data_holder.recombination_map,
                            pop_file=data_holder.popmap_file,
                            bed_file=f"{data_holder.bed_files_dir}/{file}",
                            **data_holder.ld_kwargs)
                    }
                )
                reg_num += 1
            # bootstrap data
            data = moments.LD.Parsing.bootstrap_data(region_stats)

        else:
            data = moments.LD.Parsing.compute_ld_statistics(
                data_holder.filename,
                rec_map_file=data_holder.recombination_map,
                pop_file=data_holder.popmap_file,
                bed_file=data_holder.bed_file,
                **data_holder.ld_kwargs
            )

            data = moments.LD.Parsing.means_from_region_data(
                {0: data}, data["stats"], norm_idx=0
            )

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
                dyn = MomentsLD.get_value_from_var2value(var2value,
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
            return 'Here is CustomDemographicModel'

        moments.LD = self.base_module
        try:
            rho = 4 * 10000 * eval(self.data_holder.ld_kwargs["r_bins"])
        except: # NOQA
            rho = 4 * 10000 * self.data_holder.ld_kwargs["r_bins"]
        theta = 0.001  # where take theta?

        ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
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

                ld_stats.integrate(**kwargs, rho=rho, theta=theta)

            elif isinstance(event, Split):
                ld_stats = ld_stats.split(event.n_pop - 1)
        # little data processing
        model = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(ld_stats[:-2], ld_stats[1:-1])]
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
        if self.data_holder.bed_files_dir:
            model = self.simulate(values)
            model = moments.LD.Inference.remove_normalized_lds(model)
            data = self.data
            means, varcovs = moments.LD.Inference.remove_normalized_data(
                data['means'],
                data['varcovs'],
                num_pops=model.num_pops)

            ll_model = self.base_module.Inference.ll_over_bins(means, model, varcovs)
            return ll_model
        else:
            raise ValueError('There are no way no compute likelihood with one region')

    def draw_ld_curves(self, values, save_file):
        """
        Draw plots of LD curves for observed and simulated by model data.

        :param values: Values of the model parameters, it could be list of
               values or dictionary {variable name: value}.
        :type values: list or dict
        :param save_file: File to save picture. If None then picture will be
                  displayed to the screen.
        :type save_file: str
        """
        data = self.read_data(self.data_holder)
        model = self.simulate(values=values)
        stats_to_plot = [
            [name] for name in model.names()[:-1][0] if name != 'pi2_0_0_0_0'
        ]
        # labels = 'Need some labels? If yes, need to prepare them'
        r_bins = self.data_holder.ld_kwargs['r_bins']
        if data['varcovs'] is None:
            self.base_module.Plotting.plot_ld_curves_comp(
                model,
                data[:-1],
                [],
                stats_to_plot=stats_to_plot,
                rs=r_bins,
                fig_size=(9, 9),
                show=save_file is None,
                rows=(len(stats_to_plot)/3),
                output=save_file,
                plot_means=True,
            )
        else:
            self.base_module.Plotting.plot_ld_curves_comp(
                model,
                data["means"][:-1],
                data["varcovs"][:-1],
                rs=r_bins,
                stats_to_plot=stats_to_plot,
                fig_size=(9, 9),
                show=save_file is None,
                rows=5,
                output=save_file,
                plot_means=True,
                plot_vcs=True
            )

    def generate_code(self, values, filename=None, nanc=None,
                      gen_time=None, gen_time_units="years"):
        pass


if moments_LD_available:
    register_engine(MomentsLD)
