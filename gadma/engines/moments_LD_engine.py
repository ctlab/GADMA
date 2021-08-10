# Need some imports
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

def check_all_ld_data_correct(data_holder):
    data_holder_local = data_holder

    if data_holder_local.bed_file:
        extension = data_holder_local.bed_file[-4:]
        if extension != '.bed':
            raise FileExistsError("Check passed bed file. It doesn't have"
                                  ".bed extension.")

    if data_holder_local.bed_file and data_holder_local.bed_files_dir:
        raise ValueError('Single bed file and bed files directory'
                         ' passed in the same time. Please, delete one of them '
                         'and try again.')

    if data_holder_local.bed_files_dir:
        if not isinstance(data_holder_local.bed_files_dir, list):
        # need to read all file names and collect them in the list
            bed_files_list = listdir(data_holder_local.bed_files_dir)
            bed_files_list = [(data_holder_local.bed_files_dir + '/' + file) for file in bed_files_list]

            if not bed_files_list:
                raise ValueError("You passed empty directory as bed_files_dir!"
                                 "Please, check you dir. Delete bed_files_dir "
                                 "argument from param file if you don't want to use bed files"
                                 "and try again.")

            for bed_file in bed_files_list:
                if bed_file[-4:] != '.bed':
                    raise FileExistsError('Bed files directory contains files with'
                                          'wrong extension. Please, delete unsupported files'
                                          'and try again.')

            if len(bed_files_list) == 1 and not data_holder_local.bed_file:
                data_holder_local.bed_file = bed_files_list[0]
                data_holder_local.bed_files_dir = None

                print("You've passed directory with single bed file.")
        # else:
        #     data_holder_local.bed_files_dir = sorted(bed_files_list)

    if data_holder_local.ld_kwargs:
        if 'r_bins' in data_holder_local.ld_kwargs:
            data_holder_local.ld_kwargs['r_bins'] = eval(
                data_holder_local.ld_kwargs['r_bins']
            )
        elif ('r_bins' not in data_holder_local.ld_kwargs
              and data_holder_local.recombination_map is not None):
            raise ValueError("You didn't provided r_bins argument in "
                             "dictionary with arguments for computing LD stats "
                             "but provided recombination map! "
                             "Please check your param file and add r_bins.")

        if 'bp_bins' in data_holder_local.ld_kwargs:
            if isinstance(data_holder_local.ld_kwargs['bp_bins'], str):
                data_holder_local.ld_kwargs['bp_bins'] = eval(
                    data_holder_local.ld_kwargs['bp_bins']
                )

    return data_holder_local


class MomentsLD(Engine):
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

        data_holder = check_all_ld_data_correct(data_holder)

        assert isinstance(data_holder, VCFDataHolder)
        bed_file = data_holder.bed_file
        vcf_path = data_holder.filename
        recombibation_map = data_holder.recombination_map
        pop_file = data_holder.popmap_file
        bed_files_dir = data_holder.bed_files_dir
        kwargs = data_holder.ld_kwargs

        if data_holder.population_labels is not None and 'pops' not in kwargs:
            kwargs.update({'pops': data_holder.population_labels})
        elif data_holder.population_labels is not None and 'pops' in kwargs:
            if data_holder.population_labels != kwargs['pops']:
                raise KeyError('Different population labels passed twice! '
                               'Check you param file and remove one of labels')

        if bed_files_dir:
            iter = 0
            region_stats = {}
            for file in listdir(bed_files_dir):
                region_stats.update(
                    {
                        f"{iter}": moments.LD.Parsing.compute_ld_statistics(
                            vcf_path,
                            rec_map_file=recombibation_map,
                            pop_file=pop_file,
                            bed_file=f"{bed_files_dir}/{file}",
                            **kwargs)
                    }
                )
                iter += 1

            data = moments.LD.Parsing.bootstrap_data(region_stats)

        else:
            data = moments.LD.Parsing.compute_ld_statistics(
                vcf_path, rec_map_file=recombibation_map,
                pop_file=pop_file, bed_file=bed_file, **kwargs
            )

            data = moments.LD.Parsing.means_from_region_data(
                {0: data}, data["stats"], norm_idx=0
            )

        return data

    @staticmethod
    def _get_kwargs(event, var2value):
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
        Simulate expected LDstats for proposed values of variables

        :params_shamarms
        Generate function using API of moments.LD
        """

        var2value = self.model.var2value(values)

        if isinstance(self.model, CustomDemographicModel):
            return 'Here is CustomDemographicModel'

        moments.LD = self.base_module
        # rho from r_bins
        rho_m1 = 4 * 10000 * np.logspace(-6, -3, 7)
        theta_m1 = 0.001  # temporary

        ld = moments.LD.Numerics.steady_state(rho=rho_m1, theta=theta_m1)
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

                ld_stats.integrate(**kwargs, rho=rho_m1, theta=theta_m1)

            elif isinstance(event, Split):
                ld_stats = ld_stats.split(event.n_pop - 1)
        return ld_stats

    def evaluate(self, values, **options):

        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")
        if self.data_holder.bed_files_dir:
            model = self.simulate(values)

            model = moments.LD.LDstats(
                [(y_l + y_r) / 2 for y_l, y_r in zip(model[:-2], model[1:-1])]
                + [model[-1]],
                num_pops=model.num_pops,
                pop_ids=model.pop_ids,
            )
            model = moments.LD.Inference.sigmaD2(model)
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
        pass
        # data = self.read_data(self.data_holder)
        # model = self.simulate(values=values)
        #
        # if self.data_holder.bed_file:
        #     pass
        # moments.LD.Util.perturb_params(p_guess, fold=0.1)
        # uncerts = moments.LD.Godambe.GIM_uncert(
        #     demo_func, all_boot, opt_params, mv["means"], mv["varcovs"], r_edges=r_bins,
        # )
        # r_bins = self.data_holder.kwargs_for_computing_ld_stats['r_bins']
        # moments.LD.Plotting.plot_ld_curves_comp(
        #     model,
        #     data[:-1],
        #     [],
        #     rs=r_bins,
        #     # stats_to_plot=[
        #     #     ["DD_0_0", "DD_0_1", "DD_1_1"],
        #     #     ["Dz_0_0_0", "Dz_0_1_1", "Dz_1_1_1"],
        #     #     ["pi2_0_0_1_1", "pi2_0_1_0_1", "pi2_1_1_1_1"]
        #     # ],
        #     # labels=[[r"$D_0^2$", r"$D_0 D_1$", r"$D_1^2$"],
        #     #         [r"$Dz_{0,0,0}$", r"$Dz_{0,1,1}$", r"$Dz_{1,1,1}$"],
        #     #         [r"$\pi_{2;0,0,1,1}$", r"$\pi_{2;0,1,0,1}$", r"$\pi_{2;1,1,1,1}$"]
        #     #         ],
        #     plot_vcs=True,
        #     fig_size=(8, 3),
        #     show=True,
        # )

    def generate_code(self, values, filename=None, nanc=None,
                      gen_time=None, gen_time_units="years"):
        # last priority
        pass


if moments_LD_available:
    register_engine(MomentsLD)
