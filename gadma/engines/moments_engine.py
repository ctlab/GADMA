from . import register_engine
from .dadi_moments_common import DadiOrMomentsEngine
from ..models import CustomDemographicModel, Epoch, Split
from ..utils import DynamicVariable, get_correct_dtype
from .. import SFSDataHolder, moments_available
import numpy as np


class MomentsEngine(DadiOrMomentsEngine):
    """
    Engine for using :py:mod:`moments` for demographic inference.

    Citation of :py:mod:`moments`:

    Jouganous, J., Long, W., Ragsdale, A. P., & Gravel, S. (2017).
    Inferring the joint demographic history of multiple populations:
    beyond the diffusion approximation. Genetics, 206(3), 1549-1567.
    """

    id = 'moments'  #:
    if moments_available:
        import moments as base_module
        inner_data_type = base_module.Spectrum  #:
    supported_data = [SFSDataHolder]  # , VCFDataHolder]  #:
    default_dt_fac = 0.01  #:
    can_draw = True

    @staticmethod
    def _get_kwargs(event, var2value):
        """
        Builds kwargs for moments.Spectrum.integrate functions.

        :param event: build for this event
        :type event: event.Epoch
        :param var2value: dictionary {variable: value}, it is required because
            the dynamics values should be fixed.
        """
        ret_dict = {'tf': event.time_arg}
        if event.dyn_args is not None:
            ret_dict['Npop'] = list()
            for i in range(event.n_pop):
                dyn_arg = event.dyn_args[i]
                dyn = MomentsEngine.get_value_from_var2value(var2value,
                                                             dyn_arg)
                if dyn == 'Sud':
                    ret_dict['Npop'].append(event.size_args[i])
                else:
                    ret_dict['Npop'].append('nu%d_func' % (i+1))
        else:
            ret_dict['Npop'] = event.size_args

        if event.mig_args is not None:
            ret_dict['m'] = np.zeros(shape=(event.n_pop, event.n_pop),
                                     dtype=object)
            for i in range(event.n_pop):
                for j in range(event.n_pop):
                    if i == j:
                        continue
                    ret_dict['m'][i, j] = event.mig_args[i][j]
        if event.sel_args is not None:
            ret_dict['gamma'] = event.sel_args
        if event.dom_args is not None:
            ret_dict['h'] = event.dom_args
        return ret_dict

    def _inner_func(self, values, ns, dt_fac=0.01):
        """
        Simulates expected SFS for proposed values of variables.

        :param values: values of variables
        :param ns: sample sizes of simulated SFS
        :param dt_fac: step size for numerical solution
        """
        var2value = self.model.var2value(values)

        if isinstance(self.model, CustomDemographicModel):
            values_list = [var2value[var] for var in self.model.variables]
            return self.model.function(values_list, ns)

        moments = self.base_module

        sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
        fs = moments.Spectrum(sts)

        ns_on_splits = [list(ns)]
        for ind, event in enumerate(reversed(self.model.events)):
            if isinstance(event, Split):
                p = event.pop_to_div
                ns_on_splits.append(ns_on_splits[-1][:-1])
                ns_on_splits[-1][p] += ns_on_splits[-2][-1]
        ns_on_splits = list(reversed(ns_on_splits))[1:]
        n_split = 0

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
                            addit_values['nu%d_func' % (i+1)] = func(
                                y1=y1,
                                y2=y2,
                                x_diff=x_diff)
                kwargs_with_vars = self._get_kwargs(event, var2value)
                kwargs = {}
                for x, y in kwargs_with_vars.items():
                    if x == 'm' and isinstance(y, np.ndarray):  # migration
                        l_s = np.array(y, dtype=get_correct_dtype(y))
                        for i in range(y.shape[0]):
                            for j in range(y.shape[1]):
                                yel = y[i][j]
                                l_s[i][j] = self.get_value_from_var2value(
                                    var2value, yel)
                                l_s[i][j] = addit_values.get(l_s[i][j],
                                                             l_s[i][j])
                        kwargs[x] = l_s
                    elif x == 'Npop':  # sizes
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
                    elif x in ['gamma', 'h']:
                        l_args = list()
                        for i in range(len(y)):
                            l_args.append(self.get_value_from_var2value(
                                    var2value, y[i]))
                        kwargs[x] = l_args
                    else:  # time
                        kwargs[x] = self.get_value_from_var2value(var2value, y)
                        kwargs[x] = addit_values.get(kwargs[x], kwargs[x])
                fs.integrate(dt_fac=dt_fac, **kwargs)
            elif isinstance(event, Split):
                ns_split = (ns_on_splits[n_split][event.pop_to_div],
                            ns_on_splits[n_split][-1])
                if event.n_pop == 1:
                    fs = moments.Manips.split_1D_to_2D(fs, *ns_split)
                else:
                    func_name = "split_%dD_to_%dD_%d" % (
                        event.n_pop, event.n_pop + 1, event.pop_to_div + 1)
                    fs = getattr(moments.Manips, func_name)(fs, *ns_split)
                n_split += 1
        return fs

    def draw_schematic_model_plot(self, values, save_file=None,
                                  fig_title="Demographic Model from GADMA",
                                  nref=None, gen_time=1,
                                  gen_time_units="Generations"):
        """
        Draws schematic plot of the model with values.
        See moments manual for more information.

        :param values: Values of the model parameters, it could be list of
                       values or dictionary {variable name: value}.
        :type values: list or dict
        :param save_file: File to save picture. If None then picture will be
                          displayed to the screen.
        :type save_file: str
        :param fig_title: Title of the figure.
        :type fig_title: str
        :param nref: An ancestral population size. If None then parameters
                     will be drawn in genetic units.
        :type nref: int
        :param gen_type: Time of one generation. Should be in units of
                         ``gen_time_units``.
        :type gen_type: float
        :param gen_time_units: Units of `gen_type`. For example, it
                               could be Years, Generations, Thousand Years and
                               so on.
        """
        moments = self.base_module
        # From moments docs about ns:
        # List of sample sizes to be passed as the ns argument to model_func.
        # Actual values do not matter, as long as the dimensionality is
        # correct. So we take small size for fast drawing.
        if self.data_holder is not None:
            n_pop = len(self.data_holder.projections)
            pop_labels = self.data_holder.population_labels
        else:
            n_pop = len(self.data.sample_sizes)
            pop_labels = self.data.pop_ids
        ns = [4 for _ in range(n_pop)]
        plot_mod = moments.ModelPlot.generate_model(self._inner_func,
                                                    values, ns)
        draw_scale = nref is not None
        show = save_file is None
        if nref is not None:
            nref = int(nref)
        moments.ModelPlot.plot_model(plot_mod,
                                     save_file=save_file,
                                     show=show,
                                     fig_title=fig_title,
                                     draw_scale=draw_scale,
                                     pop_labels=pop_labels,
                                     nref=nref,
                                     gen_time=gen_time,
                                     gen_time_units=gen_time_units,
                                     reverse_timeline=True)

    def simulate(self, values, ns, dt_fac=default_dt_fac):
        """
        Returns simulated expected SFS for :attr:`demographic_model` with
        values as parameters.

        :param values: Values of demographic model parameters.
        :param ns: sample sizes of the simulated SFS.
        """
        model = self._inner_func(values, ns, dt_fac)
        return model

    def get_theta(self, values, dt_fac=default_dt_fac):
        return super(MomentsEngine, self).get_theta(values, dt_fac)

    def evaluate(self, values, dt_fac=default_dt_fac):
        """
        Simulates SFS from values and evaluate log likelihood between
        simulated SFS and observed SFS.
        """
        try:
            val = super(MomentsEngine, self).evaluate(values, dt_fac)
            return val
        except RuntimeError as e:
            if str(e) == "Factor is exactly singular":
                return None
            raise e

    def generate_code(self, values, filename=None, dt_fac=default_dt_fac,
                      nanc=None, gen_time=None, gen_time_units=None):
        return super(MomentsEngine, self).generate_code(values, filename,
                                                        dt_fac, nanc, gen_time,
                                                        gen_time_units)


if moments_available:
    register_engine(MomentsEngine)
