from . import Engine, register_engine
from .read_sfs_funcs import read_dadi_data
from ..models import DemographicModel, Epoch, Split
from ..utils import DynamicVariable
from .. import SFSDataHolder


class DadiEngine(Engine):
    """
    Engine for using :py:mod:`dadi` for demographic inference.

    Citation of :py:mod:`dadi`:

    Gutenkunst RN, Hernandez RD, Williamson SH, Bustamante CD (2009)
    Inferring the Joint Demographic History of Multiple Populations
    from Multidimensional SNP Frequency Data. PLoS Genet 5(10): e1000695.
    https://doi.org/10.1371/journal.pgen.1000695

    :param pts: list of grid points numbers for simulation in :py:mod:`dadi`.
    :type pts: list of three ints
    :param data: observed SFS data.
    :param model: demographic model.
    """

    id = 'dadi'  #:
    import dadi as base_module
    supported_models = [DemographicModel]  #:
    supported_data = [SFSDataHolder]  #:
    inner_data_type = base_module.Spectrum  #:

    @staticmethod
    def read_data(data_holder):
        """
        Reads SFS data from `data_holder`.

        Could read two types of data:

            * :py:mod:`dadi` SFS data type
            * :py:mod:`dadi` SNP data type

        Check :py:mod:`dadi` manual for additional information.

        :param data_holder: holder of the data.
        :type data_holder: :class:`SFSDataHolder`
        """
        if data_holder.__class__ not in DadiEngine.supported_data:
            raise ValueError(f"Data class {data_holder.__class__.__name__} is not supported by {self.id} engine.\nThe supported classes are: {self.supported_data} and {self.inner_data_type}")
        data = read_dadi_data(DadiEngine.base_module, data_holder)
        return data

    @staticmethod
    def _get_kwargs(event, var2value):
        """
        Builds kwargs for dadi.Integration functions (one_pop, two_pops,
        three_pops).

        :param event: build for this event
        :type event: event.Epoch
        :param var2value: dictionary {variable: value}, it is required because
            the dynamics values should be fixed.
        """
        ret_dict = {'T': event.time_arg}
        if event.n_pop == 1:
            ret_dict['nu'] = event.size_args[0]
            if event.sel_args is not None:
                ret_dict['gamma'] = event.sel_args[0]
            return ret_dict
        for i in range(event.n_pop):
            if event.dyn_args is not None:
                dyn_arg = event.dyn_args[i]
                if var2value.get(dyn_arg, dyn_arg) == 'Sud':
                    ret_dict['nu%d' % (i+1)] = event.size_args[i]
                else:
                    ret_dict['nu%d' % (i+1)] = 'nu%d_func' % (i+1)
            else:
                ret_dict['nu%d' % (i+1)] = event.size_args[i]

        if event.mig_args is not None:
            for i in range(event.n_pop):
                for j in range(event.n_pop):
                    if i == j:
                        continue
                    ret_dict['m%d%d' % (i+1, j+1)] = event.mig_args[i][j]
        if event.sel_args is not None:
            for i in range(event.n_pop):
                ret_dict['gamma%d' % (i+1)] = event.sel_args[i]
        return ret_dict

    def _dadi_inner_func(self, values, ns, pts):
        """
        Simulates expected SFS for proposed values of variables.

        :param values: values of variables
        :param ns: sample sizes of simulated SFS
        :param pts: grid points for numerical solution
        """
        self.model.set_values(values)
        dadi = self.base_module

        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)

        addit_values = {}
        for ind, event in enumerate(self.model.events):
            if isinstance(event, Epoch):
                if event.dyn_args is not None:
                    for i in range(event.n_pop):
                        dyn_arg = event.dyn_args[i]
                        value = self.model.get_value(dyn_arg)
                        if value != 'Sud':
                            func = DynamicVariable.get_func_from_value(value)
                            addit_values['nu%d_func' % (i+1)] = func(
                                y1=self.model.get_value(event.init_size_args[i]),
                                y2=self.model.get_value(event.size_args[i]),
                                x_diff=self.model.get_value(event.time_arg))
                kwargs_with_vars = self._get_kwargs(event, self.model.var2value)
                kwargs = {x: self.model.get_value(y) for x, y in kwargs_with_vars.items()}
                kwargs = {x: addit_values.get(y, y) for x, y in kwargs.items()}
                if event.n_pop == 1:
                    phi = dadi.Integration.one_pop(phi, xx, **kwargs)
                if event.n_pop == 2:
                    phi = dadi.Integration.two_pops(phi, xx, **kwargs)
                if event.n_pop == 3:
                    phi = dadi.Integration.three_pops(phi, xx, **kwargs)
            elif isinstance(event, Split):
                if event.n_pop == 1:
                    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
                else:
                    func_name = "phi_%dD_to_%dD_%d".format(
                        event.n_pop-1, event.n_pop, event.pop_to_div + 1)
                    phi = getattr(dadi.PhiManip, func_name)(xx, phi)
        sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
        return sfs

    def simulate(self, values, ns, pts):
        """
        Returns simulated expected SFS for :attr:`demographic_model` with
        values as parameters. Simulation is performed with :attr:`self.pts`
        as grid points for numerical solutions.

        :param values: values of demographic model parameters.
        :param ns: sample sizes of the simulated SFS.
        """
        dadi = self.base_module
        func_ex = dadi.Numerics.make_extrap_log_func(self._dadi_inner_func)
        model = func_ex(values, ns, pts)
        return model

    def evaluate(self, values, pts):
        """
        Simulates SFS from values and evaluate log likelihood between
        simulated SFS and observed SFS.
        """
        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or use set_and_evaluate function instead.")
        dadi = self.base_module
        model = self.simulate(values, self.data.sample_sizes, pts)
        ll_model = dadi.Inference.ll_multinom(model, self.data)
        return ll_model


register_engine(DadiEngine())
