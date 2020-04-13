from . import Engine, register_engine
from .read_sfs_funcs import read_dadi_data
from ..dem_model import Epoch, Split
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
    """

    id = 'dadi' #:
    import dadi as base_module
    supported_event_classes = [Epoch, Split] #:
    supported_data_classes = [SFSDataHolder] #:
    data_type = base_module.Spectrum #:

    def __init__(self, pts):
        self.pts = pts

# <Start block for data reading>
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
        if data_holder.__class__ not in DadiEngine.supported_data_classes:
            #TODO change exception to more narrow class and throughout the file
            raise Exception("Data class (%s) is not in supported classes of this engine.\nThe admissible classes are: %s." % (data.__class__, ', '.join([x.__name__ for x in DadiEngine.admissible_data_classes])))
        data_type, data = read_dadi_data(DadiEngine.base_module, data_holder)
        data_holder._data_type = data_type
        return data

    @staticmethod
    def get_pop_labels(data):
        return data.pop_ids

    @staticmethod
    def get_sample_sizes(data):
        return data.sample_sizes

    @staticmethod
    def get_outgroup(data):
        return not data.folded

    @staticmethod
    def get_seq_len(data):
        return None
# <End block for data reading>

    @staticmethod
    def _get_kwargs(event, var2value):
        """
        Builds kwargs for dadi.Integration functions (one_pop, two_pops, three_pops)

        :param event: build for this event
        :type event: event.Epoch
        :param var2value: dictionary (variable - value) we need to fix dynamics
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
                if event.get_value(dyn_arg, var2value) == 'Sud':
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
                    ret_dict['m%d%d' % (i+1,j+1)] = event.mig_args[i][j]
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
        dadi = self.base_module
        var2value = self.demographic_model.var2value(values)

        xx = dadi.Numerics.default_grid(pts)
        phi = dadi.PhiManip.phi_1D(xx)

        for ind, event in enumerate(self.demographic_model.events):
            if event.__class__ is Epoch:
                if event.dyn_args is not None:
                    for i in range(event.n_pop):
                        dyn_arg = event.dyn_args[i]
                        value = var2value.get(dyn_arg, dyn_arg)
                        if value != 'Sud':
                            func = DynamicVariable.get_func_from_value(value)
                            varname2value['nu%d_func' % (i+1)] = func(
                                y1=var2value.get(event.init_size_args[i], event.init_size_args[i]),
                                y2=var2value.get(event.size_args[i], event.size_args[i]),
                                x_diff=var2value.get(event.time_arg, event.time_arg))
                kwargs_with_vars = self._get_kwargs(event, var2value)
                kwargs = {x: var2value.get(y, y) for x, y in kwargs_with_vars.items()}
                str_kwargs = ["%s=%s" % (k, str(v)) for k, v in kwargs_with_vars.items()]
                if event.n_pop == 1:
                    func = dadi.Integration.one_pop(phi, xx, **kwargs)
                if event.n_pop == 2:
                    func = dadi.Integration.two_pops(phi, xx, **kwargs)
                if event.n_pop == 3:
                    func = dadi.Integration.three_pops(phi, xx, **kwargs)
            elif event.__class__ is Split:
                if event.n_pop == 1:
                    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
                else:
                    func_name = 'phi_%dD_to_%dD_%d' % (event.n_pop-1, event.n_pop, event.pop_to_div + 1)
                    phi = getattr(dadi.PhiManip, func_name)(xx, phi)

        sfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))
        return sfs

    def objective_function(self, values):
        """
        Simulates SFS from values and evaluate log likelihood between simulated SFS and observed SFS.
        """
        dadi = self.base_module
        func_ex = dadi.Numerics.make_extrap_log_func(self._dadi_inner_func)
        model =  func_ex(values, self.data_holder.sample_sizes, self.pts)
        ll_model = dadi.Inference.ll_multinom(model, self.data_holder.data)
        return ll_model

register_engine(DadiEngine)
