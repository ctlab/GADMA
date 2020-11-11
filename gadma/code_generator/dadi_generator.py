from ..models import CustomDemographicModel, Epoch, Split, BinaryOperation
from ..utils import Variable, DiscreteVariable, DynamicVariable
import sys

FUNCTION_NAME = 'model_func'


def _print_dadi_func(model, values):
    """
    Returns string with function of demographic model for :py:mod:`dadi`.
    Parameter `values` is needed to fix dynamics of populations.

    :param model: Demographic model.
    :type model: :class:`gadma.models.model.Model`
    :param values: List of values for parameters of model.
    """
    from ..engines import DadiEngine  # to avoid cross import
    if isinstance(model, CustomDemographicModel):
        ret_str = "import importlib.util\n\n"
        ret_str += "spec = importlib.util.spec_from_file_location('module',"\
                   f" '{sys.modules[model.function.__module__].__file__}')\n"
        ret_str += "module = importlib.util.module_from_spec(spec)\n"
        ret_str += "spec.loader.exec_module(module)\n"
        ret_str += f"{FUNCTION_NAME} = module.model_func\n\n"
        return ret_str

    var2value = model.var2value(values)  # OrderedDict

    f_vars = [x for x in model.variables
              if not isinstance(x, DiscreteVariable)]
    ret_str = f"def {FUNCTION_NAME}(params, ns, pts):\n"
    ret_str += "\t%s = params\n" % ", ".join([x.name for x in f_vars])
    ret_str += "\txx = dadi.Numerics.default_grid(pts)\n"\
               "\tphi = dadi.PhiManip.phi_1D(xx)\n"
    for ind, event in enumerate(model.events):
        if event.__class__ is Epoch:
            if event.dyn_args is not None:
                for i in range(event.n_pop):
                    dyn_arg = event.dyn_args[i]
                    value = var2value.get(dyn_arg, dyn_arg)
                    if value != 'Sud':
                        func = DynamicVariable._help_dict[value]
                        y1 = event.init_size_args[i]
                        y2 = event.size_args[i]
                        x_diff = event.time_arg
                        if isinstance(y1, Variable):
                            y1 = y1.name
                        elif isinstance(y1, BinaryOperation):
                            y1 = f"({y1.name})"
                        if isinstance(y2, Variable):
                            y2 = y2.name
                        elif isinstance(y2, BinaryOperation):
                            y2 = f"({y2.name})"
                        if isinstance(x_diff, Variable):
                            x_diff = x_diff.name
                        elif isinstance(x_diff, BinaryOperation):
                            x_diff = f"({x_diff.name})"
                        funcstr = func.func_str(y1, y2, x_diff)
                        ret_str += "\tnu%d_func = %s\n" % (i+1, funcstr)
            kwargs_with_vars = DadiEngine._get_kwargs(event, var2value)
            str_kwargs = ["%s=%s" % (k, v.name
                                     if isinstance(v, (Variable,
                                                       BinaryOperation))
                                     else v)
                          for k, v in kwargs_with_vars.items()]
            if event.n_pop == 1:
                func = 'dadi.Integration.one_pop'
            if event.n_pop == 2:
                func = 'dadi.Integration.two_pops'
            if event.n_pop == 3:
                func = 'dadi.Integration.three_pops'
            ret_str += "\tphi = %s(phi, xx, %s)\n" % (func,
                                                      ', '.join(str_kwargs))
        elif isinstance(event, Split):
            if event.n_pop == 1:
                ret_str += "\tphi = dadi.PhiManip.phi_1D_to_2D"
            else:
                ret_str += "\tphi = dadi.PhiManip.phi_%dD_to_%dD_split_%d" %\
                           (event.n_pop, event.n_pop+1, event.pop_to_div + 1)
            ret_str += "(xx, phi)\n"
    ret_str += "\tsfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))\n"
    ret_str += "\treturn sfs\n"
    return ret_str


def _is_fs_via_dadi(data_holder):
    """
    Check that data is allele frequency spectrum saved for dadi.

    :param data_holder: Data holder with data.
    :type data_holder: :class:`gadma.data.data_holder.DataHolder`
    """
    import dadi
    try:
        data = dadi.Spectrum.from_file(data_holder.filename)
        return data
    except Exception as e:
        return None


def _print_load_data(data_holder, is_fs, mode='dadi'):
    """
    Returns strings for code that loads data for analysis.

    :param data_holder: Data holder with data.
    :param is_fs: If True then data isin AFS format of dadi.
    :param mode: Which mode to use, could be 'dadi' or 'moments'.
    """
    ret_str = ""
    fname = data_holder.filename
    if is_fs is None:
        ret_str += f"dd = {mode}.Misc.make_data_dict('{fname}')\n"
        ret_str += f"data = {mode}.Spectrum.from_data_dict"
        if (data_holder.population_labels is None or
                data_holder.projections is None or
                data_holder.outgroup is None):
            raise ValueError("Please specify data_holder carefully. "
                             "Some attribute is None.")
        lbls = list(data_holder.population_labels)
        size = list(data_holder.projections)
        outg = data_holder.outgroup
        ret_str += f"(dd, {lbls}, {size}, polarized={outg})\n"
    else:
        data = is_fs
        ret_str += f"data = {mode}.Spectrum.from_file('{fname}')\n"
        if (data_holder.projections is not None and
                list(data.sample_sizes) != list(data_holder.projections)):
            size = data_holder.projections
            ret_str += "data = data.project(%s)\n" % (str(list(size)))
        if data.folded == data_holder.outgroup:
            ret_str += "data = data.fold()\n"
        if data.pop_ids is None:
            if data_holder.population_labels is None:
                data.pop_ids = [f"Pop{i}" for i in range(data.ndim)]
            else:
                data.pop_ids = data_holder.population_labels
            ret_str += f"data.pop_ids = {data.pop_ids}\n"
        if (data_holder.population_labels is not None and
                list(data.pop_ids) != list(data_holder.population_labels)):
            d = {x: i for i, x in enumerate(data.pop_ids)}
            d = [d[x] for x in data_holder.population_labels]
            ret_str += "data = np.transpose(data, %s)\n" % (str(d))
            ret_str += f"data.pop_ids = {data_holder.population_labels}\n"
    return ret_str


def _print_dadi_load_data(data_holder):
    """
    Check :func:`_print_load_data` for more information.
    Checks that `data_holder` is in AFS format of dadi and runs
    :func:`_print_load_data` with 'dadi' mode.
    """
    is_fs = _is_fs_via_dadi(data_holder)
    return _print_load_data(data_holder, is_fs, mode='dadi')


def _print_p0(values):
    """
    Returns code for p0 with values.

    :param values: Values for p0.
    """
    return "p0 = [%s]\n" % ", ".join([str(x) for x in values])


def _print_dadi_simulation():
    """
    Returns string of code about simulation with dadi.
    """
    ret_str = f"func_ex = dadi.Numerics.make_extrap_log_func"\
              f"({FUNCTION_NAME})\n"
    ret_str += "model = func_ex(p0, ns, pts)\n"
    return ret_str


def _print_main(engine, values, mode='dadi', nanc=None):
    """
    Returns string for main part of generated code.

    :param engine: Engine that was used with data and model.
    :param values: best values of model parameters.
    :param mode: 'dadi' or 'moments'
    :param nanc: Size of ancestral population. It could be known if inference
                 was made by other engine and nanc from this engine will be
                 different to optimal.
    """
    ret_str = f"ll_model = {mode}.Inference.ll_multinom(model, data)\n"
    ret_str += "print('Model log likelihood (LL(model, data)): "\
               "{0}'.format(ll_model))\n"
    ret_str += f"\ntheta = {mode}.Inference.optimal_sfs_scaling"\
               "(model, data)\n"

    ret_str += "print('Optimal value of theta: {0}'.format(theta))\n"

    if nanc is not None:
        ret_str += f"Nanc = {nanc}  # {mode} was not used for inference\n"
        return ret_str

    mu_is_val = engine.model.mu is not None
    data_holder_is_val = engine.data_holder is not None
    seq_len_is_val = engine.data_holder.sequence_length is not None

    mu_and_L = mu_is_val and data_holder_is_val and not seq_len_is_val

    if engine.model.theta0 is not None or mu_and_L:
        if engine.model.theta0 is not None:
            ret_str += f"theta0 = {engine.model.theta0}\n"
            ret_str += "Nanc = int(theta / theta0)\n"
        elif mu_and_L:
            ret_str += f"mu = {engine.model.mu}\n"
            ret_str += f"L = {engine.data_holder.sequence_length}\n"
            ret_str += "Nanc = int(theta / (4 * mu * L))\n"
        ret_str += "print('Size of ancestral population: {0}'.format(Nanc))\n"
    else:
        ret_str += "Nanc = None\n"
    return ret_str


def _print_dadi_main(engine, values, nanc=None):
    ret_str = _print_p0(values)
    ret_str += _print_dadi_simulation()
    ret_str += _print_main(engine, values, mode='dadi', nanc=nanc)
    return ret_str


def print_dadi_code(engine, values, pts, filename,
                    nanc=None, gen_time=None, gen_time_units=None):
    """
    Generates code for `dadi` to file. Code have function of demographic model
    that simulates AFS and main part where simulation takes place as well as
    calculation of log-likelihood.

    :param engine: Engine that was used with data and model.
    :param values: Value of model parameters.
    :param pts: Grid sizes for dadi.
    :param filename: File to save generated code.
    :param nanc: Size of ancestral population. Is used when other engine was
                 used for inference.
    :param gen_time: Time of one generation in units of ``gen_time_units``.
    :param gen_time_units: Units of time. String.

    :note: the last two arguments are ignored as dadi could not draw models.
    """
    ret_str = "import dadi\nimport numpy as np\n\n"
    ret_str += _print_dadi_func(engine.model, values)
    ret_str += "\n"
    ret_str += _print_dadi_load_data(engine.data_holder)
    ret_str += "pts = %s\nns = data.sample_sizes\n\n" % str(pts)

    var2value = engine.model.var2value(values)
    values = [var2value[var] for var in engine.model.variables
              if not isinstance(var, DiscreteVariable)]
    ret_str += _print_dadi_main(engine, values, nanc)
    if filename is None:
        return ret_str
    with open(filename, 'w') as f:
        f.write(ret_str)
