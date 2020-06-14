from ..models import Epoch, Split
from ..utils import DiscreteVariable, DynamicVariable


FUNCTION_NAME = 'model_func'

def _print_dadi_func(model, values):
    """
    values are needed to fix dynamics of populations.
    """
    from ..engines import DadiEngine  # to avoid cross import
    var2value = model.var2value(values)  # OrderedDict

    f_vars = [x for x in var2value if not isinstance(x, DiscreteVariable)]
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
                        func = DynamicVariable.get_func_from_value(value)
                        funcstr = func.func_str(event.init_size_args[i], 
                                                 event.size_args[i], 
                                                 event.time_arg)
                        ret_str += "\tnu%d_func = %s\n" % (i+1, funcstr)
            kwargs_with_vars = DadiEngine._get_kwargs(event, var2value)
            str_kwargs = ["%s=%s" % (k, v.name if isinstance(v, Variable)
                                     else v) for k, v in kwargs_with_vars.items()]
            if event.n_pop == 1:
                func = 'dadi.Integration.one_pop'
            if event.n_pop == 2:
                func = 'dadi.Integration.two_pops'
            if event.n_pop == 3:
                func = 'dadi.Integration.three_pops'
            ret_str += "\tphi = %s(phi, xx, %s)\n" % (func, ', '.join(str_kwargs))
        elif isinstance(event, Split):
            if event.n_pop == 1:
                ret_str += "phi = dadi.PhiManip.phi_1D_to_2D"
            else:
                ret_str += "phi = phi_%dD_to_%dD_%d" % (event.n_pop-1, event.n_pop, event.pop_to_div + 1)
            ret_str += "(xx, phi)\n" 
    ret_str += "\tsfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))\n"
    ret_str += "\treturn sfs\n"
    return ret_str


def _is_fs_via_dadi(data_holder):
    import dadi
    try:
        data = dadi.Spectrum.from_file(data_holder.filename)
        return data
    except Exception as e:
        return None


def _print_load_data(data_holder, is_fs, mode='dadi'):
    ret_str = ""
    fname = data_holder.filename
    if is_fs is None:
        ret_str += f"dd = {mode}.Misc.make_data_dict('{fname}')\n"
        ret_str += f"data = {mode}.Spectrum.from_data_dict"
        lbls = str(data_holder.pop_labels)
        size = str(data_holder.sample_sizes)
        outg = str(data_holder.outgroup)
        ret_str += "(dd, %s, %s, %s)\n" % (lbls, size, outg)
    else:
        data = is_fs
        ret_str += f"data = {mode}.Spectrum.from_file('{fname}')\n"
        if data.sample_sizes != data_holder.projections:
            size = data_holder.sample_sizes
            ret_str += "data = data.project(%s)\n" % (str(list(size)))
        if data.folded == data_holder.outgroup:
            ret_str += "data = data.fold()\n"
        if data.pop_ids is None:
            data.pop_ids = data_holder.population_labels
            ret_str += f"data.pop_ids = {data_holder.population_labels}\n"
        if data.pop_ids != data_holder.population_labels:
            d = {x: i for i, x in enumerate(data.pop_ids)}
            ret_str += "data = np.transpose(data, %s)\n" % (str(d))
    return ret_str


def _print_dadi_load_data(data_holder):
    is_fs = _is_fs_via_dadi(data_holder)
    return _print_load_data(data_holder, is_fs, mode='dadi')


def _print_p0(values):
    return "p0 = [%s]\n" % ", ".join([str(x) for x in values])

def _print_dadi_simulation():
    ret_str = "func_ex = dadi.Numerics.make_extrap_log_func({FUNCTION_NAME})\n"
    ret_str += "model = func_ex(p0, ns, pts)\n"
    return ret_str

def _print_main(engine, values, mode='dadi'):
    ret_str = f"ll_model = {mode}.Inference.ll_multinom(model, data)\n"
    ret_str += "print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))\n"
    ret_str += f"\ntheta = {mode}.Inference.optimal_sfs_scaling(model, data)\n"

    ret_str += "print('Optimal value of theta: {0}'.format(theta))\n"

    mu_and_L = engine.model.mu is not None and\
               engine.data_holder is not None and\
               engine.data_holder.sequence_length is not None
    if engine.model.theta0 is not None or mu_and_L:
        if engine.model.theta0 is not None:
            ret_str += f"theta0 = {engine.model.theta0}\n"
            ret_str += "Nanc = int(theta / theta0)\n"
        elif mu_and_L:
            ret_str += f"mu = {engine.model.mu}\n"
            ret_str += f"L = {engine.data_holder.sequence_length}\n"
            ret_str += "Nanc = int(theta / (4 * mu * L))\n"
        ret_str += "print('Size of ancestral population: {0}'.format(Nanc))\n"
    return ret_str

def _print_dadi_main(engine, values):
    ret_str = _print_p0(values)
    ret_str += _print_dadi_simulation()
    ret_str += _print_main(engine, values, mode='dadi')
    return ret_str

def print_dadi_code(engine, values, pts, filename):
    ret_str = "import dadi\nimport numpy as np\n\n"
    ret_str += _print_dadi_func(engine.model, values)
    ret_str += "\n"
    ret_str += _print_dadi_load_data(engine.data_holder)
    ret_str += "pts = %s\nns = data.sample_sizes\n\n" % str(pts)

    values = [val for var, val in engine.model.var2value(values).items()
              if not isinstance(var, DiscreteVariable)]
    ret_str += _print_dadi_main(engine, values)
    if filename is None:
        return ret_str
    with open(filename, 'w') as f:
        f.write(ret_str)
