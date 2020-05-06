from ..models import *
from ..engines import *

def _print_dadi_func(demographic_model, values):
    var2value = demographic_model.var2value(values)

    f_vars = [x for x in var2value if not isinstance(x.__class__, DiscreteVariable) ]
    ret_str = "def model_func(params, ns, pts):\n"
    ret_str += "\t%s = params" % ", ".join([x.name for x in f_vars])

    ret_str += "\txx = dadi.Numerics.default_grid(pts)\n"\
               "\tphi = dadi.PhiManip.phi_1D(xx)\n"
    for ind, event in enumerate(demographic_model.events):
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
            str_kwargs = ["%s=%s" % (k, str(v)) for k, v in kwargs_with_vars.items()]
            if event.n_pop == 1:
                func = dadi.Integration.one_pop
            if event.n_pop == 2:
                func = dadi.Integration.two_pops
            if event.n_pop == 3:
                func = dadi.Integration.three_pops
            ret_str += "\tphi = dadi.Integration.%s(phi, xx, %s)\n" % (func.__name__, ', '.join(str_kwargs))
        elif event.__class__ is Split:
            if event.n_pop == 1:
                ret_str += "phi = dadi.PhiManip.phi_1D_to_2D"
            else:
                ret_str += "phi = phi_%dD_to_%dD_%d" % (event.n_pop-1, event.n_pop, event.pop_to_div + 1)
            ret_str += "(xx, phi)\n" 
    ret_str += "\tsfs = dadi.Spectrum.from_phi(phi, ns, [xx]*len(ns))\n"
    ret_str += "\treturn sfs\n"
    return ret_str

def _print_dadi_load_data(data_holder):
    try:
        data = dadi.Spectrum.from_file(data_holder.filename)
    except:
        data = None
    ret_str = ""
    fname = data_holder.filename
    if data is None:
        ret_str += "dd = dadi.make_data_dict(%s)\n" % fname
        ret_str += "data = dadi.Spectrum.from_data_dict"
        lbls = str(data_holder.pop_labels)
        size = str(data_holder.sample_sizes)
        outg = str(data_holder.outgroup)
        ret_str += "(dd, %s, %s, %s)\n" % (lbls, size, outg)
    else:
        ret_str += "data = dadi.Spectrum.from_file(%s)\n"
        if data.sample_sizes != data_holder.sample_sizes:
            size = data_holder.sample_sizes
            ret_str += "data = data.project(%s)\n" % (str(list(size)))
        if data.folded == data_holder.outgroup:
            ret_str += "data = data.fold()\n"
        if data.pop_ids != data_holder.pop_labels:
            d = {x: i for i, x in enumerate(data.pop_ids)}
            ret_str += "data = np.transpose(data, %s)\n" % (str(d))

def _print_dadi_main(values):
    ret_str = "p0 = [%s]\n" % ", ".join(values)
    ret_str += "func_ex = dadi.Numerics.make_extrap_log_func(model_func)\n"
    ret_str += "model = func_ex(p0, ns, pts)\n"
    ret_str += "ll_model = dadi.Inference.ll_multinom(model, data)\n"
    ret_str += "print('Model log likelihood (LL(model, data)): {0}\n'.format(ll_model))"
    ret_str += "theta = %s.Inference.optimal_sfs_scaling(model, data)\n" % mode
    ret_str += "print('Optimal value of theta: {0}'.format(theta))\n"

def print_dadi_code(engine, data_holder, demographic_model, values, filename):
    ret_str = "import dadi\nimport numpy as np\n\n"
    ret_str += _print_dadi_func(demographic_model, values)
    ret_str += "\n"
    ret_str += _print_dadi_load_data(data_holder)
    ret_str += "pts = %s\nns = data.sample_sizes\n\n" % str(engine.pts)
    
    ret_str += _print_dadi_main(values)
    with open(filename, 'w') as f:
        f.write(ret_str)
