from ..models import CustomDemographicModel, Epoch, Split, BinaryOperation
from ..utils import DiscreteVariable, DynamicVariable, Variable
from .dadi_generator import FUNCTION_NAME, _print_p0, _print_main,\
                            _print_load_data
import numpy as np
import copy
import sys


def _print_moments_func(model, values, dt_fac):
    """
    values are needed to fix dynamics of populations.
    """
    import moments
    from ..engines import MomentsEngine  # to avoid cross import

    if isinstance(model, CustomDemographicModel):
        ret_str = "import importlib.util\n\n"
        ret_str += "spec = importlib.util.spec_from_file_location('module',"\
                   f" '{sys.modules[model.function.__module__].__file__}')\n"
        ret_str += "module = importlib.util.module_from_spec(spec)\n"
        ret_str += "spec.loader.exec_module(module)\n"
        ret_str += f"{FUNCTION_NAME} = module.model_func\n\n"
        return ret_str

    var2value = model.var2value(values)  # OrderedDict

    f_vars = [x for x in var2value if not isinstance(x, DiscreteVariable)]
    ret_str = f"def {FUNCTION_NAME}(params, ns):\n"
    ret_str += "\t%s = params\n" % ", ".join([x.name for x in f_vars])
    ret_str += "\tsts = moments.LinearSystem_1D.steady_state_1D"\
               "(np.sum(ns))\n"\
               "\tfs = moments.Spectrum(sts)\n"

    ns_on_splits = [[f'ns[{i}]'
                     for i in range(model.number_of_populations())]]
    for ind, event in enumerate(reversed(model.events)):
        if isinstance(event, Split):
            p = event.pop_to_div
            ns_on_splits.append(copy.copy(ns_on_splits[-1][:-1]))
            ns_on_splits[-1][p] += " + " + ns_on_splits[-2][-1]
    ns_on_splits = list(reversed(ns_on_splits))[1:]
    n_split = 0

    for ind, event in enumerate(model.events):
        if event.__class__ is Epoch:
            all_sudden = True
            func_names = []
            if event.dyn_args is not None:
                for i in range(event.n_pop):
                    dyn_arg = event.dyn_args[i]
                    value = var2value.get(dyn_arg, dyn_arg)
                    if value != 'Sud':
                        all_sudden = False
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
                        func_names.append("nu%d_func" % (i+1))
            kwargs_with_vars = MomentsEngine._get_kwargs(event, var2value)
            kwargs = {}
            for x, y in kwargs_with_vars.items():
                if isinstance(y, np.ndarray):  # migration
                    l_s = "m = np.array(["
                    rows = []
                    for i in range(y.shape[0]):
                        elems = []
                        for j in range(y.shape[1]):
                            if i == j:
                                elems.append("0")
                                continue
                            yel = y[i][j]
                            if isinstance(yel, Variable):
                                elems.append(yel.name)
                            else:
                                elems.append(str(yel))
                        rows.append(f"[{', '.join(elems)}]")
                    ret_str += f"\tmigs = np.array([{', '.join(rows)}])\n"
                    kwargs[x] = 'migs'
                elif isinstance(y, list):  # pop. sizes, selection, dynamics
                    varnames = [var.name
                                if isinstance(var, (Variable,
                                                    BinaryOperation))
                                else str(var) for var in y]
                    if x == 'Npop':  # pop size
                        if all_sudden:
                            kwargs[x] = f"[{', '.join(varnames)}]"
                        else:
                            strs = []
                            for y1 in y:
                                if isinstance(y1, (Variable,
                                                   BinaryOperation)):
                                    strs.append(y1.name)
                                else:
                                    if y1 in func_names:
                                        strs.append(y1 + "(t)")
                                    else:
                                        strs.append(y1)
                            kwargs[x] = f"lambda t: [{', '.join(strs)}]"
                    else:
                        kwargs[x] = f"[{','.join(varnames)}]"
                else:  # time
                    kwargs[x] = y.name if isinstance(
                        y, (Variable, BinaryOperation)) else y

            kwargs = [f"{key}={value}" for key, value in kwargs.items()]
            ret_str += f"\tfs.integrate({', '.join(kwargs)}, "\
                       f"dt_fac={dt_fac})\n"
        elif event.__class__ is Split:
            ns_split = (ns_on_splits[n_split][event.pop_to_div],
                        ns_on_splits[n_split][-1])
            if event.n_pop == 1:
                ret_str += "\tfs = moments.Manips.split_1D_to_2D"
            else:
                ret_str += "\tfs = moments.Manips.split_%dD_to_%dD_%d" % (
                           event.n_pop, event.n_pop + 1, event.pop_to_div + 1)

            ret_str += f"(fs, {', '.join(ns_split)})\n"
            n_split += 1
    ret_str += "\treturn fs\n"
    return ret_str


def _is_fs_via_moments(data_holder):
    import moments
    try:
        data = moments.Spectrum.from_file(data_holder.filename)
        return data
    except Exception as e:
        return None


def _print_moments_load_data(data_holder):
    is_fs = _is_fs_via_moments(data_holder)
    return _print_load_data(data_holder, is_fs, mode='moments')


def _print_moments_simulation():
    ret_str = f"model = {FUNCTION_NAME}(p0, ns)\n"
    return ret_str


def _print_moments_main(engine, values):
    ret_str = _print_p0(values)
    ret_str += _print_moments_simulation()
    ret_str += _print_main(engine, values, mode='moments')
    return ret_str


def print_moments_code(engine, values, dt_fac, filename):
    ret_str = "import moments\nimport numpy as np\n\n"
    ret_str += _print_moments_func(engine.model, values, dt_fac)
    ret_str += "\n"
    ret_str += _print_moments_load_data(engine.data_holder)
    ret_str += "ns = data.sample_sizes\n\n"

    values = [val for var, val in engine.model.var2value(values).items()
              if not isinstance(var, DiscreteVariable)]
    ret_str += _print_moments_main(engine, values)
    if filename is None:
        return ret_str
    with open(filename, 'w') as f:
        f.write(ret_str)
