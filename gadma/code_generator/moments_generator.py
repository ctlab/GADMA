from ..models import CustomDemographicModel, EpochDemographicModel,\
    Epoch, Split, BinaryOperation
from ..utils import DiscreteVariable, DynamicVariable, Variable
from .dadi_generator import FUNCTION_NAME, _print_p0, _print_main,\
                            _print_load_data
import numpy as np
import copy
import sys


def _print_moments_func(model, values, dt_fac):
    """
    Returns string with function of demographic model for :py:mod:`moments`.
    Parameter `values` is needed to fix dynamics of populations.

    :param model: Demographic model.
    :type model: :class:`gadma.models.model.Model`
    :param values: List of values for parameters of model.
    """
    from ..engines import MomentsEngine  # to avoid cross import

    if isinstance(model, CustomDemographicModel):
        path_repr = repr(sys.modules[model.function.__module__].__file__)
        ret_str = "import importlib.util\n\n"
        ret_str += "spec = importlib.util.spec_from_file_location('module', "\
                   f"{path_repr})\n"
        ret_str += "module = importlib.util.module_from_spec(spec)\n"
        ret_str += "spec.loader.exec_module(module)\n"
        ret_str += f"{FUNCTION_NAME} = module.model_func\n\n"
        return ret_str

    assert isinstance(model, EpochDemographicModel)

    var2value = model.var2value(values)  # OrderedDict

    f_vars = [x for x in model.variables
              if not isinstance(x, DiscreteVariable)]
    if model.has_anc_size:
        if isinstance(model.Nanc_size, Variable):
            f_vars.pop(f_vars.index(model.Nanc_size))
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
                                        strs.append(str(y1))
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
    """
    Check that data is allele frequency spectrum saved for moments.

    :param data_holder: Data holder with data.
    :type data_holder: :class:`gadma.data.data_holder.DataHolder`
    """

    import moments
    try:
        data = moments.Spectrum.from_file(data_holder.filename)
        return data
    except Exception:
        return None


def _print_moments_load_data(data_holder):
    is_fs = _is_fs_via_moments(data_holder)
    return _print_load_data(data_holder, is_fs, mode='moments')


def _print_moments_simulation():
    ret_str = f"model = {FUNCTION_NAME}(p0, ns)\n"
    return ret_str


def _print_model_plotting(engine, nanc, gen_time, gen_time_units):
    ret_str = "\n\nplot_ns = [4 for _ in ns]  # small sizes for fast drawing\n"
    ret_str += f"gen_mod = moments.ModelPlot.generate_model({FUNCTION_NAME},\n"
    ret_str += "                                           p0, plot_ns)\n"

    filename = 'model_from_GADMA.png'
    if engine.data_holder.population_labels is None:
        pop_ids = [f"Pop{i}" for i in range(engine.data.ndim)]
    else:
        pop_ids = list(engine.data_holder.population_labels)
    draw_scale = nanc is not None
    units = gen_time_units
    if nanc is not None:
        nanc = int(nanc)
    if gen_time is None:
        gen_time = 1.0
    fig_title = "Demographic model from GADMA"
    ret_str += f"moments.ModelPlot.plot_model(gen_mod,\n"
    ret_str += f"                             save_file='{filename}',\n"
    ret_str += f"                             fig_title='{fig_title}',\n"
    ret_str += f"                             draw_scale={draw_scale},\n"
    ret_str += f"                             pop_labels={pop_ids},\n"
    ret_str += f"                             nref={nanc},\n"
    ret_str += f"                             gen_time={gen_time},\n"
    ret_str += f"                             gen_time_units='{units}',\n"
    ret_str += f"                             reverse_timeline=True)"
    return ret_str


def _print_moments_main(engine, values, nanc, gen_time, gen_time_units):
    ret_str = _print_p0(engine, values)
    ret_str += _print_moments_simulation()
    ret_str += _print_main(engine, values, mode='moments', nanc=nanc)
    ret_str += _print_model_plotting(engine, nanc, gen_time, gen_time_units)
    return ret_str


def print_moments_code(engine, values, dt_fac, filename,
                       nanc=None, gen_time=None, gen_time_units=None):
    """
    Generates code for `moments` to file. Code have function of demographic
    model that simulates AFS and main part where simulation takes place as well
    as calculation of log-likelihood.

    :param engine: Engine that was used with data and model.
    :param values: Value of model parameters.
    :param dt_fac: Grid step for moments.
    :param filename: File to save generated code.
    :param nanc: Size of ancestral population. Is used when other engine was
                 used for inference.
    :param gen_time: Time of one generation in units of ``gen_time_units``.
    :param gen_time_units: Units of time. String.

    """
    # check if multinom and we should save Nanc value
    if not engine.multinom:
        var2value = engine.model.var2value(values)
        Nanc_value = engine.get_value_from_var2value(var2value,
                                                     engine.model.Nanc_size)
        assert nanc is None or Nanc_value == nanc, f"{nanc}, {Nanc_value}"
        nanc = Nanc_value
    values = engine.model.translate_values(units="genetic", values=values)
    var2value = engine.model.var2value(values)
    ret_str = "import moments\nimport numpy as np\n\n"
    ret_str += _print_moments_func(engine.model, values, dt_fac)
    ret_str += "\n"
    ret_str += _print_moments_load_data(engine.data_holder)
    ret_str += "ns = data.sample_sizes\n\n"

    values = [var2value[var] for var in engine.model.variables
              if not isinstance(var, DiscreteVariable)]
    ret_str += _print_moments_main(engine, values, nanc,
                                   gen_time, gen_time_units)
    if filename is None:
        return ret_str
    with open(filename, 'w') as f:
        f.write(ret_str)
