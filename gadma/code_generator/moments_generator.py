from ..models import CustomDemographicModel, EpochDemographicModel, \
    Epoch, Split
from ..utils import DiscreteVariable
from .dadi_generator import _print_p0, _print_main, _print_load_data, \
    _print_bounds
from .generator import FUNCTION_NAME, _value_to_code, _read_fs_or_none, \
    _get_variables_of_function, _print_unpacking_of_params, \
    _print_custom_model_import, _print_size_functions, \
    _print_integration_kwargs
import copy


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
        return _print_custom_model_import(model)

    assert isinstance(model, EpochDemographicModel)

    var2value = model.var2value(values)  # OrderedDict

    ret_str = f"def {FUNCTION_NAME}(params, ns):\n"
    ret_str += _print_unpacking_of_params(_get_variables_of_function(model))
    # Not very good solution !
    if not model.has_anc_size:
        ret_str += "\t_Nanc_size"
    else:
        ret_str += "\tNanc"
    ret_str += " = 1.0  # This value can be used in splits with fractions\n"

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
            funcs_str, func_names = _print_size_functions(event, var2value,
                                                          indent="\t")
            ret_str += funcs_str
            kwargs_with_vars = MomentsEngine._get_kwargs(event, var2value)
            migs_str, kwargs = _print_integration_kwargs(
                kwargs_with_vars, size_key="Npop", func_names=func_names,
                indent="\t")
            ret_str += migs_str
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
    if model.has_p_misid:
        name = model.p_misid.name
        ret_str += f"\tfs = (1 - {name}) * fs + " \
                   f"{name} * moments.Numerics.reverse_array(fs)\n"
    ret_str += "\treturn fs\n"
    return ret_str


def _print_moments_load_data(data_holder):
    is_fs = _read_fs_or_none("moments", data_holder)
    return _print_load_data(data_holder, is_fs, mode='moments')


def _print_moments_simulation():
    ret_str = f"model = {FUNCTION_NAME}(p0, ns)\n"
    return ret_str


def _print_model_plotting(engine, nanc, gen_time, gen_time_units):
    """
    Returns code that draws picture of the model using moments. The code is
    commented out as other tools are used for drawing now. It is still
    generated, so one can uncomment it to get the picture from moments.
    """
    ret_str = "plot_ns = [4 for _ in ns]  # small sizes for fast drawing\n"
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
    ret_str += "moments.ModelPlot.plot_model(gen_mod,\n"
    ret_str += f"                             save_file='{filename}',\n"
    ret_str += f"                             fig_title='{fig_title}',\n"
    ret_str += f"                             draw_scale={draw_scale},\n"
    ret_str += "                             pop_labels=" \
               f"{_value_to_code(pop_ids)},\n"
    ret_str += f"                             nref={nanc},\n"
    ret_str += f"                             gen_time={gen_time},\n"
    ret_str += f"                             gen_time_units='{units}',\n"
    ret_str += "                             reverse_timeline=True)"

    # Comment the code out
    commented_lines = ["# " + line for line in ret_str.split("\n")]
    header = "\n\n# Uncomment the following code to draw the model " \
             "with moments:\n"
    return header + "\n".join(commented_lines)


def _print_moments_main(engine, values, nanc, gen_time, gen_time_units):
    ret_str = _print_p0(engine, values)
    ret_str += _print_bounds(engine)
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
