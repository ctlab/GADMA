from ..models import CustomDemographicModel, EpochDemographicModel, \
    Epoch, Split, BinaryOperation, StructureDemographicModel
from ..utils import DiscreteVariable, DynamicVariable, Variable
from .dadi_generator import _print_unpacking_of_params
import numpy as np
import copy
import os
import inspect

FUNCTION_NAME = "model_func"


def _print_p0(engine, values, nanc):
    """
    Returns code for p0 with values.

    :param values: Values for p0.
    """

    f_vars = [x for x in engine.model.variables
              if not isinstance(x, DiscreteVariable)]
    for variable in f_vars:
        if variable.name == "Nanc":
            Nanc_index = f_vars.index(variable)
            values.pop(Nanc_index)
    p0 = copy.copy(values)
    return "p0 = [%s]\n" % ", ".join([str(x) for x in p0])


def _print_momentsLD_func(engine, values):
    """
    Returns string with function of demographic model for :py:mod:`momentsLD`.
    Parameter `values` is needed to fix dynamics of populations.

    :param model: Demographic model.
    :type model: :class:`gadma.models.model.Model`
    :param values: List of values for parameters of model.
    """
    from ..engines import MomentsLdEngine  # to avoid cross import

    model = engine.model
    pops = engine.data_holder.population_labels
    if isinstance(model, CustomDemographicModel):
        # the module of the function could be not among loaded modules,
        # e.g. in the process of a run, so file is taken from the function
        path_repr = repr(os.path.abspath(inspect.getfile(model.function)))
        path_repr = path_repr.replace("\\", "\\\\")
        ret_str = "import importlib.util\n\n"
        ret_str += "spec = importlib.util.spec_from_file_location('module', " \
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
        if isinstance(model, CustomDemographicModel):
            for variable in f_vars:
                if variable.name == "Nanc":
                    f_vars.pop(f_vars.index(variable))
        elif isinstance(model, StructureDemographicModel):
            if isinstance(model.Nanc_size, Variable):
                f_vars.pop(f_vars.index(model.Nanc_size))

    ret_str = f"def {FUNCTION_NAME}(params, rho=None, theta=0.001):\n"
    ret_str += _print_unpacking_of_params(f_vars, indent="    ")
    # Not very good solution !
    if not model.has_anc_size:
        ret_str += "    _Nanc_size"
    else:
        ret_str += "    Nanc"

    ret_str += " = 1.0  # This value can be used in splits with fractions\n"
    import moments
    moments_version = moments.__version__.split(".")
    ret_str += "    Y = moments.LD.Numerics.steady_state"
    # starting from 1.1.16 there is additional argument in steady_state
    if (len(moments_version) != 3 or int(moments_version[0]) > 1 or
            int(moments_version[1]) > 1 or int(moments_version[2]) > 15):
        ret_str += "(nus=[1], rho=rho, theta=theta)\n"
    else:
        ret_str += "(rho=rho, theta=theta)\n"

    ret_str += f"    Y = moments.LD.LDstats(Y, num_pops=1, pop_ids={pops})\n"
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
                        ret_str += "    nu%d_func = %s\n" % (i + 1, funcstr)
                        func_names.append("nu%d_func" % (i + 1))
            kwargs_with_vars = MomentsLdEngine._get_kwargs(event, var2value)
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
                    ret_str += f"    migs = np.array([{', '.join(rows)}])\n"
                    kwargs[x] = 'migs'
                elif isinstance(y, list):  # pop. sizes, selection, dynamics
                    varnames = [var.name
                                if isinstance(var, (Variable,
                                                    BinaryOperation))
                                else str(var) for var in y]
                    if x == 'nu':  # pop size
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
            ret_str += f"    Y.integrate({', '.join(kwargs)}, " \
                       f"rho=rho, theta=theta)\n"
        elif event.__class__ is Split:
            ret_str += f"    Y = Y.split({n_split})\n"
            n_split += 1
    ret_str += "    return Y\n\n"
    return ret_str


def _value_to_code(value):
    """
    Returns code of the value: numpy arrays and numbers are written as lists
    and numbers of Python.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value_to_code(el) for el in value) + "]"
    if isinstance(value, np.generic):
        value = value.item()
    return repr(value)


def _print_momentsLD_load_data(engine, data_holder):
    """
    Returns strings for code that loads data for analysis.

    If data was not preprocessed then LD statistics are computed for the
    same regions and with the same arguments as engine computes them.

    :param data_holder: Data holder with data.
    """
    ret_str = ""
    ret_str += f"vcf_file = {repr(data_holder.filename)}\n"
    ret_str += f"pop_map = {repr(data_holder.popmap_file)}\n"
    ret_str += f"preprocessed_data = {repr(data_holder.preprocessed_data)}\n"

    if data_holder.preprocessed_data is not None:
        ret_str += "with open(preprocessed_data, \"rb\") as fin:\n"
        ret_str += "    region_stats, data = pickle.load(fin)\n\n"
        return ret_str

    # LD statistics are computed for each region (bed file)
    all_kwargs = engine.get_kwargs_of_regions(data_holder)
    own_keys = ["vcf_file", "pop_file", "pops", "bed_file", "chromosome",
                "rec_map_file", "map_name"]
    kwargs = {key: value for key, value in all_kwargs[0].items()
              if key not in own_keys}
    ret_str += f"pops = {_value_to_code(all_kwargs[0]['pops'])}\n"
    ret_str += "kwargs = {\n"
    for key, value in kwargs.items():
        ret_str += f"    {repr(key)}: {_value_to_code(value)},\n"
    ret_str += "}\n"
    ret_str += "# Regions: bed file, chromosome and its recombination map\n"
    ret_str += "regions = [\n"
    for region_kwargs in all_kwargs:
        region = (region_kwargs["bed_file"], region_kwargs["chromosome"],
                  region_kwargs.get("rec_map_file", None))
        ret_str += f"    {_value_to_code(region)},\n"
    ret_str += "]\n"
    ret_str += "region_stats = {}\n"
    ret_str += "for reg_num, (bed_file, chrom, rec_map_file) in " \
               "enumerate(regions):\n"
    ret_str += "    region_kwargs = dict(kwargs)\n"
    ret_str += "    if rec_map_file is not None:\n"
    ret_str += "        region_kwargs[\"rec_map_file\"] = rec_map_file\n"
    ret_str += "        region_kwargs[\"map_name\"] = chrom\n"
    ret_str += "    region_stats[str(reg_num)] = " \
               "moments.LD.Parsing.compute_ld_statistics(\n"
    ret_str += "        vcf_file,\n"
    ret_str += "        bed_file=bed_file,\n"
    ret_str += "        chromosome=chrom,\n"
    ret_str += "        pop_file=pop_map,\n"
    ret_str += "        pops=pops,\n"
    ret_str += "        **region_kwargs\n"
    ret_str += "    )\n"
    ret_str += "data = moments.LD.Parsing.bootstrap_data(region_stats)\n\n"
    return ret_str


def _print_momentsLD_simulation(engine, nanc):
    ret_str = ""
    r_bins = engine.get_r_bins()
    if nanc is not None:
        ret_str += f"Nanc = {nanc}\n"
    ret_str += f"r_bins = {[ii for ii in r_bins]}\n"
    ret_str += f"mu = {engine.model.mutation_rate}\n"
    ret_str += "rhos = 4 * Nanc * np.array(r_bins)\n"
    ret_str += "theta = 4 * Nanc * mu\n\n"
    ret_str += "model = moments.LD.Inference.bin_stats(\n"
    ret_str += f"\tmodel_func={FUNCTION_NAME},\n"
    ret_str += "\tparams=p0,\n"
    ret_str += "\trho=rhos,\n"
    ret_str += "\ttheta=theta,\n"
    ret_str += ")\n"
    ret_str += "model = moments.LD.Inference.sigmaD2(model)\n"
    ret_str += "model_for_plot = copy.deepcopy(model)\n"
    return ret_str


def _print_LdCurves(engine):
    r_bins = engine.get_r_bins()
    ret_str = "stats_to_plot = [\n"
    ret_str += "    [name] for name in model.names()[:-1][0] " \
               "if name != 'pi2_0_0_0_0'\n"
    ret_str += "]\n"
    ret_str += "moments.LD.Plotting.plot_ld_curves_comp(\n"
    ret_str += "    model_for_plot,\n"
    ret_str += "    data['means'][:-1],\n"
    ret_str += "    data['varcovs'][:-1],\n"
    ret_str += f"    rs=np.array({[ii for ii in r_bins]}),\n"
    ret_str += "    stats_to_plot=stats_to_plot,\n"
    ret_str += "    fig_size=(len(stats_to_plot), 7),\n"
    ret_str += "    cols=round(len(stats_to_plot) / 3),\n"
    ret_str += "    plot_means=True,\n"
    ret_str += "    plot_vcs=True,\n"
    ret_str += ")\n\n"
    return ret_str


def _print_ll():
    ret_str = "model = moments.LD.Inference.remove_normalized_lds(model)\n"
    ret_str += "means, varcovs = " \
               "moments.LD.Inference.remove_normalized_data(\n"
    ret_str += "    data['means'],\n"
    ret_str += "    data['varcovs'],\n"
    ret_str += "    num_pops=model.num_pops,\n"
    ret_str += "    normalization=0)\n"
    ret_str += "ll_model = " \
               "moments.LD.Inference.ll_over_bins(means, model, varcovs)\n\n"
    ret_str += "print('Model log likelihood (LL(model, data)): " \
               "{0}'.format(ll_model))\n\n"
    return ret_str


def _print_main_ld(engine, nanc=None):
    """
    Returns string for main part of generated code.

    :param engine: Engine that was used with data and model.
    :param nanc: Size of ancestral population. It could be known if inference
                 was made by other engine and nanc from this engine will be
                 different to optimal.
    """
    ret_str = ""

    ret_str += "print(f'Size of ancestral population: {Nanc}')\n"
    ret_str += _print_ll()

    return ret_str


def _print_moments_ld_main(engine, values, nanc):
    ret_str = _print_p0(engine, values, nanc)
    ret_str += _print_momentsLD_simulation(engine, nanc)
    ret_str += _print_main_ld(engine, nanc=nanc)
    ret_str += _print_LdCurves(engine)
    return ret_str


def print_moments_ld_code(engine, values, filename, args=None,
                          nanc=None, gen_time=None, gen_time_units=None):
    """
    Generates code for `momentsLD` to file. Code have function of demographic
    model that simulates model and main part where simulation takes place
    as well as calculation of log-likelihood.

    :param engine: Engine that was used with data and model.
    :param values: Value of model parameters.
    :param filename: File to save generated code.
    :param nanc: Size of ancestral population. Is used when other engine was
                 used for inference.
    :param gen_time: Time of one generation in units of ``gen_time_units``.
    :param gen_time_units: Units of time. String.

    """
    if isinstance(engine.model, StructureDemographicModel):
        engine, values, fraction_vars_to_fix_list = \
            remove_fraction_variable_for_two_sudden_children(
                engine, values
            )
    var2value = engine.model.var2value(values)

    if isinstance(engine.model, CustomDemographicModel):
        if engine.model.fixed_anc_size:
            Nanc_value = engine.model.fixed_anc_size
        else:
            for variable in engine.model.variables:
                if variable.name == "Nanc":
                    Nanc_value = var2value[variable]
    else:
        Nanc_value = engine.get_value_from_var2value(
            var2value,
            engine.model.Nanc_size
        )

    assert nanc is None or Nanc_value == nanc, f"{nanc}, {Nanc_value}"
    nanc = Nanc_value
    values = engine.model.translate_values(units="genetic", values=values)
    var2value = engine.model.var2value(values)
    ret_str = "import moments.LD\n" \
              "import numpy as np\nimport pickle\nimport copy\n\n\n"

    ret_str += _print_momentsLD_func(engine, values)
    ret_str += "\n"

    ret_str += _print_momentsLD_load_data(engine, engine.data_holder)

    # Here we assume that filename ends with .py
    if filename is not None:
        assert filename.split(".")[-1] == "py"
        ci_filename = ".".join(filename.split(".")[:-1]) + "data_for_CI.py"
        generate_file_for_ci_evaluation(engine, values, nanc, ci_filename)

    values = [var2value[var] for var in engine.model.variables
              if not isinstance(var, DiscreteVariable)]
    ret_str += _print_moments_ld_main(engine, values, nanc)
    if filename is None:
        return ret_str
    with open(filename, 'w') as f:
        f.write(ret_str)

    if isinstance(engine.model, StructureDemographicModel):
        for ii, variable in enumerate(engine.model._variables):
            if variable.name in fraction_vars_to_fix_list:
                engine.model.unfix_variable(engine.model._variables[ii])


def generate_file_for_ci_evaluation(engine, values, nanc, ci_filename):
    ret_str = "import moments.LD\nimport numpy as np\n\n"
    ret_str += _print_momentsLD_func(engine, values)
    data_file = engine.data_holder.preprocessed_data
    if data_file is None:
        ret_str += "# LD statistics were computed during the run and they "\
                   "were not saved.\n# Please save them with "\
                   "gadma-precompute_ld_data and set the file here.\n"
    ret_str += f"data_file = {repr(data_file)}\n"
    ret_str += extract_opt_params(engine, values, nanc)
    ret_str += f"rs = {_value_to_code(engine.get_r_bins())}\n"

    f_vars = [x for x in engine.model.variables
              if not isinstance(x, DiscreteVariable)]
    for variable in f_vars:
        if variable.name == "Nanc":
            f_vars.pop(f_vars.index(variable))
            f_vars.append(variable)
    ret_str += f"param_names = {[x.name for x in f_vars]}"

    with open(ci_filename, "w") as file:
        file.write(ret_str)


def extract_opt_params(engine, values, nanc):

    var2value = engine.model.var2value(values)
    values = [var2value[var] for var in engine.model.variables
              if not isinstance(var, DiscreteVariable)]
    p0 = copy.copy(values)
    p0 = p0[1:]
    p0.append(values[0])
    p0[-1] = int(p0[-1] * nanc)
    return "opt_params = [%s]\n" % ", ".join([str(x) for x in p0])


def _get_variables_of_arg(arg):
    """
    Returns variables that are used in argument of event: it could be
    variable, combination of variables, number, string or list of them.
    """
    if isinstance(arg, Variable):
        return [arg]
    if isinstance(arg, (list, tuple, np.ndarray)):
        variables = []
        for el in arg:
            variables.extend(_get_variables_of_arg(el))
        return variables
    return list(getattr(arg, "variables", []))


def remove_fraction_variable_for_two_sudden_children(engine, values):
    """
    If both populations after a split have sudden change of size then their
    sizes right after the split (fraction of the split or sizes) do not
    change the model of momentsLD. Such variables are fixed to their values:
    they should not be among parameters of the code, e.g. for confidence
    intervals, as the Fisher information matrix is singular with them.

    :returns: copy of engine with fixed variables, values of not fixed
              variables of its model and names of the fixed variables.
    """
    # Just in case to avoid errors we will copy engine
    engine = copy.deepcopy(engine)
    model = engine.model
    var2value = model.var2value(values)

    vars_to_fix = []
    for i, event in enumerate(model.events):
        if not isinstance(event, Split) or i + 1 >= len(model.events):
            continue
        next_event = model.events[i + 1]
        if not isinstance(next_event, Epoch):
            continue
        # populations after the split: the divided one and the new last one
        children = [event.pop_to_div, event.n_pop]
        if next_event.dyn_args is not None:
            dyns = [model.get_value_from_var2value(
                        var2value, next_event.dyn_args[child])
                    for child in children]
            if any(dyn != "Sud" for dyn in dyns):
                continue
        candidates = _get_variables_of_arg(
            [event.size_args[child] for child in children])
        # variables that are used in other places of the model
        used = []
        for j, other_event in enumerate(model.events):
            if j == i:
                continue
            if j == i + 1:
                for name in ["time_arg", "size_args", "mig_args",
                             "dyn_args", "sel_args", "dom_args"]:
                    used.extend(_get_variables_of_arg(
                        getattr(other_event, name, None)))
                continue
            used.extend(other_event.variables)
        # size of ancestral population is not in events but it is used
        used.append(getattr(model, "Nanc_size", None))
        for variable in candidates:
            if (variable in model.variables and variable not in used and
                    variable not in vars_to_fix):
                vars_to_fix.append(variable)

    fraction_vars_to_fix_list = [variable.name for variable in vars_to_fix]
    for ii, variable in enumerate(model._variables):
        if variable.name in fraction_vars_to_fix_list:
            model.fix_variable(model._variables[ii], var2value[variable])

    values = [var2value[var] for var in model.variables]

    return engine, values, fraction_vars_to_fix_list
