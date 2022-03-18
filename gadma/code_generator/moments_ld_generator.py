import os.path

from ..models import CustomDemographicModel, EpochDemographicModel, \
    Epoch, Split, BinaryOperation, StructureDemographicModel
from ..utils import DiscreteVariable, DynamicVariable, Variable
from ..data import extract_chromosomes_from_vcf
import numpy as np
import copy
import sys
from os import listdir

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
        path_repr = repr(sys.modules[model.function.__module__].__file__)
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
    ret_str += "    %s = params\n" % ", ".join([x.name for x in f_vars])
    # Not very good solution !
    if not model.has_anc_size:
        ret_str += "    _Nanc_size"
    else:
        ret_str += "    Nanc"

    ret_str += " = 1.0  # This value can be used in splits with fractions\n"
    ret_str += "    Y = moments.LD.Numerics.steady_state" \
               "(rho=rho, theta=theta)\n" \
               f"    Y = moments.LD.LDstats(Y, num_pops=1, pop_ids={pops})\n"
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


def _print_momentsLD_load_data(engine, data_holder):
    """
    Returns strings for code that loads data for analysis.

    :param data_holder: Data holder with data.
    """
    from ..engines import extract_rec_map_name_and_extension
    ret_str = ""

    path = f"{data_holder.bed_files_dir}".replace("\\", "\\\\")
    ret_str += f"bed_files = \"{path}\"\n"
    ret_str += "reg_num = 0\n" \
               "region_stats = {}\n"
    if data_holder.filename is not None:
        chromosomes = extract_chromosomes_from_vcf(data_holder.filename)
        ret_str += f"chromosomes = {chromosomes}\n"
    else:
        ret_str += "chromosomes = None\n"
    kwargs = engine.get_kwargs()
    rec_maps = data_holder.recombination_maps
    pops = data_holder.population_labels
    if rec_maps is not None:
        if len(listdir(rec_maps)) == len(chromosomes):
            rec_map, extension = extract_rec_map_name_and_extension(
                listdir(rec_maps)[0]
            )
            ret_str += f"extension = \"{extension}\"\n"
            ret_str += f"rec_map_name = \"{rec_map}\"\n"
        else:
            rec_map = listdir(rec_maps)[0]
            ret_str += f"rec_map = \"{rec_map}\"\n"
    ret_str += "kwargs = {\n"
    ret_str += f"    \"r_bins\": {[ii for ii in kwargs['r_bins']]},\n"
    ret_str += f"    \"report\": {kwargs['report']},\n"
    ret_str += f"    \"bp_bins\": {[ii for ii in kwargs['bp_bins']]},\n"
    ret_str += f"    \"use_genotypes\": {kwargs['use_genotypes']},\n"
    ret_str += f"    \"cM\": {kwargs['cM']},\n"
    ret_str += "}\n"

    vcf_path = data_holder.filename.replace("\\", "\\\\")
    popmap_path = data_holder.popmap_file.replace("\\", "\\\\")
    ret_str += f"vcf_file = \"{vcf_path}\"\n"
    ret_str += f"pop_map = \"{popmap_path}\"\n"
    if rec_maps is not None:
        rec_maps_path = rec_maps.replace('\\', '\\\\')
        ret_str += f"rec_maps = \"{rec_maps_path}\"\n"

    prep_path = data_holder.preprocessed_data.replace("\\", "\\\\")
    ret_str += f"preprocessed_data = \"{prep_path}\"\n"

    ret_str += "if preprocessed_data is not None:\n"
    ret_str += "    with open(preprocessed_data, \"rb\") as fin:\n"
    ret_str += "        region_stats = pickle.load(fin)\n"
    ret_str += "else:\n"
    ret_str += f"    for bed_file in sorted(os.listdir(bed_files)):\n"
    ret_str += "        chrom = bed_file.split('_')[-2]\n"
    ret_str += "        region_stats.update({\n"
    ret_str += "            f\"{reg_num}\":\n" \
               "                moments.LD." \
               "Parsing.compute_ld_statistics(\n"
    ret_str += "                    vcf_file=vcf_file,\n"
    if rec_maps is not None:
        ret_str += f"                    rec_map_file=" \
                   f"f\"{rec_maps}/\"\n"
        if len(listdir(rec_maps)) == len(chromosomes):
            ret_str += "                    " \
                       "f\"{rec_map_name}_{chrom}.{extension}\",\n"
        else:
            ret_str += "                    " \
                       "f\"{rec_map}\",\n"
            ret_str += "                    " \
                       "map_name=f\"{chrom}\",\n"
            ret_str += "                    " \
                       "chromosome=f\"{chrom}\",\n"
    ret_str += f"                    pop_file=pop_map,\n"
    bed_path = "os.path.join(bed_files, bed_file)"
    ret_str += f"                    bed_file=bed_path,\n"
    ret_str += f"                    pops={pops},\n"
    ret_str += "                    **kwargs\n"
    ret_str += "                )\n"
    ret_str += "        })\n"
    ret_str += "        reg_num += 1\n"
    ret_str += "data = moments.LD.Parsing.bootstrap_data(region_stats)\n\n"
    return ret_str


def _print_momentsLD_simulation(engine, nanc):
    ret_str = ""
    r_bins = engine.get_kwargs()['r_bins']
    if nanc is not None:
        ret_str += f"Nanc = {nanc}\n"
    ret_str += f"r_bins = {[ii for ii in r_bins]}\n"
    ret_str += f"mu = {engine.model.mutation_rate}\n"
    ret_str += "rhos = 4 * Nanc * np.array(r_bins)\n"
    ret_str += "theta = 4 * Nanc * mu\n\n"
    ret_str += f"model = {FUNCTION_NAME}(p0, rho=rhos, theta=theta)\n"
    ret_str += "model = moments.LD.LDstats(\n"
    ret_str += "    [(y_l + y_r) / 2 for y_l, y_r in zip(\n"
    ret_str += "        model[:-2], model[1:-1])]\n"
    ret_str += "    + [model[-1]],\n"
    ret_str += "    num_pops=model.num_pops,\n"
    ret_str += "    pop_ids=model.pop_ids,\n)\n"
    ret_str += "model = moments.LD.Inference.sigmaD2(model)\n"
    ret_str += "model_for_plot = copy.deepcopy(model)\n"
    return ret_str


def _print_LdCurves(engine):
    r_bins = engine.get_kwargs()["r_bins"]
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
    ret_str += f"ll_model = " \
               f"moments.LD.Inference.ll_over_bins(means, model, varcovs)\n\n"
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
    ret_str += f"rep_data_file = \"{engine.data_holder.preprocessed_data}\"\n"
    ret_str += extract_opt_params(engine, values, nanc)
    if engine.data_holder.recombination_maps:
        ret_str += f"rs = {[ii for ii in engine.get_kwargs()['r_bins']]}\n"
    else:
        ret_str += "\n# You need recombination map if you wont to compute CI\n"

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


def remove_fraction_variable_for_two_sudden_children(engine, values):

    # Just in case to avoid errors we will copy engine
    engine = copy.deepcopy(engine)
    var2value = engine.model.var2value(values)

    dyn_vars_list = []
    split_num = 1
    for event in engine.model.events:
        if isinstance(event, Split):
            dyn_vars_list.append([
                f"dyn{split_num}{event.pop_to_div + 1}",
                f"dyn{split_num}{event.n_pop + 1}"
            ])
            split_num += 1

    fraction_vars_to_fix_list = []

    for split_dyn_vars_sublist in dyn_vars_list:
        split_num = 1
        first_dyn_sudden = False
        second_dyn_sudden = False
        for variable in var2value:
            if all([
                variable.name == split_dyn_vars_sublist[0],
                var2value[variable] == "Sud"
            ]):
                first_dyn_sudden = True
            elif all([
                variable.name == split_dyn_vars_sublist[1],
                var2value[variable] == "Sud"
            ]):
                second_dyn_sudden = True
        if first_dyn_sudden and second_dyn_sudden:
            fraction_vars_to_fix_list.append(f"s{split_num}")

    for ii, variable in enumerate(engine.model._variables):
        if variable.name in fraction_vars_to_fix_list:
            engine.model.fix_variable(engine.model._variables[ii], 0.5)

    values = [var2value[var] for var in engine.model.variables]

    return engine, values, fraction_vars_to_fix_list
