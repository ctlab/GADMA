from ..models import CustomDemographicModel, EpochDemographicModel, \
    Epoch, Split, BinaryOperation
from ..utils import DiscreteVariable, DynamicVariable, Variable
# from ..engines import MomentsLdEngine
from ..utils import create_bed_files_and_extract_chromosomes
from ..data import check_and_return_projections_and_labels
import numpy as np
import copy
import sys
from os import listdir

FUNCTION_NAME = "model_func"


def _print_p0(values):
    """
    Returns code for p0 with values.

    :param values: Values for p0.
    """
    p0 = copy.copy(values)
    return "p0 = [%s]\n" % ", ".join([str(x) for x in p0])


def _print_momentsLD_func(model, values):
    """
    Returns string with function of demographic model for :py:mod:`momentsLD`.
    Parameter `values` is needed to fix dynamics of populations.

    :param model: Demographic model.
    :type model: :class:`gadma.models.model.Model`
    :param values: List of values for parameters of model.
    """
    from ..engines import MomentsLdEngine  # to avoid cross import

    if isinstance(model, CustomDemographicModel):
        path_repr = repr(sys.modules[model.function.__module__].__file__)
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
        if isinstance(model.Nanc_size, Variable):
            f_vars.pop(f_vars.index(model.Nanc_size))
    else:
        raise ValueError("Nope")

    ret_str = f"def {FUNCTION_NAME}(params):\n"
    ret_str += "\t%s = params\n" % ", ".join([x.name for x in f_vars])
    ret_str += "\tY = moments.LD.Numerics.steady_state(rho=rho, theta=theta)\n" \
               "\tY = moments.LD.LDstats(Y, num_pops=1, pop_ids=pop_ids))\n"
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
                        ret_str += "\tnu%d_func = %s\n" % (i + 1, funcstr)
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
            ret_str += f"\tY.integrate({', '.join(kwargs)}\n"
        elif event.__class__ is Split:
            ret_str += f"\tY = Y.split({n_split})\n"
            n_split += 1
    ret_str += "\treturn Y\n"
    return ret_str


def _print_momentsLD_load_data(engine, data_holder):
    """
    Returns strings for code that loads data for analysis.

    :param data_holder: Data holder with data.
    """
    from ..engines import MomentsLdEngine
    ret_str = ""

    ret_str += f"bed_files = {data_holder.output_directory}" \
               + "/bed_files/\n"
    ret_str += "reg_num = 0\n" \
               "region_stats = {}\n"
    chromosomes = create_bed_files_and_extract_chromosomes(data_holder)
    ret_str += f"chromosomes = {chromosomes}\n"
    kwargs = engine.kwargs

    pops = data_holder.population_labels

    rec_map = listdir(data_holder.recombination_maps)[0]
    extension = rec_map.split(".")[1]
    rec_map = rec_map.split(".")[0]
    rec_map = "_".join(rec_map.split('_')[:-1])
    ret_str += f"extension = {extension}\n"
    ret_str += f"rec_map = {rec_map}\n"
    ret_str += "kwargs = {\n"
    ret_str += f"\tr_bins: {kwargs['r_bins']},\n"
    ret_str += f"\treport: {kwargs['report']},\n"
    ret_str += f"\tbp_bins: {kwargs['bp_bins']},\n"
    ret_str += f"\tuse_genotypes: {kwargs['use_genotypes']},\n"
    ret_str += f"\tcM: {kwargs['cM']},\n"
    ret_str += "}\n"
    ret_str += f"vcf_file = {data_holder.filename}\n"
    ret_str += f"pop_map = {data_holder.popmap_file}\n"
    ret_str += f"rec_maps = {data_holder.recombination_maps}\n"

    if data_holder.recombination_maps is not None:
        ret_str += "for chrom in chromosomes:\n"
        ret_str += "    for num in range(1, chromosomes[chrom]):\n"
        ret_str += "        region_stats.update({\n"
        ret_str += "            f\"{reg_num}\":\n" \
                   "            moments.LD.Parsing.compute_ld_statistics(\n"
        ret_str += "                vcf_file,\n"
        ret_str += f"                rec_map_file=f\"rec_maps"
        ret_str += "                         /{rec_map}_{chrom}.{extension}\"\n"
        ret_str += "                pop_file=pop_map\n"
        ret_str += "                bed_file=f\"{bed_files}/"
        ret_str += "bed_file_{chrom}_{num}.bed\"\n"
        ret_str += f"                pops={pops}\n"
        ret_str += "                **kwargs\n"
        ret_str += "            )\n"
        ret_str += "        })\n"
        ret_str += "        reg_num += 1\n"
    else:
        ret_str += "for chrom in chromosomes:\n"
        ret_str += "    for num in range(1, chromosomes[chrom]):\n"
        ret_str += "        region_stats.update({\n"
        ret_str += "            f\"{reg_num}\":\n" \
                   "            moments.LD.Parsing.compute_ld_statistics(\n"
        ret_str += "                vcf_file,\n"
        ret_str += f"                pop_file=pop_map\n"
        ret_str += "                bed_file=f\"{bed_files}/"
        ret_str += "bed_file_{chrom}_{num}.bed\"\n"
        ret_str += f"                pops={pops}\n"
        ret_str += "                **kwargs\n"
        ret_str += "            )\n"
        ret_str += "        })\n"
        ret_str += "        reg_num += 1\n"

    ret_str += "data = moments.LD.Parsing.bootstrap_data(region_stats)\n\n"
    return ret_str


def _print_momentsLD_simulation():
    ret_str = f"model = {FUNCTION_NAME}(p0)\n"
    return ret_str


def _print_LdCurves(engine):
    r_bins = engine.kwargs["r_bins"]
    ret_str = "stats_to_plot = [\n"
    ret_str += "    [name] for name in model.names()[:-1][0] if name != 'pi2_0_0_0_0'\n"
    ret_str += "]\n"
    ret_str += "moments.LD..Plotting.plot_ld_curves_comp(\n"
    ret_str += "    model,\n"
    ret_str += "    data['means'][:-1],\n"
    ret_str += "    data['varcovs'][:-1],\n"
    ret_str += f"    rs={r_bins},\n"
    ret_str += "    stats_to_plot=stats_to_plot,\n"
    ret_str += "    fig_size=(len(stats_to_plot), 9),\n"
    ret_str += "    cols=round(len(stats_to_plot) / 3),\n"
    ret_str += "    plot_means=True,\n"
    ret_str += "    plot_vcs=True,\n"
    ret_str += ")\n\n"
    return ret_str


def _print_ll():
    ret_str = "model = moments.LD.Inference.remove_normalized_lds(model)\n"
    ret_str += "means, varcovs = moments.LD.Inference.remove_normalized_data(\n"
    ret_str += "    data['means'],\n"
    ret_str += "    data['varcovs'],\n"
    ret_str += "    num_pops=model.num_pops,\n"
    ret_str += "    normalization=0)\n"
    ret_str += f"ll_model = 'moments.LD.Inference.ll_over_bins(means, model, varcovs)\n\n"
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

    if nanc is not None:
        ret_str += f"Nanc = {nanc}\n"

    ret_str += "theta = 4 * Nanc * mu\n\n"

    ret_str += "print('Size of ancestral population: {Nanc}'.format(Nanc))\n"
    ret_str += _print_ll()

    return ret_str


def _print_moments_ld_main(engine, values, nanc):
    ret_str = _print_p0(values)
    ret_str += _print_momentsLD_simulation()
    ret_str += _print_main_ld(engine, nanc=nanc)
    ret_str += _print_LdCurves(engine)
    return ret_str


def print_moments_ld_code(engine, values, filename,
                          nanc=None, gen_time=None, gen_time_units=None):
    """
    Generates code for `moments` to file. Code have function of demographic
    model that simulates AFS and main part where simulation takes place as well
    as calculation of log-likelihood.

    :param engine: Engine that was used with data and model.
    :param values: Value of model parameters.
    :param filename: File to save generated code.
    :param nanc: Size of ancestral population. Is used when other engine was
                 used for inference.
    :param gen_time: Time of one generation in units of ``gen_time_units``.
    :param gen_time_units: Units of time. String.

    """
    var2value = engine.model.var2value(values)
    Nanc_value = engine.get_value_from_var2value(var2value,
                                                 engine.model.Nanc_size)
    assert nanc is None or Nanc_value == nanc, f"{nanc}, {Nanc_value}"
    nanc = Nanc_value
    values = engine.model.translate_values(units="genetic", values=values)
    var2value = engine.model.var2value(values)
    ret_str = "import moments.LD\nimport numpy as np\n\n"
    ret_str += _print_momentsLD_func(engine.model, values)
    ret_str += "\n"
    ret_str += _print_momentsLD_load_data(engine, engine.data_holder)

    values = [var2value[var] for var in engine.model.variables
              if not isinstance(var, DiscreteVariable)]
    ret_str += _print_moments_ld_main(engine, values, nanc)
    if filename is None:
        return ret_str
    with open(filename, 'w') as f:
        f.write(ret_str)
