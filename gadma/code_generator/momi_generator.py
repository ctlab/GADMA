from ..models import CustomDemographicModel, EpochDemographicModel,\
    TreeDemographicModel
from ..utils import Variable, DiscreteVariable, DynamicVariable
from ..utils import TimeVariable, PopulationSizeVariable, FractionVariable
from ..models import Operation, BinaryOperation, UnaryOperation, Log, Exp
from ..data import SFSDataHolder, VCFDataHolder
from ..models import Leaf, PopulationSizeChange, LineageMovement
from . import dadi_generator
from ..data import read_popinfo
import sys
import os
import copy
import inspect
import numpy as np


def get_momi_name(entity):
    if isinstance(entity, Variable):
        return f"'{entity.name}'"
    if isinstance(entity, Operation):
        return f"lambda params: {get_momi_lambda(entity)}"
    return entity


def get_momi_lambda(entity):
    if isinstance(entity, Variable):
        return f"params.{entity.name}"
    if isinstance(entity, BinaryOperation):
        arg1 = get_momi_lambda(entity.arg1)
        if isinstance(entity.arg1, BinaryOperation):
            arg1 = f"({arg1})"
        arg2 = get_momi_lambda(entity.arg2)
        if isinstance(entity.arg2, BinaryOperation):
            arg2 = f"({arg2})"
        return f"{arg1} {entity.operation_str()} {arg2}"
    if isinstance(entity, UnaryOperation):
        arg = get_momi_lambda(entity.arg)
        return f"np.{entity.operation_str()}({arg})"
    return entity


def get_anc_model(original_model, values, nanc):
    if isinstance(original_model, CustomDemographicModel):
        return original_model, values
    assert original_model.has_anc_size or nanc is not None, (
        "Model should have anc. size or nanc should be set"
    )
    model = copy.deepcopy(original_model)
    var2value = original_model.var2value(values)

    # fix dynamics as we will not need them
    model.fix_dynamics(values)
    # we have to miss values of dynamics in `values`
    varname2value = {var.name: var2value[var]
                     for var in original_model.variables}
    model_values = [varname2value[var.name] for var in model.variables]

    # We set ancestral size just if it is missed
    if nanc is not None:
        if original_model.has_anc_size:
            model_Nanc = original_model.get_value_from_var2value(
                entity=original_model.Nanc_size,
                var2value=var2value,
            )
            assert np.isclose(nanc, model_Nanc), ("Nanc of model and given "
                                                  "Nanc are different")
        else:
            model.Nanc_size = nanc

    if isinstance(model, EpochDemographicModel):
        model, model_values = model.translate_to(
            TreeDemographicModel,
            model_values
        )

    return model, model_values


def print_momi_model(engine, values, nanc, gen_time):
    # Begin to generate model code
    if isinstance(engine.model, CustomDemographicModel):
        ret_str = dadi_generator._print_dadi_func(engine.model, values)
        return "import momi\n" + ret_str

    model, model_values = get_anc_model(engine.model, values, nanc)

    assert isinstance(model, TreeDemographicModel)

    variables_phys = copy.deepcopy(engine.model.variables)
    for i in range(len(variables_phys)):
        variables_phys[i].translate_units_to("physical")

    ret_str = "import momi\n"
    ret_str += "import numpy as np\n\n"

    ret_str += "def model_func(params):\n"
    params_ids = ", ".join([var.name for var in model.variables])
    ret_str += f"\t{params_ids} = params\n\n"

    ret_str += f"\tmodel = momi.DemographicModel(\n"
    ret_str += "\t\tN_e=1,\n"
    ret_str += "\t\tgen_time=1,\n"
    ret_str += f"\t\tmuts_per_gen={engine.model.mutation_rate}\n"
    ret_str += "\t)\n\n"

    for var in variables_phys:
        if isinstance(var, DynamicVariable):
            continue
        if isinstance(var, TimeVariable):
            ret_str += "\tmodel.add_time_param"
        elif isinstance(var, PopulationSizeVariable):
            ret_str += "\tmodel.add_size_param"
        elif isinstance(var, FractionVariable):
            ret_str += "\tmodel.add_pulse_param"
        else:
            continue
        ret_str += f"('{var.name}', lower={var.domain[0]}, "\
                   f"upper={var.domain[1]})\n"
    ret_str += "\n"

    n_pop = engine.model.number_of_populations()
    population_labels = [engine._get_pop_name(i) for i in range(n_pop)]
    for event in model.events:
        if isinstance(event, (Leaf, PopulationSizeChange)):
            name = population_labels[event.pop]
            if isinstance(event, Leaf):
                ret_str += "\tmodel.add_leaf(\n"
            else:
                ret_str += "\tmodel.set_size(\n"
            ret_str += f"\t\tpop_name='{name}',\n"
            ret_str += f"\t\tt={get_momi_name(event.t)},\n"
            ret_str += f"\t\tN={get_momi_name(event.size_pop)},\n"
            ret_str += f"\t\tg={get_momi_name(event.g)},\n"
            ret_str += "\t)\n"
        if isinstance(event, LineageMovement):
            name1 = population_labels[event.pop_from]
            name2 = population_labels[event.pop]
            ret_str += "\tmodel.move_lineages(\n"
            ret_str += f"\t\tpop_from='{name1}',\n"
            ret_str += f"\t\tpop_to='{name2}',\n"
            ret_str += f"\t\tt={get_momi_name(event.t)},\n"
            ret_str += f"\t\tp={get_momi_name(event.p)},\n"
            ret_str += f"\t\tN={get_momi_name(event.size_pop)},\n"
            ret_str += f"\t\tg={get_momi_name(event.g)},\n"
            ret_str += "\t)\n"

    ret_str += "\n\tmodel.set_params({\n"
    for var in model.variables:
        ret_str += f"\t\t'{var.name}': {var.name},\n"
    ret_str += "\t})\n"
    ret_str += "\treturn model\n\n"

    return ret_str


def print_data_reading(engine):
    ret_str = ""
    populations = engine.data_holder.population_labels
    if isinstance(engine.data_holder, SFSDataHolder):
        is_dadi_fs = dadi_generator._is_fs_via_dadi(engine.data_holder)
        if is_dadi_fs is not None:
            filename = f"'{engine.data_holder.filename}'"
            ret_str = f"data = momi.sfs_from_dadi({filename})\n"
        else:
            ret_str = f"import gadma\n"
            ret_str += "data_holder = gadma.SFSDataHolder("
            ret_str += f"'{engine.data_holder.filename}')\n"
            ret_str += "data = gadma.get_engine('momi')"\
                       ".read_data(data_holder)\n"

    if isinstance(engine.data_holder, VCFDataHolder):
        ind2pop = engine._get_ind2pop(engine.data_holder)
        str_list = []
        for key, value in ind2pop.items():
            str_list.append(f"'{key}': '{value}'")
        ret_str += "ind2pop = {" + ", ".join(str_list) + "}\n"
        ret_str += f"data = momi.SnpAlleleCounts.read_vcf("\
                   f"'{engine.data_holder.filename}', "\
                   "ind2pop=ind2pop).extract_sfs(n_blocks=100)\n"
        _, pops = read_popinfo(engine.data_holder.popmap_file)
        if populations is None or not set(populations) != set(pops):
            populations = pops
    if engine.data_holder.outgroup is False:
        ret_str += "data = data.fold()\n"
    if populations is not None:
        ret_str += f"data = data.subset_populations({list(populations)})\n"
    ret_str += "\n"
    return ret_str


def print_main_part(engine, values, nanc):
    model, model_values = get_anc_model(engine.model, values, nanc)

    var2value = engine.model.var2value(values)

    if isinstance(model, CustomDemographicModel):
        Nanc = nanc
        var2value = model.var2value(values)
        values_phys = [var2value[var] for var in model.variables]
    else:
        # momi has values in physical units
        Nanc = model.get_Nanc_size(model_values)
        values_phys = model.translate_values(
            units="physical",
            values=model_values,
            Nanc=Nanc
        )

    ret_str = f"params = {values_phys}\n"
    ret_str += "model = model_func(params)\n"
    if isinstance(engine.model, CustomDemographicModel):
        ret_str += "model.gen_time = 1\n"
        ret_str += f"model.muts_per_gen = {engine.model.mutation_rate}\n"
    length = engine.data_holder.get_total_sequence_length()
    assert length is not None, "Sequence length is required"

    ret_str += f"model.set_data(data, length={length})\n"
    ret_str += "ll_model = model.log_likelihood()\n"
    ret_str += "print(f'Value of log-likelihood: {ll_model}')"
    return ret_str


def print_plotting_part(engine):
    ret_str = "\nfrom matplotlib import pyplot as plt\n"
    ret_str += "momi.DemographyPlot(\n"
    ret_str += "\tmodel,\n"
    ret_str += "\tpop_x_positions=data.sampled_pops,\n"
    ret_str += "\tfigsize=(6,8),\n"
    ret_str += "\tlinthreshy=None,\n"
    ret_str += "\tpulse_color_bounds=(0,.25)\n"
    ret_str += ")\n"
    ret_str += "plt.savefig('model_from_GADMA.png')\n"
    return ret_str


def print_momi_code(engine, values, filename,
                    nanc=None, gen_time=1, gen_time_units=None):
    valid_units = ["generations", "years"]
    assert gen_time_units.lower() in valid_units, ("Units of generation time "
                                                   f"should be {valid_units}")
    # check for dynamics
    var2value = engine.model.var2value(values)
    assert "Lin" not in var2value.values(), ("Linear size function is "
                                             " not supported")

    ret_str = print_momi_model(engine, values, nanc, gen_time=gen_time)
    ret_str += "\n# Momi does not supports downsizing of the SFS so in this"\
               " code there is no downsizing.\n"
    ret_str += print_data_reading(engine)
    ret_str += print_main_part(engine, values, nanc)
    ret_str += print_plotting_part(engine)
    if filename is None:
        return ret_str
    with open(filename, 'w') as f:
        f.write(ret_str)
