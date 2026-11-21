"""
Functions that are used in the generation of code for several engines.
"""
from ..models import BinaryOperation
from ..utils import Variable, DiscreteVariable, DynamicVariable
import importlib
import inspect
import os
import numpy as np

FUNCTION_NAME = "model_func"


def _value_to_code(value):
    """
    Returns code of the value: numpy arrays and numbers are written as lists
    and numbers of Python, strings are in quotes.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value_to_code(el) for el in value) + "]"
    if isinstance(value, dict):
        items = [f"{_value_to_code(key)}: {_value_to_code(el)}"
                 for key, el in value.items()]
        return "{" + ", ".join(items) + "}"
    if isinstance(value, np.generic):
        value = value.item()
    return repr(value)


def _read_fs_or_none(module_name, data_holder):
    """
    Returns allele frequency spectrum of the data if its file is in the
    format of the module (dadi or moments), otherwise returns None.

    :param module_name: Name of the module, 'dadi' or 'moments'.
    :param data_holder: Data holder with data.
    :type data_holder: :class:`gadma.data.data_holder.DataHolder`
    """
    module = importlib.import_module(module_name)
    try:
        return module.Spectrum.from_file(data_holder.filename)
    except Exception:
        return None


def _get_variables_of_function(model):
    """
    Returns variables that are parameters of the function of demographic
    model in generated code: they are all variables of the model except
    discrete ones, as dynamics are fixed in the code, and except the size of
    ancestral population, as it is equal to 1 in the function.
    """
    f_vars = [x for x in model.variables
              if not isinstance(x, DiscreteVariable)]
    if model.has_anc_size and isinstance(model.Nanc_size, Variable):
        f_vars.pop(f_vars.index(model.Nanc_size))
    return f_vars


def _print_unpacking_of_params(variables, indent="\t"):
    """
    Returns line of code that takes values of `variables` from `params` in
    the beginning of the function of demographic model.

    The line is empty if there are no variables. For one variable it is
    `name, = params`, as `name = params` is the whole list and not the value.

    :param variables: Variables which values are parameters of the function.
    :param indent: Indent of the line.
    """
    if len(variables) == 0:
        return ""
    names = ", ".join([var.name for var in variables])
    if len(variables) == 1:
        names += ","
    return f"{indent}{names} = params\n"


def _print_custom_model_import(model):
    """
    Returns code that takes function of custom demographic model from its
    file. The module of the function could be not among loaded modules, e.g.
    in the process of a run, so the file is taken from the function.
    """
    path_repr = repr(os.path.abspath(inspect.getfile(model.function)))
    ret_str = "import importlib.util\n\n"
    ret_str += "spec = importlib.util.spec_from_file_location('module', " \
               f"{path_repr})\n"
    ret_str += "module = importlib.util.module_from_spec(spec)\n"
    ret_str += "spec.loader.exec_module(module)\n"
    ret_str += f"{FUNCTION_NAME} = module.model_func\n\n"
    return ret_str


def _print_size_functions(event, var2value, indent):
    """
    Returns code of functions of population sizes for populations of the
    epoch which sizes do not change suddenly, e.g. `nu1_func = lambda t: ...`,
    and list of names of these functions.

    :param event: Epoch of the model.
    :param var2value: Dictionary {variable: value}, values of dynamics are
                      required as they are fixed in the code.
    :param indent: Indent of the lines.
    """
    def arg_to_code(arg):
        if isinstance(arg, Variable):
            return arg.name
        if isinstance(arg, BinaryOperation):
            return f"({arg.name})"
        return arg

    ret_str = ""
    func_names = []
    if event.dyn_args is None:
        return ret_str, func_names
    for i in range(event.n_pop):
        dyn_arg = event.dyn_args[i]
        value = var2value.get(dyn_arg, dyn_arg)
        if value == 'Sud':
            continue
        func = DynamicVariable._help_dict[value]
        funcstr = func.func_str(arg_to_code(event.init_size_args[i]),
                                arg_to_code(event.size_args[i]),
                                arg_to_code(event.time_arg))
        ret_str += f"{indent}nu{i + 1}_func = {funcstr}\n"
        func_names.append(f"nu{i + 1}_func")
    return ret_str, func_names


def _print_integration_kwargs(kwargs_with_vars, size_key, func_names, indent):
    """
    Returns code that creates matrix of migrations of the epoch (it is empty
    if there are no migrations) and list of strings `key=value` with
    arguments of integration of moments or moments.LD.

    :param kwargs_with_vars: Arguments of integration with variables, they
                             are from `_get_kwargs` of the engine.
    :param size_key: Name of the argument with sizes of populations.
    :param func_names: Names of functions of population sizes, see
                       :func:`_print_size_functions`.
    :param indent: Indent of the lines.
    """
    ret_str = ""
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
            ret_str += f"{indent}migs = np.array([{', '.join(rows)}])\n"
            kwargs[x] = 'migs'
        elif isinstance(y, list):  # pop. sizes, selection, dynamics
            varnames = [var.name
                        if isinstance(var, (Variable, BinaryOperation))
                        else str(var) for var in y]
            if x == size_key:  # pop size
                if len(func_names) == 0:  # all sizes change suddenly
                    kwargs[x] = f"[{', '.join(varnames)}]"
                else:
                    strs = []
                    for y1 in y:
                        if isinstance(y1, (Variable, BinaryOperation)):
                            strs.append(y1.name)
                        elif y1 in func_names:
                            strs.append(y1 + "(t)")
                        else:
                            strs.append(str(y1))
                    kwargs[x] = f"lambda t: [{', '.join(strs)}]"
            else:
                kwargs[x] = f"[{','.join(varnames)}]"
        else:  # time
            kwargs[x] = y.name if isinstance(
                y, (Variable, BinaryOperation)) else y
    return ret_str, [f"{key}={value}" for key, value in kwargs.items()]
