import numpy as np
from ..utils import Variable, VariablePool, float_repr


class Model(object):
    """
    Abstract class of model. Contains variables of class :class:`Variable`.

    :param raise_excep: if `True` then raises exception when something except
                        :class:`Variable` is added as variable.
    :type raise_excep: bool
    """
    def __init__(self, raise_excep=False):
        self.is_fixed = []
        self.fixed_values = {}
        self._variables = VariablePool()
        self.raise_excep = raise_excep

    @property
    def variables(self):
        return [var for var, fixed in zip(self._variables, self.is_fixed)
                if not fixed]

    @variables.setter
    def variables(self, new_variables):
        self._variables = new_variables
        self.is_fixed = [False for var in new_variables]

    def add_variable(self, variable):
        """
        Adds one variable to the model.

        :param variable: variable to add.
        :type variable: :class:`Variable`
        """
        if not isinstance(variable, (Variable, Model)):
            if self.raise_excep:
                raise ValueError("Only instances of class Variable could be"
                                 " added to the model as variables.")
        else:
            if isinstance(variable, Model):
                for var in variable.variables:
                    self.add_variable(var)
                return
            if variable not in self._variables:
                self._variables.append(variable)
                self.is_fixed.append(False)

    def add_variables(self, variables):
        """
        Adds several variables to the model.

        :param variables: variables to add.
        :type variables: list
        """
        for variable in np.array(variables).flatten():
            self.add_variable(variable)

    def get_variable(self, name):
        """
        Finds the variable of the model by its name.

        :param name: name of the model's variable.
        :type name: str

        :raises ValueError: if there is no variable with that name.
        """
        for var in self.variables:
            if var.name == name:
                return var

    def fix_variable(self, variable, value):
        if variable not in self.variables:
            raise ValueError(f"There is no such unfixed variable {variable} "
                             "in the model.")
        self.is_fixed[self._variables.index(variable)] = True
        self.fixed_values[variable] = value

    def unfix_variable(self, variable):
        if variable not in self.fixed_values:
            raise ValueError(f"There is no such fixed variable {variable} "
                             "in the model.")
        ind = self._variables.index(variable)
        self.is_fixed[ind] = False
        del self.fixed_values[variable]

    def unfix_if_fixed(self, variable):
        try:
            self.unfix_variable(variable)
        except ValueError:
            pass

    def var2value(self, values):
        """
        Returns dictionary {variable: value}.

        :param values: List or dict {var_name: value} of values.
        """
        if len(self.variables) == 0:
            return {}
        if isinstance(values, list) or isinstance(values, np.ndarray):
            ret_dict = {var: value for var, value in zip(self.variables,
                                                         values)}
        elif isinstance(values, dict):
            ret_dict = {}
            for key in values:
                if isinstance(key, str):
                    var = self.get_variable(key)
                    if var is not None:
                        ret_dict[var] = values[key]
                elif isinstance(key, Variable):
                    if key in self.variables:
                        ret_dict[key] = values[key]
                    else:
                        var = self.get_variable(key.name)
                        if var is not None:
                            ret_dict[var] = values[key]
        else:
            raise TypeError("Values are either not list nor dict.")

        assert len(ret_dict) == len(self.variables)
        return {**ret_dict, **self.fixed_values}

    def string_repr(self, values):
        """
        Returns string representation of variables and values.

        :param values: Values of the variables in model.
        """
        strings = []
        var2value = self.var2value(values)
        for var in self.variables:
            strings.append(f"{var.name}={var2value[var]}")
        return ", ".join(strings)

    def _arg_val_repr(self, arg, values):
        """
        Returns list of arg and its value string representations
        """
        from .variables_combinations import BinaryOperation
        if isinstance(arg, Variable):
            val = self.var2value(values)[arg]
            arg_repr = f"{arg.name}"
        elif isinstance(arg, BinaryOperation):
            var2value = self.var2value(values)
            val = arg.get_value([var2value[var] for var in arg.variables])
            arg_repr = f"{arg.name}"
        else:
            val = arg
            arg_repr = ""
        if isinstance(val, float):
            val_repr = float_repr(val, precision=3)
        else:
            val_repr = str(val)
        return arg_repr, val_repr
