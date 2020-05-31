import numpy as np
from ..utils import Variable, VariablePool


class Model(object):
    """
    Abstract class of model. Contains variables of class :class:`Variable`.

    :param raise_excep: if `True` then raises exception when something except
                        :class:`Variable` is added as variable.
    :type raise_excep: bool
    """
    def __init__(self, raise_excep=True):
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
        if not isinstance(variable, Variable):
            if self.raise_excep:
                raise ValueError("Only instances of class Variable could be"
                                 " added to the model as variables.")
        else:
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

    def unfix(self, variable):
        if variable not in self.fixed_values:
            raise ValueError(f"There is no such fixed variable {variable} "
                             "in the model.")
        ind = self._variables.index(variable)
        self.is_fixed[ind] = False
        del self.fixed_values[variable]

    def unfix_if_fixed(self, variable):
        try:
            self.unfix(variable)
        except ValueError:
            pass

    def var2value(self, values):
        if isinstance(values, list) or isinstance(values, np.ndarray):
            ret_dict = {var: value for var, value in zip(self.variables,
                                                               values)}
        elif isinstance(values, dict):
            for key in values:
                if isinstance(key, str):
                    ret_dict = {var: values[var.name] for var in self.variables}
                elif isinstance(key, Variable):
                    ret_dict = {var: values[var] for var in self.variables}
        else:
            raise TypeError("Values are either not list nor dict.")

        return {**ret_dict, **self.fixed_values}

    def string_repr(self, values):
        strings = []
        var2value = self.var2value(values)
        for var in self.variables:
            strings.append(f"{var.name}={var2value[var]}")
        return ", ".join(strings)
