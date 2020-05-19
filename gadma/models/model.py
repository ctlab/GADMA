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
        self.variables = VariablePool()
        self.raise_excep = raise_excep

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
            if variable not in self.variables:
                self.variables.append(variable)

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

    def var2value(self, values):
        if isinstance(values, list):
            return {var: value for var, value in zip(self.variables,
                                                               values)}
        elif isinstance(values, dict):
            for key in values:
                if isinstance(key, str):
                    return {var: values[var.name] for var in self.variables}
                elif isinstance(key, Variable):
                    return {var: values[var] for var in self.variables}
