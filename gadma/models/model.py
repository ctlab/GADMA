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
        self.var2value = None

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

    def randomize(self):
        """
        Sets values of the model variables to random values.
        For every variable :meth:`.Variable.resample` is calling.
        """
        self.var2value = {var: var.resample() for var in self.variables}

    def set_values(self, values):
        """
        Sets variable values to `values`. `values` could be the list of values
        or a dictionary of values with names of the variables as the keys.

        :param values: values of variables of the model.
        :type values: list or dict
        """
        if isinstance(values, list):
            self.var2value = {var: value for var, value in zip(self.variables,
                                                               values)}
        elif isinstance(values, dict):
            self.var2value = {var: value[var.name] for var in self.variables}
        else:
            raise ValueError("Values are not either list nor dict instance.")

    def get_value(self, variable):
        """
        Returns value of the variable.

        :param variable: variable to get its value in the model.
        :type variable:  :class:`.Variable`

        :raises AttributeError: if there is no such variable or if no values\
            were set for the model.
        """
        if self.var2value is None:
            raise AttributeError("Model has no setted values of the"
                                 " variables. Please call set_values or"
                                 " randomize function first.")
        return self.var2value[variable]

    def get_values(self):
        """
        Returns list of variables values.
        """
        return list([self.var2value[var] for var in self.variables])
