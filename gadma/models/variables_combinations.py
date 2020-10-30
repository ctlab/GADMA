from ..utils import Variable
from .model import Model


class VariablesCombination(Model):
    """
    Base class for combination of variables.
    """
    def __init__(self):
        super(VariablesCombination, self).__init__(raise_excep=False)

    def __str__(self):
        return f"VariablesCombination of {self.variables} variables"


class BinaryOperation(VariablesCombination):
    """
    Combination of two variables.

    :param arg1: First argument.
    :type arg1: :class:`gadma.Variable` or value.
    :param arg2: Second argument.
    :type arg2: :class:`gadma.Variable` or value.

    :raises AssertError: if both arguments are not variables.
    """
    def __init__(self, arg1, arg2):
        super(BinaryOperation, self).__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.add_variables([arg1, arg2])
        assert len(self.variables) > 0

    @property
    def name(self):
        """
        Generates name from variables names like:
        `self.arg1.name operation self.arg1.name`
        """
        if isinstance(self.arg1, (Variable, BinaryOperation)):
            arg1_name = self.arg1.name
            if isinstance(self.arg1, BinaryOperation):
                arg1_name = f"({arg1_name})"
        else:
            arg1_name = self.arg1
        if isinstance(self.arg2, (Variable, BinaryOperation)):
            arg2_name = self.arg2.name
            if isinstance(self.arg2, BinaryOperation):
                arg2_name = f"({arg2_name})"
        else:
            arg2_name = self.arg2
        return f"{arg1_name} {self.operation_str()} {arg2_name}"

    def get_value(self, values):
        """
        Returns value of the combination from variables values.

        :param values: Values of the variables.
        :type values: list of dict
        """
        var2value = self.var2value(values)
        if isinstance(self.arg1, BinaryOperation):
            vals = [var2value[var] for var in self.arg1.variables]
            val1 = self.arg1.get_value(vals)
        else:
            val1 = var2value.get(self.arg1, self.arg1)
        if isinstance(self.arg2, BinaryOperation):
            vals = [var2value[var] for var in self.arg2.variables]
            val2 = self.arg2.get_value(vals)
        else:
            val2 = var2value.get(self.arg2, self.arg2)
        return self.operation(val1, val2)

    def string_repr(self, values):
        """
        Returns string representation of combination with defined values.

        :param values: Values of the variables.
        :type values: list of dict
        """
        val = self.get_value(values)
        return f"{self.name}={val}"

    def operation(self, val1, val2):
        """
        Returns the result of binary operation from two values.
        """
        raise NotImplementedError

    def operation_str(self):
        """
        Returns string representation of binary operation.
        """
        raise NotImplementedError


class Addition(BinaryOperation):
    """
    The sum of two variables.
    """
    def operation(self, val1, val2):
        return val1 + val2

    def operation_str(self):
        return "+"


class Subtraction(BinaryOperation):
    """
    The subtraction of two variables.
    """
    def operation(self, val1, val2):
        return val1 - val2

    def operation_str(self):
        return "-"


class Multiplication(BinaryOperation):
    """
    The multiplication of two variables.
    """
    def operation(self, val1, val2):
        return val1 * val2

    def operation_str(self):
        return "*"


class Division(BinaryOperation):
    """
    The division of one variable by another.
    """
    def operation(self, val1, val2):
        return val1 / val2

    def operation_str(self):
        return "/"
