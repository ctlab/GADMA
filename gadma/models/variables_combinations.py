import math

import numpy as np

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


class Operation(VariablesCombination):

    def name(self):
        raise NotImplementedError

    def get_value(self, values):
        raise NotImplementedError

    def string_repr(self, values):
        raise NotImplementedError


class UnaryOperation(Operation):
    """
    Unary operation of two variables.

    :param arg: First argument.
    :type arg: :class:`gadma.Variable` or value.

    :raises AssertError: if arguments are not variables.
    """

    def __init__(self, arg):
        super(UnaryOperation, self).__init__()
        self.arg = arg
        self.add_variable(arg)
        assert len(self.variables) > 0

    def __eq__(self, other):
        if not isinstance(other, UnaryOperation):
            return False
        return self.operation_str() == other.operation_str() \
            and self.arg == other.arg

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        """
        Generates name from variables names like:
        `operation self.arg.name`
        """
        assert isinstance(self.arg, (Variable, Operation))
        arg_name = self.arg.name
        if isinstance(self.arg, Operation):
            arg_name = f"({arg_name})"
        return f"{self.operation_str()} {arg_name}"

    def get_value(self, values):
        """
        Returns value of the unary operation from variables values.

        :param values: Values of the variables.
        :type values: list of dict
        """
        var2value = self.var2value(values)
        if isinstance(self.arg, Operation):
            vals = [var2value[var] for var in self.arg.variables]
            val = self.arg.get_value(vals)
        else:
            val = var2value.get(self.arg, self.arg)
        return self.operation(val)

    def string_repr(self, values):
        """
        Returns string representation of unary operation with defined values.

        :param values: Values of the variables.
        :type values: list of dict
        """
        val = self.get_value(values)
        return f"{self.name}={val}"

    @staticmethod
    def operation(val):
        """
        Returns the result of unary operation from value.
        """
        raise NotImplementedError

    @staticmethod
    def operation_str():
        """
        Returns string representation of unary operation.
        """
        raise NotImplementedError


class BinaryOperation(Operation):
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

    def __eq__(self, other):
        if self is other:
            return True
        return type(self) == type(other) and (
                self.arg1 == other.arg1 and self.arg2 == other.arg2
                or
                self.is_commutative() and
                self.arg1 == other.arg2 and self.arg2 == other.arg1
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        """
        Generates name from variables names like:
        `self.arg1.name operation self.arg1.name`
        """
        if isinstance(self.arg1, (Variable, Operation)):
            arg1_name = self.arg1.name
            if isinstance(self.arg1, Operation):
                arg1_name = f"({arg1_name})"
        else:
            arg1_name = self.arg1
        if isinstance(self.arg2, (Variable, Operation)):
            arg2_name = self.arg2.name
            if isinstance(self.arg2, Operation):
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
        if isinstance(self.arg1, Operation):
            vals = [var2value[var] for var in self.arg1.variables]
            val1 = self.arg1.get_value(vals)
        else:
            val1 = var2value.get(self.arg1, self.arg1)
        if isinstance(self.arg2, Operation):
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

    def is_commutative(self):
        raise NotImplementedError

    @staticmethod
    def operation(val1, val2):
        """
        Returns the result of binary operation from two values.
        """
        raise NotImplementedError

    @staticmethod
    def operation_str():
        """
        Returns string representation of binary operation.
        """
        raise NotImplementedError


class Addition(BinaryOperation):
    """
    The sum of two variables.
    """

    def is_commutative(self):
        return True

    @staticmethod
    def operation(val1, val2):
        return val1 + val2

    @staticmethod
    def operation_str():
        return "+"


class Subtraction(BinaryOperation):
    """
    The subtraction of two variables.
    """

    def is_commutative(self):
        return False

    @staticmethod
    def operation(val1, val2):
        return val1 - val2

    @staticmethod
    def operation_str():
        return "-"


class Multiplication(BinaryOperation):
    """
    The multiplication of two variables.
    """

    def is_commutative(self):
        return True

    @staticmethod
    def operation(val1, val2):
        return val1 * val2

    @staticmethod
    def operation_str():
        return "*"


class Division(BinaryOperation):
    """
    The division of one variable by another.
    """

    def is_commutative(self):
        return False

    @staticmethod
    def operation(val1, val2):
        return val1 / val2

    @staticmethod
    def operation_str():
        return "/"


class Pow(BinaryOperation):

    def is_commutative(self):
        return False

    @staticmethod
    def operation(val1, val2):
        return val1 ** val2

    @staticmethod
    def operation_str():
        return "**"


class Exp(UnaryOperation):

    @staticmethod
    def operation(val):
        return np.exp(val)

    @staticmethod
    def operation_str():
        return "exp"


class Log(UnaryOperation):

    @staticmethod
    def operation(val):
        return math.log(val)

    @staticmethod
    def operation_str():
        return "log"


def operation_creation(operation, arg1, arg2=None):
    """
    Create of operation with some simplifications.

    :param operation: class of operation
    :type operation: Name of operation
    :param arg1: First argument.
    :type arg1: :class:`gadma.Variable` or value.
    :param arg2: Second argument.
    :type arg2: :class:`gadma.Variable` or value.

    :raises ValueError: if type of operation is unknown.
    """
    if issubclass(operation, UnaryOperation):
        if arg2 is not None:
            raise ValueError(
                f"{operation} should take 1 arguments"
            )
    else:
        if arg2 is None:
            raise ValueError(
                f"{operation} should take 2 arguments"
            )
    if isinstance(arg1, (Model, Variable)) or \
            isinstance(arg2, (Model, Variable)):
        if operation is Addition:
            return create_addition(arg1, arg2)
        if operation is Subtraction:
            return create_subtraction(arg1, arg2)
        if operation is Division:
            return create_division(arg1, arg2)
        if operation is Multiplication:
            return create_multiplication(arg1, arg2)
        if operation is Exp:
            return create_exp(arg1)
        if operation is Log:
            return create_log(arg1)
        raise ValueError(
            f"Can not create operation {operation}"
        )
    else:
        if issubclass(operation, UnaryOperation):
            return operation.operation(arg1)
        return operation.operation(arg1, arg2)


def create_addition(arg1, arg2):
    if not isinstance(arg1, (Model, Variable)):
        if np.isclose(arg1, 0):
            return arg2
    if not isinstance(arg2, (Model, Variable)):
        if np.isclose(arg2, 0):
            return arg1
    return Addition(arg1, arg2)


def create_subtraction(arg1, arg2):
    if not isinstance(arg2, (Model, Variable)):
        if np.isclose(arg2, 0):
            return arg1
    if isinstance(arg1, (Model, Variable)) \
            and isinstance(arg2, (Model, Variable)):
        if arg1 == arg2:
            return 0
    return Subtraction(arg1, arg2)


def create_division(arg1, arg2):
    if isinstance(arg1, (Model, Variable)) \
            and isinstance(arg2, (Model, Variable)):
        if arg1 == arg2:
            return 1
    if not isinstance(arg2, (Model, Variable)):
        if np.isclose(arg2, 0):
            raise ValueError("Can not divide by zero")
        if np.isclose(arg2, 1):
            return arg1
    return Division(arg1, arg2)


def create_multiplication(arg1, arg2):
    if not isinstance(arg1, (Model, Variable)):
        if np.isclose(arg1, 0):
            return 0
        if np.isclose(arg1, 1):
            return arg2
    if not isinstance(arg2, (Model, Variable)):
        if np.isclose(arg2, 0):
            return 0
        if np.isclose(arg2, 1):
            return arg1
    return Multiplication(arg1, arg2)


def create_exp(arg):
    if isinstance(arg, Log):
        return arg.arg
    return Exp(arg)


def create_log(arg):
    if isinstance(arg, Exp):
        return arg.arg
    return Log(arg)
