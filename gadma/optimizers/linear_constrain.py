import scipy
import numpy as np


def my_dot(A_i, x):
    """
    Implemented easy version of multiplication of two vectors (row and column).
    """
    res = 0
    for _a, _x in zip(A_i, x):
        if _a != 0:
            res += _x * _a
    return res


class LinearConstrain(object):
    """
    Class containing linear constrain. `lb <= A*x <= ub`

    :param A: Matrix.
    :param lb: Lower bound.
    :param ub: Upper bound.
    """
    def __init__(self, A, lb, ub):
        lb = np.nan_to_num(np.array(lb, dtype=np.float), nan=-np.inf)
        ub = np.nan_to_num(np.array(ub, dtype=np.float), nan=np.inf)
        self.constrain = scipy.optimize.LinearConstraint(np.array(A), lb, ub)

    def _get_value(self, x):
        """
        Multiply matrix A by `x`.

        :param x: Vector.
        """
        return np.array([my_dot(A_i, x) for A_i in self.constrain.A])

    def fits(self, x):
        """
        Checks that `x` is good for constrain.

        :param x: Vector to check that `lb <= A*x <=ub`.
        """
        Ax = self._get_value(x)
        lb = self.constrain.lb
        ub = self.constrain.ub
        return (np.all(np.logical_or(Ax > lb, np.isclose(Ax, lb))) and
                np.all(np.logical_or(Ax < ub, np.isclose(Ax, ub))))

    def try_to_transform(self, x):
        """
        Try to transform x to fit the constrain.
        Current implementation go through each pair of bounds and if `x`
        multiplied by the corresponding column of `A` does not fit bounds then
        values of `x` that take part in this constrain are changed so that
        they fit the bounds.

        :param x: Vector to change.

        :returns: transformed x and bool if transformation was successful.
        """
        x_tr = np.array(x)
        for A_i, lb_i, ub_i in zip(self.constrain.A,
                                   self.constrain.lb, self.constrain.ub):
            A_ix = my_dot(A_i, x_tr)
            if A_ix < lb_i:
                C = lb_i
            if A_ix > ub_i:
                C = ub_i
            if A_ix < lb_i or A_ix > ub_i:
                involved = (A_i != 0)
                sign = np.sign(A_i[involved])
                x_tr[involved] = sign * C / np.sum(np.abs(A_i[involved]))
        success = self.fits(x)
        return x_tr, success

    @property
    def lb(self):
        return self.constrain.lb

    @property
    def ub(self):
        return self.constrain.ub

    @property
    def A(self):
        return self.constrain.A

    @lb.setter
    def lb(self, new_value):
        new_value = np.nan_to_num(np.array(new_value, dtype=np.float),
                                  nan=-np.inf)
        self.constrain.lb = new_value

    @ub.setter
    def ub(self, new_value):
        new_value = np.nan_to_num(np.array(new_value, dtype=np.float),
                                  nan=np.inf)
        self.constrain.ub = new_value

    @A.setter
    def A(self, new_value):
        self.constrain.A = new_value

    def __str__(self):
        """
        String representation of constrain.
        """
        lb_size = 0
        ub_size = 0
        A_size = 0
        for lb_i, A_i, ub_i in zip(self.lb, self.A, self.ub):
            lb_size = max(lb_size, len(str(self.lb)))
            ub_size = max(ub_size, len(str(self.ub)))
            A_size = max(A_size, len(str(A_i)))

        lb_str = "{0: <" + str(lb_size) + "}"
        ub_str = "{0: <" + str(ub_size) + "}"
        A_str = "{0: <" + str(A_size) + "}"

        ret_str = "\t".join([lb_str.format("lb"), "|",  A_str.format("A"),
                             "|", ub_str.format("ub")]) + "\n"
        for lb_i, A_i, ub_i in zip(self.lb, self.A, self.ub):
            ret_str += "\t".join([lb_str.format(lb_i), "|",
                                  A_str.format(str(A_i)), "|",
                                  ub_str.format(ub_i)]) + "\n"
        return ret_str


# class LinearConstrainDemographics(LinearConstrain):
#     def __init__(self, A, lb, ub, engine, engine_args):
#         self.engine = engine
#         self.engine_args = engine_args
#         self.original_lb = lb
#         self.original_ub = ub
#         super(LinearConstrainDemographics, self).__init__(A, lb, ub)
#
#     def fits(self, x):
#         theta = self.engine.get_theta(x, *self.engine_args)
#         self.constrain.lb = self.original_lb / theta
#         self.constrain.ub = self.original_ub / theta
#         return super(LinearConstrainDemographics, self).fits(x)
#
#     def try_to_transform(self, x):
#         theta = self.engine.get_theta(x, *self.engine_args)
#         lb = self.original_lb / theta
#         ub = self.original_ub / theta
#         lin_constr = LinearConstrain(self.constrain.A, lb, ub)
#         print("!!!", ub)
#         return lin_constr.try_to_transform(x)
