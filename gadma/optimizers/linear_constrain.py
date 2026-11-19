import scipy
import numpy as np

from ..utils import get_correct_dtype


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

    Constrain could have a factor: `lb <= factor * A*x <= ub`. The factor is
    either a number or one of the values of `x` that is not involved in
    `A*x`. E.g. bounds of the split times are in generations and time in `x`
    is in genetic units, then the factor is the size of ancestral population.
    It is constant or it is one of the values of `x`.

    :param A: Matrix.
    :param lb: Lower bound.
    :param ub: Upper bound.
    :param factor: Number that `A*x` is multiplied by.
    :param factor_index: Index of the value of `x` that `A*x` is multiplied
                         by. If it is set then `factor` is not used.
    """
    factor = 1
    factor_index = None

    def __init__(self, A, lb, ub, factor=1, factor_index=None):
        lb = np.nan_to_num(np.array(lb, dtype=np.float64), nan=-np.inf)
        ub = np.nan_to_num(np.array(ub, dtype=np.float64), nan=np.inf)
        self.constrain = scipy.optimize.LinearConstraint(np.array(A), lb, ub)
        self.factor = factor
        self.factor_index = factor_index

    def _get_value(self, x):
        """
        Multiply matrix A by `x`.

        :param x: Vector.
        """
        return np.array([my_dot(A_i, x) for A_i in self.constrain.A])

    def _get_factor(self, x):
        """
        Returns the number that `A*x` is multiplied by for the given `x`.

        :param x: Vector.
        """
        if self.factor_index is not None:
            return x[self.factor_index]
        return self.factor

    def fits(self, x, factor=None):
        """
        Checks that `x` is good for constrain.

        :param x: Vector to check that `lb <= A*x <=ub`.
        :param factor: Number to multiply `A*x` by before the check, i.e.
                       it is checked that `lb <= factor*A*x <= ub`. E.g.
                       bounds of the split times are in generations and time
                       in `x` is in genetic units, then factor is the size
                       of ancestral population. If None then the factor of
                       the constrain is used.
        """
        if factor is None:
            factor = self._get_factor(x)
        Ax = factor * self._get_value(x)
        lb = self.constrain.lb
        ub = self.constrain.ub
        return (np.all(np.logical_or(Ax > lb, np.isclose(Ax, lb))) and
                np.all(np.logical_or(Ax < ub, np.isclose(Ax, ub))))

    def try_to_transform(self, x, factor=None):
        """
        Try to transform x to fit the constrain.

        If a row of the constrain is broken then values of `x` that are
        involved in it are multiplied by the same number so that the row
        becomes equal to its bound, i.e. the proportions between these values
        are kept.

        Rows are processed from the row with the smallest number of involved
        values. In order to keep the rows that are satisfied already, at
        first it is tried to change only the values that are not involved in
        the rows that were processed. E.g. for bounds of the split times the
        time of the last split is corrected first and then for the previous
        split only the time between two splits is changed.

        :param x: Vector to change.
        :param factor: Number to multiply `A*x` by (see :meth:`fits`). If
                       None then the factor of the constrain is used.

        :returns: transformed x and bool if transformation was successful,
                  i.e. if transformed x fits the constrain. If it was not
                  successful then x is returned without changes.
        """
        if factor is None:
            factor = self._get_factor(x)
        x_orig = np.array(x, dtype=get_correct_dtype(x))
        x_tr = np.array(x_orig)
        A = np.array(self.constrain.A)
        order = np.argsort([np.sum(A_i != 0) for A_i in A], kind="stable")
        is_processed = np.zeros(A.shape[1], dtype=bool)
        for i in order:
            A_i, lb_i, ub_i = A[i], self.constrain.lb[i], self.constrain.ub[i]
            is_involved = (A_i != 0)
            value = factor * my_dot(A_i, x_tr)
            bound = None
            if value < lb_i and not np.isclose(value, lb_i):
                bound = lb_i
            elif value > ub_i and not np.isclose(value, ub_i):
                bound = ub_i
            if bound is not None:
                is_own = np.logical_and(is_involved,
                                        np.logical_not(is_processed))
                # If own values of the row are not enough then all involved
                # values are changed, the final check will tell if other
                # rows are still satisfied.
                for is_changed in [is_own, is_involved]:
                    changed = factor * my_dot(A_i * is_changed, x_tr)
                    new_changed = bound - (value - changed)
                    # values are multiplied by positive number only
                    if changed != 0 and new_changed / changed > 0:
                        for j in np.where(is_changed)[0]:
                            x_tr[j] = x_tr[j] * (new_changed / changed)
                        break
                else:
                    return x_orig, False
            is_processed = np.logical_or(is_processed, is_involved)
        if not self.fits(x_tr, factor=factor):
            return x_orig, False
        return x_tr, True

    def for_part_of_x(self, is_in_part):
        """
        Returns constrain for the vector that consists of a part of the
        values of `x`. E.g. local search optimizes continuous values only.
        Values that are not taken should not be involved in the constrain.

        :param is_in_part: List of bools, True for values of `x` that the new
                           vector consists of.
        """
        is_in_part = np.array(is_in_part, dtype=bool)
        A = np.array(self.constrain.A)
        if np.any(A[:, np.logical_not(is_in_part)] != 0):
            raise ValueError("Values that are involved in linear constrain "
                             "could not be excluded.")
        factor_index = self.factor_index
        if factor_index is not None:
            if not is_in_part[factor_index]:
                raise ValueError("Value that is the factor of linear "
                                 "constrain could not be excluded.")
            factor_index = int(np.sum(is_in_part[:factor_index]))
        return LinearConstrain(A[:, is_in_part], self.lb, self.ub,
                               factor=self.factor, factor_index=factor_index)

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
        new_value = np.nan_to_num(np.array(new_value, dtype=np.float64),
                                  nan=-np.inf)
        self.constrain.lb = new_value

    @ub.setter
    def ub(self, new_value):
        new_value = np.nan_to_num(np.array(new_value, dtype=np.float64),
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
