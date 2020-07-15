import scipy
import numpy as np

def my_dot(A_i, x):
    res = 0
    for _a, _x in zip(A_i, x):
        if _a != 0:
            res += _x * _a
    return res

class LinearConstrain(object):
    def __init__(self, A, lb, ub):
        self.constrain = scipy.optimize.LinearConstraint(A, lb, ub)

    def _get_value(self, x):
        return np.array([my_dot(A_i, x) for A_i in self.constrain.A])

    def fits(self, x):
        Ax = self._get_value(x)
        lb = self.constrain.lb
        ub = self.constrain.ub
        return (np.all(np.logical_or(Ax > lb, np.isclose(Ax, lb))) and
                np.all(np.logical_or(Ax < ub, np.isclose(Ax, ub))))

    def try_to_transform(self, x):
        x_tr = np.array(x)
        for A_i, lb_i, ub_i in zip(self.constrain.A,
                                   self.constrain.lb, self.constrain.ub):
            A_ix = my_dot(A_i, x)
            if A_ix < lb_i:
                C = lb_i / A_ix 
            if A_ix > ub_i:
                C = A_ix / ub_i
            if A_ix < lb_i or A_ix > ub_i:
                involved = (A_i != 0)
                x_tr[involved] /= C * A_i[involved] / np.sum(A_i[involved])
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
        self.constrain.lb = new_value

    @ub.setter
    def ub(self, new_value):
        self.constrain.ub = new_value

    @A.setter
    def A(self, new_value):
        self.constrain.A = new_value

    def __str__(self):
        lb_size = 0
        ub_size = 0
        A_size = 0
        for lb_i, A_i, ub_i in zip(self.lb, self.A, self.ub):
            lb_size = max(lb_size, len(str(self.lb)))
            ub_size = max(ub_size, len(str(self.ub)))
            A_size = max(A_size, len(A_i))

        lb_str = "{<" + str(lb_size) + "}"
        ub_str = "{<" + str(ub_size) + "}"
        A_str = "{<" + str(A_size) + "}"
        ret_str = "\t".join([lb_str.format("lb"), "|",  A_str.format("A"),
                             "|", ub_str.format("ub")]) + "\n"
        for lb_i, A_i, ub_i in zip(self.lb, self.A, self.ub):
            ret_str += "\t".join([lb_str.format(lb_i), "|", 
                                  A_str.format(A_i), "|",
                                  ub_str.format(ub_i)]) + "\n"


class LinearConstrainDemographics(LinearConstrain):
    def __init__(self, A, lb, ub, engine, engine_args):
        self.engine = engine
        self.engine_args = engine_args
        self.original_lb = lb
        self.original_ub = ub
        super(LinearConstrainDemographics, self).__init__(A, lb, ub)

    def fits(self, x):
        theta = self.engine.get_theta(x, *self.engine_args)
        self.constrain.lb = self.original_lb / theta
        self.constrain.ub = self.original_ub / theta
        return super(LinearConstrainDemographics, self).fits(x)

    def try_to_transform(self, x):
        theta = self.engine.get_theta(x, *self.engine_args)
        lb = self.original_lb / theta
        ub = self.original_ub / theta
        lin_constr = LinearConstrain(self.constrain.A, lb, ub)
        print("!!!", ub)
        return lin_constr.try_to_transform(x)
