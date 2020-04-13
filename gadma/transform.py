import numpy as np

class Transform(object):
    def f(self, x):
        raise NotImplementedError
    def finv(self, f):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError

class Logarithm(Transform):
    def f(self, x):
        return np.log(x)
    def finv(self, f):
        return np.exp(f)
    def __str__(self):
        return "Logarithm function"

class Exponent(Transform):
    def f(self, x):
        return np.exp(x)
    def finv(self, f):
        return np.log(f)
    def __str__(self):
        return "Exponent function"

