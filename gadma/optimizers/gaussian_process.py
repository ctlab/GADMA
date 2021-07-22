import numpy as np
import copy


class GaussianProcess(object):
    """
    Base class to keep Gaussian process for Bayesian optimization.
    """
    def __init__(self, gp_model):
        self.gp_model = gp_model

    def train(self, X, Y, optimize=True):
        raise NotImplementedError

    def get_noise(self):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def get_K(self):
        raise NotImplementedError

    def get_hypers(self):
        raise NotImplementedError


class GPyGaussianProcess(GaussianProcess):
    def _convert(self, X, Y=None):
        X = np.array(X, dtype=float)
        if Y is not None:
            return X, np.array(Y).reshape(len(Y), -1)
        return X

    def train(self, X, Y, optimize=True):
        X, Y = self._convert(X, Y)
        if self.gp_model.model is None:
            self.gp_model._create_model(X=X, Y=Y)
        else:
            self.gp_model.model.set_XY(X, Y)
        if optimize:
            self.gp_model.updateModel(X_all=X, Y_all=Y, X_new=X, Y_new=Y)

    def get_noise(self):
        return float(self.gp_model.model.Gaussian_noise.variance)

    def predict(self, X):
        X = self._convert(X)
        mu, sigma = self.gp_model.predict(X)
        return mu.reshape(mu.shape[0]), sigma.reshape(sigma.shape[0])

    def get_K(self):
        X = self.gp_model.model.X
        return self.gp_model.model.kern.K(X, X)

    def get_hypers(self):
        theta = []
        theta.extend(self.gp_model.model.kern.lengthscale)
        theta.append(self.get_noise())
        return theta


class SMACGaussianProcess(GaussianProcess):
    def __init__(self, gp_model):
        super(SMACGaussianProcess, self).__init__(gp_model=gp_model)
        self.gp_model.normalize_y = True

    def train(self, X, Y, optimize=True):
        self.gp_model._train(X, Y, do_optimize=optimize)

    def get_noise(self):
        return 0

    def predict(self, X):
        mu, var = self.gp_model.predict_marginalized_over_instances(
            np.array(X)
        )
        sigma = np.sqrt(var)
        return mu.reshape(mu.shape[0]), sigma.reshape(sigma.shape[0])

    def get_K(self):
        return self.gp_model.kernel(self.gp_model.gp.X_train_)

    def get_hypers(self):
        return np.exp(self.gp_model.hypers)
