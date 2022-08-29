from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import os
import joblib

from .variables import DynamicVariable
from ..models import StructureDemographicModel
from .variables import DynamicVariable


_struct_dict = {
    "has_migs": True,
    "has_sels": False,
    "has_dom": False,
    "has_dyns": True,
    "sym_migs": False,
    "frac_split": True,
    "migs_mask": None,
}

_model_2_1 = StructureDemographicModel(
    initial_structure=[2, 1],
    final_structure=[2, 1],
    **_struct_dict
)


# 1. Model Random Forest with independent estimators for parameters
class SklearnRFIndependent(object):
    def __init__(self):
        # We create our models
        # We have RF regressors for continous variables and
        # classifiers for dynamics
        self.models = {}
        for var in _model_2_1.variables:
            if not isinstance(var, DynamicVariable):
                self.models[var.name] = RandomForestRegressor()
            else:
                self.models[var.name] = RandomForestClassifier()

    def fit(self, X, y):
        name2ind = {var.name: i for i, var in enumerate(_model_2_1.variables)}
        for var in _model_2_1.variables:
            if not isinstance(var, DynamicVariable):
                self.models[var.name].fit(X, y[:, name2ind[var.name]])
            else:
                y = np.array(y[:, name2ind[var.name]], dtype=str)
                self.models[var.name].fit(X, y)

    def predict(self, X):
        y_pred = {}
        for var in _model_2_1.variables:
            y_pred[var.name] = self.models[var.name].predict(X)

        final_y_pred = []
        for _i in range(len(y_pred["nu11"])):
            _y = []
            for var in _model_2_1.variables:
                _y.append(y_pred[var.name][_i])
            final_y_pred.append(_y)
        return final_y_pred

    def predict_for_1_1_structure(self, X):
        y_pred = np.array(self.predict(X), dtype=object)
        return y_pred[:, 3:]

    def dump(self, dirname):
        for key in self.models:
            filename = os.path.join(dirname, f"{key}.joblib")
            joblib.dump(self.models[key], filename, compress=9)

    def load(self, dirname):
        for key in self.models:
            filename = os.path.join(dirname, f"{key}.joblib")
            self.models[key] = joblib.load(filename)


# 2. Model Random Forest with dependent estimators for parameters
class SklearnRFDependent(object):
    def __init__(self):
        # We create our models
        # We have RF regressors for continous variables and
        # classifiers for dynamics
        self.models = {}
        for var in _model_2_1.variables:
            if not isinstance(var, DynamicVariable):
                self.models[var.name] = RandomForestRegressor()
            else:
                self.models[var.name] = RandomForestClassifier()

    def _add_values(self, X, y, var_names):
        name2ind = {var.name: i
                    for i, var in enumerate(_model_2_1.variables)}
        new_X = []
        for _x, _y in zip(X, y):
            new_X.append(_x.tolist())
            for name in var_names:
                new_X[-1].append(_y[name2ind[name]])
        return np.array(new_X, dtype=object)

    def fit(self, X, y):
        # 0) get indices for var names
        name2ind = {var.name: i
                    for i, var in enumerate(_model_2_1.variables)}
        # 1) RF for nu11, t1 and dyn11 are independent
        self.models["nu11"].fit(X, y[:, name2ind["nu11"]])
        self.models["t1"].fit(X, y[:, name2ind["t1"]])
        self.models["dyn11"].fit(
            X,
            np.array(y[:, name2ind["dyn11"]], dtype=str)
        )

        # 2) Split fraction, times, sizes and dynamics after split depend
        # on those before split
        X_with_prev_values_1 = self._add_values(
            X=X, y=y, var_names=["nu11", "t1", "dyn11"])
        self.models["s1"].fit(X_with_prev_values_1, y[:, name2ind["s1"]])
        self.models["nu21"].fit(X_with_prev_values_1, y[:, name2ind["nu21"]])
        self.models["nu22"].fit(X_with_prev_values_1, y[:, name2ind["nu22"]])
        self.models["t2"].fit(X_with_prev_values_1, y[:, name2ind["t2"]])
        self.models["dyn21"].fit(
            X_with_prev_values_1,
            np.array(y[:, name2ind["dyn21"]], dtype=str)
        )
        self.models["dyn22"].fit(
            X_with_prev_values_1,
            np.array(y[:, name2ind["dyn22"]], dtype=str)
        )

        # 3) The migrations depends on time, sizes and dynamics
        X_with_prev_values_2 = self._add_values(
            X=X_with_prev_values_1,
            y=y,
            var_names=["s1", "nu21", "nu22", "t2", "dyn21", "dyn22"],
        )
        self.models["m2_21"].fit(X_with_prev_values_2, y[:, name2ind["m2_21"]])
        self.models["m2_12"].fit(X_with_prev_values_2, y[:, name2ind["m2_12"]])

    def predict(self, X):
        # 1) predict nu11, t1, d11
        y_pred = {}
        y_pred["nu11"] = self.models["nu11"].predict(X)
        y_pred["t1"] = self.models["t1"].predict(X)
        y_pred["dyn11"] = self.models["dyn11"].predict(X)

        # 2)
        new_X = []
        for i, x in enumerate(X):
            new_X.append(x.tolist())
            new_X[-1].extend(
                [y_pred["nu11"][i], y_pred["t1"][i], y_pred["dyn11"][i]]
            )
        new_X = np.array(new_X)
        y_pred["s1"] = self.models["s1"].predict(new_X)
        y_pred["nu21"] = self.models["nu21"].predict(new_X)
        y_pred["nu22"] = self.models["nu22"].predict(new_X)
        y_pred["t2"] = self.models["t2"].predict(new_X)
        y_pred["dyn21"] = self.models["dyn21"].predict(new_X)
        y_pred["dyn22"] = self.models["dyn22"].predict(new_X)

        # 3)
        new_new_X = []
        for i, x in enumerate(new_X):
            new_new_X.append(x.tolist())
            new_new_X[-1].extend([
                y_pred["s1"][i], y_pred["nu21"][i], y_pred["nu22"][i],
                y_pred["t2"][i], y_pred["dyn21"][i], y_pred["dyn22"][i]]
            )

        y_pred["m2_21"] = self.models["m2_21"].predict(new_new_X)
        y_pred["m2_12"] = self.models["m2_12"].predict(new_new_X)

        final_y_pred = []
        for _i in range(len(y_pred["nu11"])):
            _y = []
            for var in _model_2_1.variables:
                _y.append(y_pred[var.name][_i])
            final_y_pred.append(_y)
        return final_y_pred

    def predict_for_1_1_structure(self, X):
        # 1) predict nu11, t1, d11
        y_pred = {}
        y_pred["nu11"] = [1.0 for _ in X]
        y_pred["t1"] = [_model_2_1.variables[0].resample() for _ in X]
        y_pred["dyn11"] = ["0" for _ in X]

        # 2)
        new_X = []
        for i, x in enumerate(X):
            new_X.append(x.tolist())
            new_X[-1].extend(
                [y_pred["nu11"][i], y_pred["t1"][i], y_pred["dyn11"][i]]
            )
        new_X = np.array(new_X, dtype=object)
        y_pred["s1"] = self.models["s1"].predict(new_X)
        y_pred["nu21"] = self.models["nu21"].predict(new_X)
        y_pred["nu22"] = self.models["nu22"].predict(new_X)
        y_pred["t2"] = self.models["t2"].predict(new_X)
        y_pred["dyn21"] = self.models["dyn21"].predict(new_X)
        y_pred["dyn22"] = self.models["dyn22"].predict(new_X)

        # 3)
        new_new_X = []
        for i, x in enumerate(new_X):
            new_new_X.append(x.tolist())
            new_new_X[-1].extend([
                y_pred["s1"][i], y_pred["nu21"][i], y_pred["nu22"][i],
                y_pred["t2"][i], y_pred["dyn21"][i], y_pred["dyn22"][i]]
            )

        y_pred["m2_21"] = self.models["m2_21"].predict(new_new_X)
        y_pred["m2_12"] = self.models["m2_12"].predict(new_new_X)

        final_y_pred = []
        for _i in range(len(y_pred["nu11"])):
            _y = []
            for var in _model_2_1.variables[3:]:
                _y.append(y_pred[var.name][_i])
            final_y_pred.append(_y)
        return final_y_pred

    def dump(self, dirname):
        for key in self.models:
            filename = os.path.join(dirname, f"{key}.joblib")
            joblib.dump(self.models[key], filename, compress=9)

    def load(self, dirname):
        for key in self.models:
            filename = os.path.join(dirname, f"{key}.joblib")
            self.models[key] = joblib.load(filename)


# 3. Model Random Forest with multioutput
class SklearnRFMultiOutput(object):
    def __init__(self):
        # We create our models
        # We have RF regressor for continous variables and
        # classifier for dynamics
        self.models = [
            RandomForestRegressor(),
            RandomForestClassifier(),
        ]

    def fit(self, X, y):
        is_cont = np.array([not isinstance(var, DynamicVariable)
                            for var in _model_2_1.variables])
        is_disc = np.array([not cont for cont in is_cont])

        y_cont = [np.array(_y)[is_cont] for _y in y]
        y_disc = [np.array(_y)[is_disc] for _y in y]

        self.models[0].fit(X, y_cont)
        self.models[1].fit(X, y_disc)

    def predict(self, X):
        is_cont = np.array([not isinstance(var, DynamicVariable)
                            for var in _model_2_1.variables])
        is_disc = np.array([not cont for cont in is_cont])

        y_pred = np.zeros(
            shape=(len(X), len(_model_2_1.variables)), dtype=object
        )

        y_pred[:, is_cont] = self.models[0].predict(X)
        y_pred[:, is_disc] = self.models[1].predict(X)
        return y_pred

    def predict_for_1_1_structure(self, X):
        # 1) predict nu11, t1, d11
        y_pred = np.array(self.predict(X))
        return y_pred[:, 3:]

    def dump(self, dirname):
        for key in range(2):
            filename = os.path.join(dirname, f"{key}.joblib")
            joblib.dump(self.models[key], filename, compress=9)

    def load(self, dirname):
        for key in range(2):
            filename = os.path.join(dirname, f"{key}.joblib")
            self.models[key] = joblib.load(filename)


# 4. CNN with independent estimators for parameters
def get_custom_activation(bounds):
    def custom_activation(x):
        return (K.sigmoid(x) * (bounds[1] - bounds[0])) + bounds[0]
    return custom_activation


class KerasModelIndependent():
    def make_default_hidden_layers(self, inputs):
        filters = (16, 32, 64)
        for (i, f) in enumerate(filters):
            if i == 0:
                x = inputs
            x = Conv2D(f, (2, 2), padding="same", activation="relu")(x)
            x = BatchNormalization(axis=-1)(x)
            if i == (len(filters) - 1):
                x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
        return x

    def build_float_branch(self, inputs, name, bounds):
        x = self.make_default_hidden_layers(inputs)
        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(8, activation="relu")(x)
        # we have regression
        x = Dense(1)(x)
        x = Activation(get_custom_activation(bounds), name=name)(x)
        return x

    def build_discrete_branch(self, inputs, name):
        x = self.make_default_hidden_layers(inputs)
        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(8, activation="relu")(x)
        x = Dense(3)(x)
        x = Activation("softmax", name=name)(x)
        return x

    def assemble_full_model(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 1)
        inputs = Input(shape=input_shape)
        branches = []
        for var in _model_2_1.variables:
            if isinstance(var, DynamicVariable):
                branches.append(
                    self.build_discrete_branch(
                        inputs,
                        name=f"{var.name}_output"
                    )
                )
            else:
                branches.append(
                    self.build_float_branch(
                        inputs,
                        name=f"{var.name}_output",
                        bounds=var.domain
                    )
                )
        model = Model(
            inputs=inputs,
            outputs=branches,
            name="keras_net"
        )
        return model
