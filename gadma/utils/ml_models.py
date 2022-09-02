import os
import numpy as np

from .variables import DynamicVariable
from .sklearn_and_tf_models import _struct_dict, _model_2_1
from .sklearn_and_tf_models import SklearnRFIndependent, SklearnRFDependent
from .sklearn_and_tf_models import SklearnRFMultiOutput, KerasModelIndependent


SAMPLE_SIZE_PER_POP = 5
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_data(data):
    """
    Takes AFS data and transforms it so it will fit ML models.

    :param data: data in the format of dadi or moments spectrum
    """
    data = data.project([SAMPLE_SIZE_PER_POP, SAMPLE_SIZE_PER_POP]).fold()
    data = np.array(data).flatten()
    data = np.nan_to_num(data, 0)
    data /= np.sum(data)
    return np.array(data)


def transform_y_pred_to_dem_model(params_pred, dem_model):
    """
    Transforms predicted parameters so that they will be parameters
    for given demographic model. It could have different settings
    than default model so we should change predicted values
    according to them.

    :param params_pred: Predicted parameters.
    :param dem_model: Demographic model for which we want parameters.
    """
    from ..models import StructureDemographicModel
    # Create the structure model that was used for ML model training
    orig_dem_model = StructureDemographicModel(
        initial_structure=dem_model.get_structure(),
        final_structure=dem_model.get_structure(),
        **_struct_dict,
    )
    new_params_pred = dem_model.transform_values_from_other_model(
        model=orig_dem_model,
        x=params_pred
    )
    return new_params_pred


def dynamics2classes(y, back=False):
    """
    Change strings of '0', '1', '2' to classes 'Sud', 'Lin', 'Exp' or back.

    :param y: List of sets of predicted parameters.
    :param back: If True then classes are transformed to int strings.
    """
    _map = [("Sud", '0'), ("Lin", '1'), ("Exp", '2')]
    dyn2cls = {dyn: cls for dyn, cls in _map}
    cls2dyn = {cls: dyn for dyn, cls in _map}
    y_fixed = []
    for _y in y:
        _y = np.array(_y)
        if not back:
            transf_dict = dyn2cls
        else:
            transf_dict = cls2dyn
        for key in transf_dict:
            _y[_y == key] = transf_dict[key]
        y_fixed.append(_y)
    return y_fixed


class MLModel(object):
    """
    Abstract class to keep ML models in GADMA for predictions.
    """
    id = None

    def __init__(self):
        self.model = self._load()

    def _load(self):
        """
        Loads self from file.
        """
        raise NotImplementedError

    def _predict(self, dem_model, data):
        raise NotImplementedError

    def predict_for_structure_model(self, dem_model, data):
        """
        Predicts values for model parameters according to the data.
        """
        struct = dem_model.get_structure()
        assert len(struct) == 2, struct
        assert (struct == [1, 1] or struct == [2, 1])

        params_pred = self._predict(
            dem_model=dem_model,
            data=prepare_data(data),
        )
        return transform_y_pred_to_dem_model(
            params_pred=params_pred,
            dem_model=dem_model
        )


class RFModel(MLModel):
    """
    Abstract class for Random forest in GADMA.
    """
    id = None
    model_cls = None

    def _load(self):
        model = self.model_cls()
        model.load(os.path.join(CURRENT_DIR, "saved_ml_models", f"{self.id}"))
        return model

    def _predict(self, dem_model, data):
        struct = dem_model.get_structure()
        if struct == [2, 1]:
            pred = self.model.predict([data])
        else:  # struct == [1, 1]
            pred = self.model.predict_for_1_1_structure([data])
        pred = np.array(pred, dtype=object)
        params = dynamics2classes(pred, back=True)[0]
        return params


class RFIndependent(RFModel):
    """
    Class for Random forest with independent estimators.
    """
    id = "RandomForestIndependent"
    model_cls = SklearnRFIndependent


class RFDependent(RFModel):
    """
    Class for Random forest with dependent estimators.
    """
    id = "RandomForestDependent"
    model_cls = SklearnRFDependent


class RFMultiOutput(RFModel):
    """
    Class for Random forest with multi output.
    """
    id = "RandomForestMultiOutput"
    model_cls = SklearnRFMultiOutput


def convert_cnn_output_to_rf_output(cnn_output):
    """
    Converts output of CNN to the output of Random Forest.
    """
    rf_output = []
    for i in range(len(cnn_output[0])):
        out = []
        for j, var in enumerate(_model_2_1.variables):
            if isinstance(var, DynamicVariable):
                argmax = max(list(range(3)), key=lambda t: cnn_output[j][i][t])
                out.append(str(argmax))
            else:
                out.append(cnn_output[j][i][0])
        rf_output.append(np.array(out, dtype=object))
    return rf_output


class CNNModel(MLModel):
    """
    Class for Convolutional Neural Network in GADMA.
    """
    id = "CNN"

    def _load(self):
        model = KerasModelIndependent().assemble_full_model(
            width=SAMPLE_SIZE_PER_POP + 1,
            height=SAMPLE_SIZE_PER_POP + 1,
        )
        filename = os.path.join(
            CURRENT_DIR,
            "saved_ml_models",
            "model_cnn_weights.h5"
        )
        model.load_weights(filename)
        return model

    def _predict(self, dem_model, data):
        struct = dem_model.get_structure()

        pred = self.model.predict(np.array([np.reshape(data, (6, 6, 1))]))
        pred = convert_cnn_output_to_rf_output(pred)
        params = dynamics2classes(pred, back=True)[0]

        if struct == [1, 1]:
            params = params[3:]
        return params


id2class = {
    RFIndependent.id: RFIndependent,
    RFDependent.id: RFDependent,
    RFMultiOutput.id: RFMultiOutput,
    CNNModel.id: CNNModel
}
