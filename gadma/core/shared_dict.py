from ..utils import WeightedMetaArray
import copy
from multiprocessing import Manager
from functools import partial
from collections import OrderedDict
import numpy as np

class SharedDict(object):
    def __init__(self):
        self.dict = Manager().dict()

    def defaut_key(self, group):
        return None

    @staticmethod
    def get_value(model, key):
        if key is None:
            return model
        return key(model)

    def update_best_model_for_process(self, process, group, model, key=None):
        if key is None:
            key = self.defaut_key(group)
        new_value = self.get_value(model, key)
        if process in self.dict and group in self.dict[process]:
            old_model = self.get_best_model_in_group(process, group)
            old_value = self.get_value(old_model, key)
        if (process not in self.dict or group not in self.dict[process] or
                np.allclose(new_value, old_value) or new_value > old_value):
            if process not in self.dict.keys():
                new_dict = OrderedDict()
            else:
                new_dict = OrderedDict(self.dict[process])
            new_dict[group] = copy.deepcopy(model)
            self.dict[process] = new_dict
            return True
        return False

    def add_model_for_process(self, process, group, model, key=None):
        if key is None:
            key = self.defaut_key(group)
        if process in self.dict and group in self.dict[process]:
            models = self.get_models_in_group(process, group)
        else:
            models = []
        models = models.extend(model)
        if process not in self.dict.keys():
            new_dict = OrderedDict()
        else:
            new_dict = OrderedDict(self.dict[process])
        new_dict[group] = copy.deepcopy(models)
        self.dict[process] = new_dict

    def get_models_for_process_in_group(self, process, group, key=None):
        if key is None:
            key = self.defaut_key(group)
        if process not in self.dict or group not in self.dict[process]:
            return []
        if isinstance(self.dict[process][group], (list, np.ndarray)):
            return sorted(self.dict[process][group], key=key)
        return [self.dict[process][group]]

    def get_models_in_group(self, group, key=None):
        if key is None:
            key = self.defaut_key(group)
        models = []
        for process in self.dict.keys():
            process_models = self.get_models_for_process_in_group(process,
                                                                  group,
                                                                  key=key)
            if len(process_models) > 0:
                for model in process_models:
                    models.append([process, model])
            models = sorted(models, key=lambda x: key(x[1]))
        return models

    def get_best_model_in_group(self, process, group, key=None):
        if key is None:
            key = self.defaut_key(group)
        models = self.get_models_in_group(group, key)
        if len(models) == 0:
            return None
        return models[0][1]

class SharedDictForCoreRun(SharedDict):
    def defaut_key(self, group):
        return partial(self._key, group)

    def _model(self, group, engine, x, y):
        if not isinstance(y, dict):
            y = OrderedDict({group: y})
        #print(type(x), x)
        if isinstance(x, WeightedMetaArray):
            return (engine, (x, x.metadata), y)
        return (engine, x, y)

    def _extract_model(self, model):
        engine, x_with_metadata, y = model
        if isinstance(x_with_metadata, tuple):
            x = WeightedMetaArray(x_with_metadata[0])
            x.metadata = x_with_metadata[1]
        else:
            x = x_with_metadata
        return engine, x, y

    def _key(self, group, model):
        sign = -1
        if group == 'log-likelihood':
            sign = 1
        if isinstance(model[2], dict):
            return sign * model[2][group]
        return sign * model[2]

    def update_best_model_for_process(self, process, group, engine, x, y):
        #print(x)
        model = self._model(group, engine, x, y)
        r = super(SharedDictForCoreRun, self).update_best_model_for_process(
            process, group, model)
        return r

    def add_model_for_process(self, process, group, engine, x, y):
        model = self._model(group, engine, x, y)
        r = super(SharedDictForCoreRun, self).add_model_for_process(
            process, group, model)
        return r

    def get_available_groups(self):
        names = []
        for process in self.dict.keys():
            for group in self.dict[process]:
                if group not in names:
                    names.append(group)
        return names

    def get_models_for_group(self, group, key=None, align_y_dict=False):
        models = super(SharedDictForCoreRun, self).get_models_in_group(group,
                                                                       key)
        if not align_y_dict:
            return models
        y_dict_keys = list()
        for process, model in models:
            for k in model[2]:
                if k not in y_dict_keys:
                    y_dict_keys.append(k)
        for i, (process, model) in enumerate(models):
            for k in y_dict_keys:
                if k not in model[2]:
                    models[i][1][2][k] = None
                models[i][1] = self._extract_model(models[i][1])
        return models
