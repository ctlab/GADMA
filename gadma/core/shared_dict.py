from ..utils import WeightedMetaArray, is_pickleable
from ..engines import Engine
import copy
from multiprocessing import Manager
from functools import partial
from collections import OrderedDict
import numpy as np
import warnings


class SharedDict(object):
    """
    Wrapper class on ``multiprocessing.Manager.dict`` that could be used for
    multiprocessing applications. Is used as shared memory of processes that
    are run in parallel for GADMA.

    All models in this dict can be divided in several processes and in several
    groups. When new model is added it has its own process and group. When
    dict should return some models it sort them by some value that is defined
    by `key` function and could be specific for each group.

    :param multiprocessing: If False than usual dict will be used inside.
    """
    def __init__(self, multiprocessing=True):
        if multiprocessing:
            self.dict = Manager().dict()
        else:
            self.dict = dict()

    def default_key(self, group):
        """
        Returns function for `key` in sort. Sorted elements will be compared by
        the value of `default_key(element)`.

        :param group: Name of group of models.
        """
        return None

    @staticmethod
    def get_value(model, key):
        """
        Returns value of `key` from model.

        :param model: Model.
        :param key: Key function.
        """
        if key is None:
            return model
        return key(model)

    def _put_new_model_for_process(self, process, group, model, key=None):
        if key is None:
            key = self.default_key(group)
        copy_dict = dict(self.dict)
        try:
            process_dict = OrderedDict(copy_dict[process])
        except KeyError:
            process_dict = OrderedDict()
        process_dict[group] = copy.deepcopy(model)
        try:
            self.dict[process] = process_dict
        except TypeError:
            try:
                self.dict[process] = process_dict
                warnings.warn(f"First attempt of connection failed but the "
                              f"second worked (run {process})")
            except TypeError:
                warnings.warn(f"Both attempt of connection failed for run "
                              f"{process}. Maybe this output is not updated "
                              f"for this run.")

    def update_best_model_for_process(self, process, group, model, key=None):
        """
        Updates best model for process. Models are compared by the value of
        `key` function. So if new model has greater value of `key` function
        then it is saved in the dictionary as best for this process in this
        group.

        :param process: Name of process.
        :param group: Name of model group.
        :param model: Model.
        :param key: `key` function, if None then :meth:`default_key` is used.
        """
        if key is None:
            key = self.default_key(group)
        new_value = self.get_value(model, key)

        copy_dict = dict(self.dict)
        try:
            process_dict = OrderedDict(copy_dict[process])
        except KeyError:
            process_dict = OrderedDict()

        if group in process_dict:
            old_model = self.get_best_model_for_process_in_group(process,
                                                                 group)
            old_value = self.get_value(old_model, key)
        if (group not in process_dict or old_value is None or
                np.allclose(new_value, old_value) or new_value >= old_value):
            self._put_new_model_for_process(process, group, model, key)
            return True
        return False

    def add_model_for_process(self, process, group, model, key=None):
        """
        Adds additional model for process. After this method dict will have
        list of models saved under (process, group).

        :param process: Name of process.
        :param group: Name of model group.
        :param model: Model.
        :param key: `key` function, if None then :meth:`default_key` is used.
        """
        if key is None:
            key = self.default_key(group)
        copy_dict = dict(self.dict)
        try:
            process_dict = OrderedDict(copy_dict[process])
        except KeyError:
            process_dict = OrderedDict()
        if group in process_dict:
            models = self.get_models_for_process_in_group(process, group)
        else:
            models = []
        models.append(model)
        process_dict[group] = copy.deepcopy(models)
        self.dict[process] = process_dict

    def get_models_for_process_in_group(self, process, group, key=None):
        """
        Returns models for (process, group).
        """
        if key is None:
            key = self.default_key(group)
        try:
            process_dict = OrderedDict(self.dict[process])
        except KeyError:
            process_dict = OrderedDict()
        if group not in process_dict:
            return []
        if isinstance(process_dict[group], (list, np.ndarray)):
            return sorted(process_dict[group], key=key, reverse=True)
        return [process_dict[group]]

    def get_models_in_group(self, group, key=None):
        """
        Returns dict of sorted pairs (process, model) for group
        (across all processes).
        """
        if key is None:
            key = self.default_key(group)
        models = []
        for process in self.dict.keys():
            process_models = self.get_models_for_process_in_group(process,
                                                                  group,
                                                                  key=key)
            if len(process_models) > 0:
                for model in process_models:
                    models.append([process, model])
            models = sorted(models, key=lambda x: key(x[1]), reverse=True)
        return models

    def get_best_model_in_group(self, group, key=None):
        """
        Returns best model for group.
        """
        if key is None:
            key = self.default_key(group)
        models = self.get_models_in_group(group, key)
        if len(models) == 0:
            return None
        return models[0][1]

    def get_best_model_for_process_in_group(self, process, group, key=None):
        """
        Returns best model for process, group.
        """
        if key is None:
            key = self.default_key(group)
        models = self.get_models_for_process_in_group(process, group, key)
        if len(models) == 0:
            return None
        return models[0]

    def get_available_groups(self):
        """
        Returns all available groups across processes.
        """
        names = []
        copy_dict = dict(self.dict)
        for process in copy_dict.keys():
            for group in copy_dict[process]:
                if group not in names:
                    names.append(group)
        return names


class SharedDictForCoreRun(SharedDict):
    """
    Class for shared dict in :class:`gadma.core.core_run.CoreRun`.

    Process is name of the process or index of CoreRun. Group is name of
    fitness function: log-likelihood, AIC, CLAIC. Model is tuple of
    demographic model, engine and fitness for this engine.
    """
    def default_key(self, group):
        """
        For not `log-likelihood` groups sort should be reversed so `key`
        function multiplies fitness by -1. Also fitness could be dict of
        several values of several groups (group of `log-likelihood` but
        fitness has also value of AIC or CLAIC). So this function extracts
        correct fitness.

        :param group: Name of fitness function (log-likelihood, AIC, CLAIC).
        """
        return partial(self._key, group)

    def construct_model(self, group, engine, x, y):
        """
        Constructs model for group, engine and x, y. Model is a tuple of
        (engine, x, fitness), where engine contain demographic model and
        fitness is dict with available fitnesses (logLL, AIC, CLAIC).

        :param group: Name of fitness (log-likelihood, AIC, CLAIC)
        :param engine: Engine with dem. model and data.
        :param x: Values of dem. model parameters.
        :param y: Value of fitness defined by `group`.

        :note: Model from engine could be lost if it is unpickleable
        """
        if not isinstance(y, dict):
            y = OrderedDict({group: y})
        if isinstance(engine, Engine) and not is_pickleable(engine.model):
            engine = copy.deepcopy(engine)
            super(Engine, engine).__setattr__("_model", None)
        # print(type(x), x)
        if isinstance(x, WeightedMetaArray):
            return (engine, (x, x.metadata), y)
        return (engine, x, y)

    def _extract_model(self, model):
        """
        Extract engine, x, y from the model in shared dict. Reverse function
        to :meth:`construct_model`.
        """
        engine, x_with_metadata, y = model
        if isinstance(x_with_metadata, tuple):
            x = WeightedMetaArray(x_with_metadata[0])
            x.metadata = x_with_metadata[1]
        else:
            x = x_with_metadata
        return engine, x, y

    def _key(self, group, model):
        """
        See :meth:`default_key`
        """
        sign = -1
        if group == 'log-likelihood':
            sign = 1
        _1, _2, y = self.construct_model(group, *model)
        if isinstance(y[group], tuple):
            ff = y[group][0]
        else:
            ff = y[group]
        if ff is None:
            return sign * np.inf
        return sign * ff

    def _put_new_model_for_process(self, process, group, model, key=None):
        if isinstance(model, tuple):
            engine, x, y = model
        model = self.construct_model(group, engine, x, y)
        r = super(SharedDictForCoreRun, self)._put_new_model_for_process(
            process, group, model)
        return r

    def update_best_model_for_process(self, process, group, engine, x, y):
        # print(x)
        model = self.construct_model(group, engine, x, y)
        r = super(SharedDictForCoreRun, self).update_best_model_for_process(
            process, group, model)
        return r

    def add_model_for_process(self, process, group, engine, x, y):
        model = self.construct_model(group, engine, x, y)
        r = super(SharedDictForCoreRun, self).add_model_for_process(
            process, group, model)
        return r

    def get_models_in_group(self, group, key=None, align_y_dict=False):
        """
        Returns models for specified group.

        :param group: Name of fitness (log-likelihood, AIC, CLAIC).
        :param key: `Key` function.
        :param align_y_dict: If True then all fitnesses are dict with keys
                             across all available groups and None's if value
                             for some group is not available.
        """
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
