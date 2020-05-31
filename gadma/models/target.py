

class Target(Model):
    def __init__(self, data, model, engine_id):
        self._data = data
        self._model = model
        self.engine = get_engine(engine_id)
        super(Target, self).__init__()
        self._variables = self.model._variables
        self.is_fixed = self.model.is_fixed
        self.fixed_values = self.model.fixed_values

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model
        self.variables = self._model.variables
        self.is_fixed = self.model.is_fixed
        self.fixed_values = self.model.fixed_values
        self.engine.set_model(model)

    @property
    def data(self):
        return self._data

    @model.setter
    def data(self, new_data):
        self._data = new_data
        self.engine.set_data(data)

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, new_engine):
        self._engine = new_engine
        self._engine.set_data(self.data)
        self._engine.set_model(self.model)

    def evaluate(self, x, *args):
        return self.engine.evaluate(self, x, *args)


def increase_structure(target, X, Y, X_total=None, Y_total=None):
    # TODO
    new_target = copy.deepcopy(target)
    new_target.model.increase_structure()
    n_var = len(new_target.model.variables)
    new_X = copy.deepcopy(X)
    for i in range(len(new_X)):
        new_X[i] = WeightedMetaArray(np.zeros(n_var), dtype=object)
        if isinstance(X[i], WeightedMetaArray):
            new_X[i].weights = copy.copy(X[i].weights)
         
    return target, X, Y

def from_global_to_local_optimizer(target, X, Y, X_total=None, Y_total=None):
    x = X[0]
    y = Y[0]
    new_target = copy.deepcopy(target)
    new_target.model.fix_dynamics()
    return new_target, x, y

def get_best_n_from_optimizer(n, target, X, Y, X_total=None, Y_total=None):
    X, Y = sort_by_other_list(X, Y)
    return target, X[:n], Y[:n]

def get_last_n_from_total(n, target, X, Y, X_total, Y_total):
    new_X = copy.deepcopy(X)
    new_X.append(X_total[:-n])
    new_Y = copy.deepcopy(Y)
    new_Y.append(Y_total[:-n])
    return target, new_X, new_Y

def from_local_to_global_optimizer(target, X, Y, X_total=None, Y_total=None):
    new_target = copy.deepcopy(target)
    new_target.model.unfix_dynamics()
    return new_target, X, Y

def from_local_increase_structure_to_global(n, target, X, Y, X_total=None, Y_total=None):
    target, X0, Y0 = get_first_n_from_optimizer(1, from_local_to_global_optimizer(target, X, Y))
    target, X_init, Y_init = get_last_n_from_total(n, target, X0, Y0, X_total, Y_total)
    return increase_structure(target, X_init, Y_init)
