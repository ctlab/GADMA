

class Target(Model):
    def __init__(self, data, model, engine_id):
        self._data = data
        self._model = model
        self.engine = get_engine(engine_id)
        super(Target, self).__init__()
        self.variables = self.model.variables

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model
        self.variables = self._model.variables
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

    def objective(self, x, *args):
        return self.engine.evaluate(self, x, *args)

    def increase_structure():
        self.model = self.model.increase_structure()
