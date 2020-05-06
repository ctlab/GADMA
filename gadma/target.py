
class Target(object):
    """
    Class for target of optimization.

    :param model: model that will be used incide. By now it should be 
                  demographic model.
    :type model: :class:`gadma.DemographicModel`
    :param data: observed data that should be used to evaluate objective
                 function.
    :type data: :class:`gadma.DataHolder`
    :param engine: engine to evaluate objective function.
    :type engine: :class:`gadma.Engine`
    """
    def __init__(self, model, data, engine):
        self.model = model
        self.data = data
        self.engine = engine
        self.variables = self.model.variables
        self.domain = [x.get_domain() for x in self.variables]

    def objective_func(self, x):
        """
        Evaluates objective function from values x.
        """
        return self.engine.set_and_evaluate(x, self.model, self.data)
