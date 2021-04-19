from ..utils import VariablePool
from .demographic_model import DemographicModel


class CustomDemographicModel(DemographicModel):
    """
    Demographic model that was defined by the user in some file. Engines
    should be able to deal with such kind of models.

    :param function: function that creates the demographic model or
                     something like that.
    :param variables: Variables of the function. I.e. model parameters.
    :param gen_time: Time of one generation.
    :param theta0: Mutation flux. See :class:`gadma.models.DemographicModel`
                   for more information.
    :param mu: Mutation rate. See :class:`gadma.models.DemographicModel`
               for more information.
    """
    def __init__(self, function, variables,
                 gen_time=None, theta0=None, mu=None):
        self.function = function
        super(CustomDemographicModel, self).__init__(gen_time=gen_time,
                                                     theta0=theta0,
                                                     mu=mu,
                                                     Nref=None,
                                                     has_anc_size=False)
        if variables is None:
            variables = VariablePool()
        self.variables = VariablePool(variables)
