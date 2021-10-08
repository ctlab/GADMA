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
    :param mutation_rate: Mutation rate. See
                          :class:`gadma.models.DemographicModel` for more
                          information.
    :param recombination_rate: Recombination rate.
    """
    def __init__(self, function, variables,
                 gen_time=None, theta0=None,
                 mutation_rate=None, recombination_rate=None,
                 fixed_anc_size=None, has_anc_size=False):
        self.function = function
        self.fixed_anc_size = fixed_anc_size
        super(CustomDemographicModel, self).__init__(
            gen_time=gen_time,
            theta0=theta0,
            mutation_rate=mutation_rate,
            recombination_rate=recombination_rate,
            Nref=None,
            has_anc_size=has_anc_size
        )
        if variables is None:
            variables = VariablePool()
        self.variables = VariablePool(variables)

    def translate_values(self, units, values, Nanc=None,
                         time_in_generations=False, rescale_back=False):
        return values
