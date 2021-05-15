import unittest
from gadma import *
from gadma.engines.momi_engine import MomiEngine
from gadma.models.coalescent_demographic_model import CoalescentDemographicModel


class TestMomiEngine(unittest.TestCase):

    def get_absolute_variables_3pop(self):
        nu1F = PopulationSizeVariable('nu1F', units="physical")
        nu2F = PopulationSizeVariable('nu2F', units="physical")
        Tf = TimeVariable("Tf", units="physical")
        Tp = TimeVariable("Tp", units="physical")
        return nu1F, nu2F, Tf, Tp

    def get_variables_for_gut_2009(self):
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        return nu1F, nu2B, nu2F, Tp, T
