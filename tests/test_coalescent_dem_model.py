import unittest
from gadma import *
from gadma.engines.momi_engine import MomiEngine
from gadma.models.coalescent_demographic_model import CoalescentDemographicModel


class TestCoalescentDemModel(unittest.TestCase):

    def get_variables_for_gut_2009(self):
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        return nu1F, nu2B, nu2F, Tp, T

    def get_absolute_variables_3pop(self):
        nu1F = PopulationSizeVariable('nu1F', units="physical")
        nu2F = PopulationSizeVariable('nu2F', units="physical")
        Tf = TimeVariable("Tf", units="physical")
        Tp = TimeVariable("Tp", units="physical")
        return nu1F, nu2F, Tf, Tp

    def test_3pop_model_translation(self):
        nu1F, nu2F, Tf, Tp = self.get_absolute_variables_3pop()
        m = CoalescentDemographicModel(N_e=1e4, sequence_length=1e6, gen_time=29, mu=1.25e-8)
        m.add_leaf(0, size_pop=nu1F)
        m.add_leaf(1, size_pop=nu2F, g=5e-4)
        m.add_leaf(2, t=Tf)
        m.move_lineages(pop=0, pop_from=1, t=8.5e4, size_pop=1.2e4)
        m.move_lineages(pop=2, pop_from=0, t=Tp)
        var2values = {
            'nu1F': 14000,
            'nu2F': 7500,
            'Tf': 1e5,
            'Tp': 1e4
        }

        m.translate_into(EpochDemographicModel, var2values)


    def test_gut2009_translation(self):
        nu1F, nu2B, nu2F, Tp, T = self.get_variables_for_gut_2009()
        N_a = PopulationSizeVariable("N_a", units="physical")
        m = CoalescentDemographicModel(N_e=1.2e4,
                                       N_a=N_a,
                                       sequence_length=1e6,
                                       gen_time=29,
                                       mu=1.29e-8,
                                       rec_rate=0)
        # YRI - 0
        # CEU - 1
        m.add_leaf(pop=0, size_pop=Multiplication(N_a, nu1F))
        m.add_leaf(pop=1, size_pop=Multiplication(N_a, nu2F), g=np.log(1.764 / 0.0724) / (0.112 * 50 * 7300))
        time_move = Multiplication(Multiplication(2 * m.gen_time, N_a), T)
        time_change = Multiplication(Addition(Tp, T), Multiplication(2 * m.gen_time, N_a))
        m.move_lineages(pop_from=1,
                        pop=0,
                        t=time_move,
                        size_pop=Multiplication(nu1F, N_a))
        m.change_pop_size(pop=0,
                          size_pop=N_a,
                          g=0,
                          t=time_change)
        var2values = {
            'N_a': 7300,
            'nu1F': 1.880,
            'nu2F': 1.764,
            'T': 0.112,
            'Tp': 0.363
        }
        m.translate_into(EpochDemographicModel, var2values)
