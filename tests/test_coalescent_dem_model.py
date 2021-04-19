import unittest
from gadma import *
from gadma.engines.momi_engine import MomiEngine
from gadma.models.coalescent_demographic_model import CoalescentDemographicModel


class TestCoalescentDemModel(unittest.TestCase):

    @staticmethod
    def get_variables_for_gut_2009():
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        return nu1F, nu2B, nu2F, Tp, T

    @staticmethod
    def get_absolute_variables_3pop():
        nu1F = PopulationSizeVariable('nu1F', units="physical")
        nu2F = PopulationSizeVariable('nu2F', units="physical")
        Tf = TimeVariable("Tf", units="physical")
        Tp = TimeVariable("Tp", units="physical")
        return nu1F, nu2F, Tf, Tp

    @staticmethod
    def get_absolute_variables_model1():
        N_a = PopulationSizeVariable('N_a', units="physical")
        nu1 = PopulationSizeVariable('nu1', units="genetic")
        nu2 = PopulationSizeVariable('nu2', units="genetic")
        nu2F = PopulationSizeVariable('nu2F', units="genetic")
        t1 = TimeVariable('t1', units="genetic")
        t2 = TimeVariable('t2', units="genetic")
        return N_a, nu1, nu2, nu2F, t1, t2

    def test_model1_translation(self):
        N_a, nu1, nu2, nu2F, t1, t2 = self.get_absolute_variables_model1()
        var2values = {
            'nu1': 0.4,
            'nu2F': 0.7,
            'nu2': 0.5,
            'N_a': 1e5,
            't1': 1,
            't2': 5
        }
        m = CoalescentDemographicModel(N_e=1e4, sequence_length=1e6, gen_time=29, mu=1.25e-8)
        m.add_leaf(0, size_pop=nu1)
        m.add_leaf(1, size_pop=nu2F)
        m.change_pop_size(1, t=t1, size_pop=nu2)
        m.move_lineages(1, 0, t=t2, size_pop=N_a)
        translated_model = m.translate_into(EpochDemographicModel, var2values)
        em = EpochDemographicModel(gen_time=29, mu=1.25e-8)
        em.add_split(0, [nu1, nu2])
        em.add_epoch(Subtraction(t2, t1), [nu1, nu2])
        em.add_epoch(Subtraction(t1, 0), [nu1, nu2F])
        self.assertEqual(translated_model.as_custom_string(values=var2values),
                         em.as_custom_string(values=var2values))

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
