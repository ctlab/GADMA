import unittest
from gadma import *
from gadma.models import TreeDemographicModel
from gadma.models.variables_combinations import operation_creation, Exp


class TestTreeDemModel(unittest.TestCase):

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
    def get_genetic_variables_model1():
        N_a = PopulationSizeVariable('N_a', units="physical")
        nu1 = PopulationSizeVariable('nu1', units="genetic")
        nu2 = PopulationSizeVariable('nu2', units="genetic")
        nu2F = PopulationSizeVariable('nu2F', units="genetic")
        t1 = TimeVariable('t1', units="genetic")
        t2 = TimeVariable('t2', units="genetic")
        return N_a, nu1, nu2, nu2F, t1, t2

    def test_model1_translation(self):
        N_a, nu1, nu2, nu2F, t1, t2 = self.get_genetic_variables_model1()
        var2values = {
            'nu1': 0.4,
            'nu2F': 0.7,
            'nu2': 0.5,
            'N_a': 1e5,
            't1': 1,
            't2': 5
        }
        m = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        m.add_leaf(0, size_pop=nu1)
        m.add_leaf(1, size_pop=nu2F)
        m.change_pop_size(1, t=t1, size_pop=nu2)
        m.move_lineages(1, 0, t=t2, size_pop=N_a)

        self.assertEqual(m.number_of_populations(), 2)

        translated_model = EpochDemographicModel.create_from(m, var2values)
        em = EpochDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        em.add_split(0, [nu1, nu2])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1, nu2F])
        self.assertEqual(translated_model, em)

    @staticmethod
    def get_genetic_variables_model3():
        N_a = PopulationSizeVariable('N_a', units="physical")
        nu1B = PopulationSizeVariable('nu1B', units="genetic")
        nu1 = PopulationSizeVariable('nu1', units="genetic")
        nu1F = PopulationSizeVariable('nu1F', units="genetic")
        nu2B = PopulationSizeVariable('nu2B', units="genetic")
        nu2F = PopulationSizeVariable('nu2F', units="genetic")
        t1 = TimeVariable('t1', units="genetic")
        t2 = TimeVariable('t2', units="genetic")
        t3 = TimeVariable('t3', units="genetic")
        return N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3

    def test_model2_translation(self):
        N_a, nu1, nu2, nu2F, t1, t2 = self.get_genetic_variables_model1()
        var2values = {
            'nu1': 0.4,
            'nu2F': 0.7,
            'nu2': 0.5,
            'N_a': 1e5,
            't1': 1,
            't2': 5
        }
        m = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        m.add_leaf(0, size_pop=nu1)
        m.add_leaf(1, size_pop=nu2F)
        m.change_pop_size(1, t=t1, size_pop=nu2, dyn="Exp", g=0.9)
        m.move_lineages(1, 0, t=t2, size_pop=N_a)
        em = EpochDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        # g = log(N / N0) / t where N is final size (closer to nowdays) and
        # N0 is start size (the most ancient) t - time interval
        # THEN we have g, t and N and we want N0 = N / exp(gt)
        exp_op = operation_creation(
            Exp,
            operation_creation(
                Multiplication,
                0.9,
                operation_creation(
                    Subtraction,
                    t2,
                    t1
                )
            ))
        nu20 = operation_creation(
            Division,
            nu2,
            exp_op
        )
        # Construct our model
        em.add_split(0, [nu1, nu20])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2], dyn_args=['Sud', "Exp"])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1, nu2F])
        translated_model = m.translate_to(EpochDemographicModel, var2values)

    def test_model3_translation(self):
        N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3 = self.get_genetic_variables_model3()
        var2values = {
            'N_a': 1e5,
            'nu1B': 0.4,
            'nu1': 0.8,
            'nu1F': 0.1,
            'nu2B': 0.5,
            'nu2F': 0.2,
            't1': 2,
            't2': 4,
            't3': 5
        }
        m = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        m.add_leaf(0, size_pop=nu1F)
        m.add_leaf(1, size_pop=nu2F)
        m.change_pop_size(0, t=t1, size_pop=nu1)
        m.change_pop_size(0, t=t2, size_pop=nu1B)
        m.change_pop_size(1, t=t2, size_pop=nu2B)
        m.move_lineages(1, 0, t=t3, size_pop=N_a)
        translated_model = m.translate_to(EpochDemographicModel, var2values)
        em = EpochDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        em.add_split(0, [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t3, t2), [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2F])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1F, nu2F])
        self.assertEqual(translated_model, em)

    def test_model4_translation(self):
        N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3 = self.get_genetic_variables_model3()
        var2values = {
            'N_a': 1e5,
            'nu1B': 0.4,
            'nu1': 0.8,
            'nu1F': 0.1,
            'nu2B': 0.2,
            'nu2F': 0.5,
            't1': 2,
            't2': 4,
            't3': 5
        }
        m = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        m.add_leaf(0, size_pop=nu1F)
        m.add_leaf(1, size_pop=nu2F)
        m.change_pop_size(0, t=t1, size_pop=nu1)
        m.change_pop_size(0, t=t2, size_pop=nu1B)
        m.change_pop_size(1, t=t1, size_pop=nu2B)
        m.move_lineages(1, 0, t=t3, size_pop=N_a)
        translated_model = m.translate_to(EpochDemographicModel, var2values)
        em = EpochDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        em.add_split(0, [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t3, t2), [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2B])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1F, nu2F])
        self.assertEqual(translated_model, em)

    def test_model5_translation(self):
        N_a, nu1B, nu1, nu1F, nu2B, nu2F, t1, t2, t3 = self.get_genetic_variables_model3()
        t4 = TimeVariable('t4', units="genetic")
        var2values = {
            'N_a': 1e5,
            'nu1B': 0.4,
            'nu1': 0.8,
            'nu1F': 0.1,
            'nu2B': 0.5,
            'nu2F': 0.2,
            't1': 2,
            't2': 3,
            't3': 4,
            't4': 5
        }
        m = TreeDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        m.add_leaf(0, size_pop=nu1F)
        m.add_leaf(1, size_pop=nu2F)
        m.change_pop_size(0, t=t1, size_pop=nu1)
        m.change_pop_size(0, t=t3, size_pop=nu1B)
        m.change_pop_size(1, t=t2, size_pop=nu2B)
        m.move_lineages(1, 0, t=t4, size_pop=N_a)
        translated_model = m.translate_to(EpochDemographicModel, var2values)
        em = EpochDemographicModel(gen_time=29, mutation_rate=1.25e-8)
        em.add_split(0, [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t4, t3), [nu1B, nu2B])
        em.add_epoch(operation_creation(Subtraction, t3, t2), [nu1, nu2B])
        em.add_epoch(operation_creation(Subtraction, t2, t1), [nu1, nu2F])
        em.add_epoch(operation_creation(Subtraction, t1, 0), [nu1F, nu2F])
        self.assertEqual(2, translated_model.number_of_populations())
        self.assertEqual(translated_model, em)

