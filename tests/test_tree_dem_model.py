import unittest
import os
import numpy as np
import pytest
import gadma
from gadma import *
from gadma.models import TreeDemographicModel, LineageMovement
from gadma.models.variables_combinations import operation_creation, Exp

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data", "DATA")
YRI_CEU_DATA = os.path.join(DATA_PATH, "sfs", "YRI_CEU.fs")


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

    @staticmethod
    def get_structure_model_and_values(gen_time=None, has_dyns=False):
        """
        Model with structure (2, 1). Size of ancestral population is 10000,
        so its size is changed 3000 generations ago and split happens
        1000 generations ago.
        """
        model = StructureDemographicModel(
            initial_structure=[2, 1], final_structure=[2, 1],
            has_migs=False, has_sels=False, has_dom=False,
            has_dyns=has_dyns, sym_migs=False, frac_split=False,
            has_p_misid=False, has_anc_size=True, gen_time=gen_time,
            mutation_rate=1.25e-8,
        )
        values = {"Nanc": 10000, "t1": 0.1, "nu11": 0.5, "nu11_1": 0.3,
                  "nu11_2": 0.4, "t2": 0.05, "nu21": 0.7, "nu22": 0.6,
                  "dyn11": "Sud", "dyn21": "Exp", "dyn22": "Sud"}
        return model, values

    def test_split_time_in_translation_from_epoch_model(self):
        """
        Time of split should be the same no matter if size of population
        is changed during the split or not.
        """
        model, values = self.get_structure_model_and_values()
        # In the second case size of the first population is the same
        # before and after the split.
        for nu21 in [0.7, 0.5]:
            values["nu21"] = nu21
            x = [values[var.name] for var in model.variables]
            tree_model, _ = model.translate_to(TreeDemographicModel, x)
            phys_values = model.translate_values(units="physical", values=x,
                                                 time_in_generations=True)
            var2value = model.var2value(phys_values)

            def get_value(entity):
                return model.get_value_from_var2value(var2value, entity)

            splits = [event for event in tree_model.events
                      if isinstance(event, LineageMovement)]
            self.assertEqual(len(splits), 1)
            self.assertTrue(np.isclose(get_value(splits[0].t), 1000),
                            msg=f"Time of split is {get_value(splits[0].t)}"
                                f" instead of 1000 (nu21={nu21})")
            # size of the population after the merge
            self.assertTrue(np.isclose(get_value(splits[0].size_pop), 5000))
            # change of the size 3000 generations ago is still in the model
            all_times = [get_value(event.t) for event in tree_model.events]
            self.assertTrue(np.any(np.isclose(all_times, 3000)),
                            msg=f"Times of events: {all_times}")

    @pytest.mark.skipif(not gadma.momi_available, reason="No momi2")
    def test_momi_time_is_in_generations(self):
        """
        Time for one generation should not change likelihood: momi model is
        created with time in generations.
        """
        data_holder = SFSDataHolder(YRI_CEU_DATA, sequence_length=50818468)
        ll_values = []
        for gen_time in [None, 25]:
            model, values = self.get_structure_model_and_values(
                gen_time=gen_time, has_dyns=True
            )
            engine = get_engine("momi2")
            engine.set_data(data_holder)
            engine.set_model(model)
            x = [values[var.name] for var in model.variables]
            ll_values.append(engine.evaluate(x))

            # split is 1000 generations ago
            momi_model = engine.get_momi_model(x)
            split_time = momi_model.topology_events[0].t.x
            self.assertTrue(np.isclose(split_time, 1000),
                            msg=f"Time of split is {split_time} "
                                f"(gen_time={gen_time})")

            # generated code should give the same value
            code = engine.generate_code(x, None, nanc=None,
                                        gen_time=gen_time or 1,
                                        gen_time_units="years")
            # we do not want to draw picture
            code = code[:code.index("from matplotlib import pyplot")]
            namespace = {}
            exec(code, namespace)
            self.assertTrue(np.isclose(ll_values[-1], namespace["ll_model"]))
        self.assertTrue(np.isclose(ll_values[0], ll_values[1]),
                        msg=f"Likelihood depends on the generation time: "
                            f"{ll_values}")

        # Model with time in other units, it is used for drawing.
        # It is the same model, so likelihood is the same.
        momi_model_years = engine.get_momi_model(x, gen_time=25)
        split_time_years = momi_model_years.topology_events[0].t.x
        self.assertTrue(np.isclose(split_time_years, 25000))
        momi_model_years.set_data(engine.inner_data, length=50818468)
        self.assertTrue(np.isclose(ll_values[-1],
                                   momi_model_years.log_likelihood()))

