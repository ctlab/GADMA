import unittest
from gadma import *
import itertools
import copy
import numpy as np
import pytest
import multiprocessing
from multiprocessing import Queue, Process

TEST_STRUCTURES = [(1,), (2,),
                   (1,1), (2,1), (1,2),
                   (1,1,1)]

BASE_TEST_STRUCTURES = [(1,), (2,), (2,1), (1, 1, 1)]

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.structure_demographic_model', lineno=77)
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.structure_demographic_model', lineno=81)


def func_in_separate_process(function, *args):
    multiprocessing.set_start_method('spawn', force=True)
    queue = Queue()
    er_queue = Queue()
    p = Process(target=function,
                args=(queue, er_queue, *args))
    p.start()
    p.join() # this blocks until the process terminates
    if not er_queue.empty():
        raise RuntimeError() from er_queue.get(0)
    if not queue.empty():
        return queue.get(0)


def run_dical2_eval(q, er_q, data_holder, models_pairs, vals_pairs):
    try:
        engine = get_engine("diCal2")
        engine.data = engine.read_data(data_holder)
        result_list = []
        for model_pair, vals_pair in zip(models_pairs, vals_pairs):
            model_1, model_2 = model_pair
            vals_1, vals_2 = vals_pair
            engine.model = model_1
            ll1 = engine.evaluate(vals_1)
            engine.model = model_2
            ll2 = engine.evaluate(vals_2)
            result_list.append([ll1, ll2])
        engine._stopJVM()
        q.put(result_list)
    except Exception as e:
        er_q.put(e)
        raise e


class TestModelStructure(unittest.TestCase):
    def _generate_mig_mask(self, structure, sym_migs):
        masks = []
        for npop, nint in enumerate(structure[1:]):
            npop += 2
            for _ in range(nint):
                mask = []
                for i in range(npop):
                    mask.append([])
                    for j in range(npop):
                        if i == j:
                            mask[-1].append(0)
                            continue
                        if sym_migs and j < i:
                            mask[-1].append(mask[j][i])
                            continue
                        mask[-1].append(np.random.choice([0, 1]))
                masks.append(mask)
        return masks

    def test_initialization(self):
        for structure in TEST_STRUCTURES:
            for create_migs, create_sels, create_dyns, sym_migs, fracs, inbr in\
                    list(itertools.product([False, True],repeat=6)):
                dm = StructureDemographicModel(structure, structure,
                                               has_migs=create_migs,
                                               has_sels=create_sels,
                                               has_dyns=create_dyns,
                                               sym_migs=sym_migs,
                                               frac_split=fracs,
                                               has_inbr=inbr)
                self.assertRaises(ValueError, dm.increase_structure)
                struct = dm.get_structure()
                ind = np.random.choice(range(len(struct)))
                struct[ind] += 2
                self.assertRaises(ValueError, dm.increase_structure, struct)
                dm.final_structure = struct
                self.assertRaises(ValueError, dm.increase_structure, struct)
                dm.final_structure = structure

                # for splits variables
                n_par = (1 + int(not fracs)) * (len(structure) - 1)
                for i, str_val in enumerate(structure):
                    if i == 0:
                        # for each interval (except first) there is time,
                        # size of population, selection and dynamic
                        n_par += (str_val - 1) * (2 + int(create_sels)\
                                 + int(create_dyns))
                    else:
                        # for other intervals there are also migrations
                        n_pop = i + 1
                        n_migs = int(create_migs) * (n_pop * (n_pop - 1))
                        if sym_migs:
                            n_migs /= 2
                        n_par += str_val * (n_pop * (1 + int(create_dyns)\
                                 + int(create_sels)) + n_migs + 1)
                n_par += int(inbr) * len(structure)
                msg = f"Parameters are not equal for dem model with structure "\
                      f"{structure} and create_migs ({create_migs}), "\
                      f"create_sels ({create_sels}), create_dyns ({create_dyns}), "\
                      f"sym_migs ({sym_migs}), fracs ({fracs}) {dm.variables}," \
                      f"inbr ({inbr})"
                self.assertEqual(len(dm.variables), n_par, msg=msg)

                if len(structure) > 1:
                    masks = self._generate_mig_mask(structure, sym_migs)
                    dm = StructureDemographicModel(structure, structure,
                                                   has_migs=create_migs,
                                                   has_sels=create_sels,
                                                   has_dyns=create_dyns,
                                                   sym_migs=sym_migs,
                                                   migs_mask=masks,
                                                   frac_split=fracs,
                                                   has_inbr=inbr)
                    self.assertRaises(ValueError, dm.increase_structure)
                    if sym_migs and create_migs:
                        masks[0][0][1] = 1
                        masks[0][1][0] = 0
                        self.assertRaises(
                            ValueError, StructureDemographicModel, structure, 
                            structure, has_migs=create_migs,
                            has_sels=create_sels, has_dyns=create_dyns,
                            sym_migs=sym_migs, migs_mask=masks,
                            frac_split=fracs, has_inbr=inbr)
                dm.events.append(1.0)
                self.assertRaises(ValueError, dm.get_structure)


    def test_migration_masks_failures(self):
        for structure in TEST_STRUCTURES:
            if len(structure) < 2:
                dm = StructureDemographicModel(structure, structure,
                                               has_migs=True,
                                               has_sels=True,
                                               has_dyns=True,
                                               sym_migs=False,
                                               migs_mask=[],
                                               frac_split=True,
                                               has_inbr=False)
                self.assertEqual(dm.migs_mask, None)
                continue
            final_structure = list(structure)
            final_structure[np.random.choice(range(len(final_structure)))] += 1

            masks = self._generate_mig_mask(structure, True)
            self.assertRaises(
                ValueError, StructureDemographicModel,
                structure, final_structure, has_migs=True,
                has_sels=True, has_dyns=True, sym_migs=False,
                migs_mask=masks, frac_split=True, has_inbr=False)

            masks = self._generate_mig_mask(structure, False)
            masks[-1] = np.zeros(shape=(5, 5))
            self.assertRaises(
                ValueError, StructureDemographicModel,
                structure, structure, has_migs=True,
                has_sels=True, has_dyns=True, sym_migs=False,
                migs_mask=masks, frac_split=True, has_inbr=False)

            masks = self._generate_mig_mask(structure, False)
            masks = masks[:-1]
            self.assertRaises(
                ValueError, StructureDemographicModel,
                structure, structure, has_migs=True,
                has_sels=True, has_dyns=True, sym_migs=False,
                migs_mask=masks, frac_split=True, has_inbr=False)

    @pytest.mark.timeout(0)
    def test_likelihood_after_increase(self):
        failed = 0
        for structure in BASE_TEST_STRUCTURES:
            models_pairs = []
            values_pairs = []
            messages = []
            for create_migs, create_sels, create_dyns, sym_migs, fracs, has_anc, inbr in\
                    list(itertools.product([False, True],repeat=7)):
                if not create_migs:
                    sym_migs = False
                def model_generator(structure):
                    return StructureDemographicModel(structure,
                                                     np.array(structure) + 1,
                                                     has_migs=create_migs,
                                                     has_sels=create_sels,
                                                     has_dyns=create_dyns,
                                                     sym_migs=sym_migs,
                                                     frac_split=fracs,
                                                     has_anc_size=has_anc,
                                                     has_inbr=inbr)
                dm = model_generator(structure)

                # for dical2
                dm.mu = 1.25e-8
                dm.Nref = 10000

                variables = dm.variables
                x = [var.resample() for var in variables]
                if has_anc:
                    x[0] = np.random.uniform(1000, 20000)
                bad_structure = list(structure)
                for i in range(len(bad_structure)):
                    if bad_structure[i] == 1:
                        continue
                    bad_structure[i] -= 1
                    self.assertRaises(ValueError, dm.increase_structure,
                                      bad_structure)

                check_ll = np.random.choice([True, False], p=[1/6, 5/6])

                for engine in all_engines():
                    #print(engine.id)
                    if (engine.id == "diCal2" and
                            (not has_anc or inbr or create_sels)):
                        continue
                    if engine.id == "diCal2":
                        original_x = list(x)
                        x = ["Exp" if el == "Lin" else el for el in x]
                    engine.set_model(dm)
                    if engine.id == 'dadi':
                        sizes = [8 for _ in range(len(structure))]
                        if len(structure) == 1:
                            sizes = [20]
                        args = ([4, 6, 8],)  # pts
                    else:
                        sizes = [4 for _ in range(len(structure))]
                        args = ()
                    if engine.can_simulate:
                        # simulate data
                        data = engine.simulate(x, sizes, *args)
                        engine.set_data(data)
                    else:
                        assert engine.id == "diCal2"
                        VCF_DATA = os.path.join(DATA_PATH, "DATA", "vcf",
                                                "small.vcf")
                        POPMAP = os.path.join(DATA_PATH, "DATA", "vcf",
                                              f"popmap_{len(structure)}pop")
                        REFERENCE = os.path.join(DATA_PATH, "DATA", "vcf",
                                                 "reference.fa")
                        data_holder = VCFDataHolder(vcf_file=VCF_DATA,
                                                    popmap_file=POPMAP,
                                                    reference_file=REFERENCE)

                    if check_ll and engine.can_simulate:
                        # get ll of data
                        try:
                            ll_true = engine.evaluate(x, *args)
                        except AttributeError:
                            assert engine.id == "dadi"
                        random_int = np.random.choice(range(len(structure)))

                    # increase structure
                    for i in range(len(structure)):
                        new_structure = list(copy.copy(structure))
                        new_structure[i] += 1
                        msg = f"Increase structure from {structure} to "\
                              f"{new_structure} for engine {engine.id}. "\
                              f"create_migs: {create_migs}, "\
                              f"create_sels: {create_sels}, "\
                              f"create_dyns: {create_dyns}, "\
                              f"sym_migs: {sym_migs}, "\
                              f"fracs: {fracs}, " \
                              f"has_anc: {has_anc}, "\
                              f"inbr: {inbr}"
                        #print(msg)
                        new_dm = copy.deepcopy(dm)
                        new_dm, new_X = new_dm.increase_structure(
                            new_structure, [x])
                        an_dm = copy.deepcopy(dm)
                        _, X_none = an_dm.increase_structure()
                        self.assertEqual(X_none, None)
                        engine.set_model(new_dm)
#                        print("!!!", dm.var2value(x), new_dm.var2value(new_X[0]))
                        if check_ll and random_int == i:
                            if engine.can_simulate:
                                try:
                                    new_ll = engine.evaluate(new_X[0], *args)
                                    if not has_anc:
                                        self.assertTrue(np.allclose(ll_true, new_ll),
                                                        msg=f"{ll_true} != {new_ll} : {msg}")
                                    else:
                                        self.assertTrue(abs(ll_true - new_ll) < 500,
                                                        msg=f"{ll_true} != {new_ll} : {msg}")

                                except AttributeError:
                                    assert engine.id == "dadi"
                                    failed += 1
                                except AssertionError as e:
                                    if not str(e).startswith("masked is not true"):
                                        raise e
                                    assert engine.id == "dadi"
                                    failed += 1
                            else:
                                assert engine.id == "diCal2"
                                models_pairs.append([copy.deepcopy(dm), copy.deepcopy(new_dm)])
                                values_pairs.append([x, new_X[0]])
                                messages.append(msg)
                    # we must return original value with Lin to x
                    if engine.id == "diCal2":
                        x = original_x

                dm.final_structure = dm.get_structure()
                self.assertRaises(ValueError, dm.increase_structure)

            # run all together for dical2
            engine = get_engine("diCal2")
            results = func_in_separate_process(run_dical2_eval, data_holder,
                                               models_pairs, values_pairs)
            for ll_pair, msg in zip(results, messages):
                ll1, ll2 = ll_pair
                if ll1 is None or ll2 is None:
                    continue
                self.assertTrue(np.isclose(ll1, ll2, rtol=1e-3),
                                msg=f"{ll1} != {ll2} : {msg}")
        self.assertTrue(failed <= 5)

    def test_fails(self):
        bad_struct = [[0], [0, 1], [1, 0]]

        bad_cur_init_final_structs = [([1, 1], [2, 1], [3, 1]),
                                      ([2, 2], [1, 1], [1, 2]),
                                      ([1, 2], [1, 1], [1, 1])]


        for create_migs, create_sels, create_dyns, sym_migs, fracs, inbr in\
                list(itertools.product([False, True],repeat=6)):
            def build_model(init_struct, final_struct):
                return StructureDemographicModel(init_struct, final_struct,
                                                 has_migs=create_migs,
                                                 has_sels=create_sels,
                                                 has_dyns=create_dyns,
                                                 sym_migs=sym_migs,
                                                 frac_split=fracs,
                                                 has_inbr=inbr)
            # bad strcutures
            for struct in bad_struct:
                self.assertRaises(ValueError, build_model, struct, struct)

            # bad final structure
            for struct in TEST_STRUCTURES:
                for i in range(len(struct)):
                    final_struct = list(struct)
                    final_struct[i] -= 1
                    self.assertRaises(ValueError, build_model,
                                      struct, final_struct)

            # bigger or lesser structure in from_structure
            for cur_str, init_str, final_str in bad_cur_init_final_structs:
                model = build_model(init_str, final_str)
                self.assertRaises(ValueError, model.from_structure, cur_str)
                self.assertRaises(ValueError, model.from_structure, cur_str)

            # not possible to increase structure
            for _, init_str, final_str in bad_cur_init_final_structs:
                model = build_model(init_str, final_str)
                model.from_structure(final_str)
                self.assertRaises(ValueError, model.increase_structure)
            init_str, final_str = [2, 3], [3, 4]
            model = build_model(init_str, final_str)
            self.assertRaises(ValueError, model.increase_structure, [1, 3])
            self.assertRaises(ValueError, model.increase_structure, [2, 2])

    def test_transform(self):
        for structure in TEST_STRUCTURES:
            for base_migs, base_sels, base_dyns, base_symms, base_fracs, base_inbr in\
                    list(itertools.product([False, True],repeat=6)):
                base_mig_masks = [None, self._generate_mig_mask(structure,
                                                                base_symms)]
                if len(structure) == 1 or not base_migs:
                    base_mig_masks = [None]
                for mask in base_mig_masks:
                    dm = StructureDemographicModel(structure, structure,
                                                   has_migs=base_migs,
                                                   has_sels=base_sels,
                                                   has_dyns=base_dyns,
                                                   sym_migs=base_symms,
                                                   migs_mask=mask,
                                                   frac_split=base_fracs,
                                                   has_inbr=base_inbr)
                    for new_migs, new_sels, new_dyns, new_symms, new_fracs, new_inbr in\
                        list(itertools.product([False, True],repeat=6)):
                        new_mig_masks = [None, self._generate_mig_mask(
                            structure, new_symms)]
                        if len(structure) == 1 or not new_migs:
                            new_mig_masks = [None]
                        for new_mask in new_mig_masks:
                            new = StructureDemographicModel(structure,
                                                            structure,
                                                            has_migs=new_migs,
                                                            has_sels=new_sels,
                                                            has_dyns=new_dyns,
                                                            sym_migs=new_symms,
                                                            migs_mask=new_mask,
                                                            frac_split=new_fracs,
                                                            has_inbr=new_inbr)
                            x = [var.resample() for var in new.variables]
                            new_x = dm.transform_values_from_other_model(new, x)
                            new.add_variable(TimeVariable("t_some"))
                            new_x.append(3)
                            self.assertRaises(
                                ValueError,
                                new.transform_values_from_other_model,
                                dm, new_x)
