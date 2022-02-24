import json
import unittest
import gadma
from gadma import *
import itertools
import os
import copy
import numpy as np
import pytest
from gadma.engines.engine import get_engine

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

POP_MAP = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "pop_map.txt")
REC_MAPS_DIR = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "rec_maps")
VCF_DATA_LD = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "vcf_data.vcf")
TEST_OUTPUT = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "test_output")
SAVE_IMAGE = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "ld_curves.jpg")
LL_TEST_DATA = os.path.join(
    DATA_PATH, 'DATA', 'vcf_ld', "ll_test_data")

DATA_HOLDER_FOR_MODELS = VCFDataHolder(
            vcf_file=VCF_DATA_LD,
            popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR
)

TEST_STRUCTURES = [(1,), (2,),
                   (1,1), (2,1), (1,2),
                   (1,1,1)]

BASE_TEST_STRUCTURES = [(1,), (2,), (2,1), (1, 1, 1)]

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.structure_demographic_model', lineno=77)
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.structure_demographic_model', lineno=81)

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
                dm.mutation_rate = 1e-8
                # Change domain as when time is very small then our check
                # sometimes is wrong
                for i in range(len(dm.variables)):
                    if isinstance(dm.variables[i], TimeVariable):
                        dm.variables[i].domain = [1e-2, 5]
                variables = dm.variables
                x = [var.resample() for var in variables]
                bad_structure = list(structure)
                for i in range(len(bad_structure)):
                    if bad_structure[i] == 1:
                        continue
                    bad_structure[i] -= 1
                    self.assertRaises(ValueError, dm.increase_structure,
                                      bad_structure)

                check_ll = np.random.choice([True, False], p=[1/6, 5/6])
                for engine in all_simulation_engines():
                    if engine.id not in ["dadi", "moments"] and not has_anc:
                        continue
                    engine.set_model(dm)
                    if engine.id == 'dadi':
                        sizes = [8 for _ in range(len(structure))]
                        if len(structure) == 1:
                            sizes = [20]
                        kwargs = {"pts": [4, 6, 8]}  # pts
                    else:
                        sizes = [4 for _ in range(len(structure))]
                        kwargs = {}

                    # check that our x is valid for the engine
                    if engine.id == "momi":
                        x_cor = [el if el != "Lin" else "Exp" for el in x]
                    else:
                        x_cor = x

                    # simulate data
                    try:  # momi fails sometimes
                        data = engine.simulate(
                            values=x_cor,
                            ns=sizes,
                            sequence_length=1e6,
                            population_labels=[f"Pop{i}" for i in range(len(sizes))],
                            **kwargs
                        )
                    except ValueError as e:
                        if str(e).startswith("zero-size array to reduction"):
                            failed += 1
                            continue
                        raise e
                    engine.set_data(data)

                    if check_ll:
                        # get ll of data
                        try:
                            ll_true = engine.evaluate(x_cor, **kwargs)
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
#                        print(msg)
                        new_dm = copy.deepcopy(dm)
                        new_dm, new_X = new_dm.increase_structure(
                            new_structure, [x_cor])
                        an_dm = copy.deepcopy(dm)
                        _, X_none = an_dm.increase_structure()
                        self.assertEqual(X_none, None)
                        engine.set_model(new_dm)
#                        print("!!!", dm.var2value(x), new_dm.var2value(new_X[0]))
                        if check_ll and random_int == i:
                            try:
                                new_ll = engine.evaluate(new_X[0], **kwargs)
                                if has_anc and engine.id in ["dadi", "moments"]:
                                    continue
                                else:
                                    is_equal = np.allclose(ll_true, new_ll)
                                self.assertTrue(is_equal,
                                                msg=f"{ll_true} != {new_ll} : {msg}")
                            except AttributeError:
                                assert engine.id in ["dadi"]
                                failed += 1
                            except TypeError:
                                assert engine.id in ["dadi"]
                                failed += 1

                dm.final_structure = dm.get_structure()
                self.assertRaises(ValueError, dm.increase_structure)
        self.assertTrue(failed <= 5, "Dadi failed more that 5 times")

    def _data_for_ll_ld_test(self, ii):
        return VCFDataHolder(
            vcf_file=None,
            preprocessed_data=os.path.join(
                LL_TEST_DATA, f"preprocessed_data_{ii}.bp"
            ),
            popmap_file=None,
            population_labels=["pop0", "pop1"],
        )

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_test_likelihood_after_increase_moments_ld(self):
        LL_TEST_STRUCTURE = [(2, 1)]
        args_for_ld_test_ll_after_increase = []
        for ii in [2, 5, 19]:
            args_for_ld_test_ll_after_increase.append(
                list(itertools.product([False, True], repeat=7))[ii]
            )
        ii = 1
        for structure in LL_TEST_STRUCTURE:
            for create_migs, create_sels, create_dyns, sym_migs, fracs, has_anc, inbr in \
                    args_for_ld_test_ll_after_increase:

                moments.LD.Inference._varcov_inv_cache = {}

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
                                                     has_anc_size=True)
                dm = model_generator(structure)
                dm.mutation_rate = 1e-8
                engine = get_engine("momentsLD")
                engine.model = dm
                engine.set_data(self._data_for_ll_ld_test(ii))
                x_file = os.path.join(LL_TEST_DATA, "all_x.txt")
                with open(x_file, "r") as file:
                    all_x = json.load(file)
                x = all_x[f"{ii}"]
                ll_true = engine.evaluate(x)
                for i in range(len(structure)):
                    new_structure = list(copy.copy(structure))
                    new_structure[i] += 1
                    msg = f"Increase structure from {structure} to " \
                          f"{new_structure} for engine {engine.id}. " \
                          f"create_migs: {create_migs}, " \
                          f"create_sels: {create_sels}, " \
                          f"create_dyns: {create_dyns}, " \
                          f"sym_migs: {sym_migs}, " \
                          f"fracs: {fracs}, " \
                          f"has_anc: {True}, "
                    #                        print(msg)
                    new_dm = copy.deepcopy(dm)
                    new_dm, new_X = new_dm.increase_structure(
                        new_structure, [x])
                    an_dm = copy.deepcopy(dm)
                    _, X_none = an_dm.increase_structure()
                    self.assertEqual(X_none, None)
                    engine.set_model(new_dm)
                    new_ll = engine.evaluate(new_X[0])

                    is_equal = np.allclose(ll_true, new_ll)
                    self.assertTrue(is_equal,
                                    msg=f"{ll_true} != {new_ll} : {msg}")
                ii += 1

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
        for structure in BASE_TEST_STRUCTURES:
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
