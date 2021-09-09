# import shutil
# import unittest
# import sys
# import os
# import pickle
# import moments.LD
# import numpy as np
# from os import listdir
# from gadma import *
# from pathlib import Path
# from gadma.utils.utils import create_bed_files
#
# try:
#     import moments.LD
#
#     MOMENTS_LD_NOT_AVAILABLE = False
# except ImportError:
#     MOMENTS_LD_NOT_AVAILABLE = True
#
# DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
#
# POP_MAP = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "pop_map.txt")
# REC_MAPS_DIR = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "rec_maps")
# VCF_DATA_FEW_CHR = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "vcf_data_few_chr.vcf")
# VCF_DATA_LD = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "vcf_data.vcf")
# TEST_OUTPUT = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "test_output")
# TEST_BED_FILES = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "test_bed_files")
# SFS_DATA = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "wrong_data.fs")
#
# PREPROCESSED_DATA = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "preprocessed_data.bp")
#
# SAVE_IMAGE = os.path.join(
#     DATA_PATH, 'DATA', 'vcf_ld', "ld_curves.jpg")
#
# DATA_HOLDER_FOR_MODELS = VCFDataHolder(
#             vcf_file=VCF_DATA_LD,
#             popmap_file=POP_MAP,
#             recombination_maps=REC_MAPS_DIR,
#             ld_kwargs={
#                 'r_bins': 'np.logspace(-6, -3, 7)',
#                 'report': False,
#             }
# )
#
#
# class TestVCFDataHolderLD(unittest.TestCase):
#
#     def tearDown(self):
#         if Path(f"{TEST_OUTPUT}/bed_files/").exists():
#             shutil.rmtree(f"{TEST_OUTPUT}/bed_files/")
#
#     def test_vcf_data_holder_ld_init(self):
#         ld_data = VCFDataHolder(
#             vcf_file=VCF_DATA_LD,
#             popmap_file=POP_MAP,
#             recombination_maps=REC_MAPS_DIR,
#             output_directory=TEST_OUTPUT
#         )
#         self.assertEqual(ld_data.filename, VCF_DATA_LD)
#         self.assertEqual(ld_data.popmap_file, POP_MAP)
#         self.assertEqual(ld_data.recombination_maps, REC_MAPS_DIR)
#
#     def test_sfs_data_holder(self):
#         ld_wrong_data = SFSDataHolder(
#             sfs_file=SFS_DATA)
#
#         settings = SettingsStorage()
#         settings.engine = "momentsLD"
#         settings.data_holder = ld_wrong_data
#         self.assertRaises(ValueError, settings.read_data)
#
#
#
# def get_settings_test():
#     settings, args = get_settings()
#     check_required_settings(settings)
#     return settings, args
#
#
# class TestSettingStorageLDStats(unittest.TestCase):
#     def tearDown(self):
#         if Path(f"{TEST_OUTPUT}/bed_files/").exists():
#             shutil.rmtree(f"{TEST_OUTPUT}/bed_files/")
#
#     def test_param_file_with_ld(self):
#         param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_test_correct')
#
#         sys.argv = ['gadma', '-p', param_file]
#         settings, _ = get_settings_test()
#         settings.read_data()
#
#         self.assertEqual(settings.output_directory, abspath(TEST_OUTPUT))
#         self.assertEqual(settings.data_holder.filename, VCF_DATA_LD)
#         self.assertEqual(settings.data_holder.popmap_file, POP_MAP)
#         self.assertEqual(settings.data_holder.recombination_maps, REC_MAPS_DIR)
#
#     def test_errors_in_param_file(self):
#         param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_param_file_with_wrong_keys_in_dict')
#         sys.argv = ['gadma', '-p', param_file]
#         self.assertRaises(KeyError, get_settings_test)
#
#         param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_param_file_with_dict_and_wrong_engine')
#         sys.argv = ['gadma', '-p', param_file]
#         self.assertRaises(ValueError, get_settings_test)
#
#         param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_without_anc_size_as_parameter')
#         sys.argv = ['gadma', '-p', param_file]
#         settings, args = get_settings_test()
#         self.assertTrue(settings.ancestral_size_as_parameter, True)
#
#     def test_correct_LD_data_processing(self):
#         try:
#             with open(PREPROCESSED_DATA, "rb") as fin:
#                 ld_stats_moments = pickle.load(fin)
#         except: # NOQA
#             pops = ["deme0", "deme1"]
#             r_bins = np.logspace(-6, -3, 7)
#             moments_regions = {}
#
#             for ii in range(1, 16):
#                 moments_regions.update(
#                     {
#                         f"{ii}": moments.LD.Parsing.compute_ld_statistics(
#                             VCF_DATA_LD,
#                             rec_map_file=f"{REC_MAPS_DIR}/rec_map_1.txt",
#                             pop_file=POP_MAP,
#                             bed_file=f"{TEST_BED_FILES}/bed_file_1_{ii}.bed",
#                             pops=pops,
#                             r_bins=r_bins,
#                             report=False,
#                         )
#                     }
#                 )
#
#             ld_stats_moments = moments.LD.Parsing.bootstrap_data(moments_regions)
#             with open(f"{PREPROCESSED_DATA}", "wb+") as fout:
#                 pickle.dump(ld_stats_moments, fout)
#         param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_test_correct')
#         sys.argv = ['gadma', '-p', param_file]
#         settings, _ = get_settings_test()
#         ld_stats_gadma = settings.read_data()
#         self.assertEqual(len(ld_stats_moments), len(ld_stats_gadma))
#         for arr in range(len(ld_stats_moments["means"])):
#             self.assertTrue(np.allclose(
#                 ld_stats_moments["means"][arr],
#                 ld_stats_gadma["means"][arr]))
#
#
# class TestModelSimulation(unittest.TestCase):
#
#     def tearDown(self):
#         if Path(f"{TEST_OUTPUT}/bed_files/").exists():
#             shutil.rmtree(f"{TEST_OUTPUT}/bed_files/")
#
#     def model_one_pop_moments_ld(self):
#         import moments.LD
#
#         def moments_one_pop_func(params, rho, theta):
#             nu, tf = params
#
#             ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
#             ld_stats = moments.LD.LDstats(ld, num_pops=1)
#             ld_stats.integrate(tf=tf, nu=[nu], rho=rho, theta=theta)
#
#             return ld_stats
#
#         nu = PopulationSizeVariable('nu')
#         tf = TimeVariable('tf')
#         rhos = 4 * 10000 * np.logspace(-6, -3, 7)
#         theta = 4 * 10000 * 6.0e-5
#
#         values = {'nu': nu.resample(),
#                   'tf': tf.resample()}
#
#         data = moments_one_pop_func([values[x] for x in values], rhos, theta)
#         data = moments.LD.LDstats(
#             [(y_l + y_r) / 2 for y_l, y_r in zip(data[:-2], data[1:-1])]
#             + [data[-1]],
#             num_pops=data.num_pops,
#             pop_ids=data.pop_ids,
#         )
#         data = moments.LD.Inference.sigmaD2(data)
#
#         dm = EpochDemographicModel()
#         dm.add_epoch(tf, [nu])
#
#         return values, data, dm
#
#     def test_one_pop_moments_ld(self):
#         engine = get_engine('momentsLD')
#         values, data, dm = self.model_one_pop_moments_ld()
#         vals = [values[var.name] for var in dm.variables
#                 if (var.name in values)]
#         engine.set_data(data)
#         engine.set_model(dm)
#         engine.model.Nanc_size = 10000
#         engine.model.mutation_rate = 6.0e-5
#         engine.data_holder = DATA_HOLDER_FOR_MODELS
#         model = engine.simulate(vals)
#
#         self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
#                                                             'engine and momentsLD')
#         self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
#                                                             'engine and momentsLD')
#
#     def model_two_pops_moments_ld(self):
#         import moments.LD
#
#         def moments_two_pops_func(params, rho, theta):
#             nu1, nu2, tf = params
#
#             ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
#             ld_stats = moments.LD.LDstats(ld, num_pops=1)
#             ld_stats.integrate(tf=tf, nu=[nu1], rho=rho, theta=theta)
#             ld_stats = ld_stats.split(0)
#             nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
#             ld_stats.integrate(tf=tf, nu=lambda t: [nu1, nu2_func(t)], rho=rho, theta=theta)
#
#             return ld_stats
#
#         nu1 = PopulationSizeVariable('nu1')
#         nu2 = PopulationSizeVariable('nu2')
#         t = TimeVariable('t')
#         dyn = DynamicVariable('dyn_exp')
#
#         values = {'nu1': nu1.resample(),
#                   'nu2': nu2.resample(),
#                   't': t.resample(),
#                   'dyn_exp': 'Exp'}
#
#         list_for_moments_ld = ['nu1', 'nu2', 't']
#         rhos = 4 * 10000 * np.logspace(-6, -3, 7)
#         theta = 4 * 10000 * 6.0e-5
#         data = moments_two_pops_func([values[x] for x in list_for_moments_ld], rhos, theta)
#         data = moments.LD.LDstats(
#             [(y_l + y_r) / 2 for y_l, y_r in zip(data[:-2], data[1:-1])]
#             + [data[-1]],
#             num_pops=data.num_pops,
#             pop_ids=data.pop_ids,
#         )
#         data = moments.LD.Inference.sigmaD2(data)
#
#         dm = EpochDemographicModel()
#         dm.add_epoch(t, [nu1])
#         dm.add_split(0, [nu1, nu1])
#         dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])
#
#         return values, data, dm
#
#     def test_two_pops_moments_ld(self):
#         engine = get_engine('momentsLD')
#         values, data, dm = self.model_two_pops_moments_ld()
#         vals = [values[var.name] for var in dm.variables
#                 if (var.name in values)]
#         engine.data_holder = DATA_HOLDER_FOR_MODELS
#         engine.set_model(dm)
#         engine.model.Nanc_size = 10000
#         engine.model.mutation_rate = 6.0e-5
#         model = engine.simulate(vals)
#         self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
#                                                             'engine and momentsLD')
#         self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
#                                                             'engine and momentsLD')
#
#     def model_4pops_moments_ld(self):
#         import moments.LD
#
#         def moments_ld_func(params, rho, theta):
#             nu1, nu2, nu3, nu4, tf, m12 = params
#
#             ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
#             ld_stats = moments.LD.LDstats(ld, num_pops=1)
#
#             ld_stats.integrate(tf=tf, nu=[nu1], rho=rho, theta=theta)
#             ld_stats = ld_stats.split(0)
#
#             nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
#             ld_stats.integrate(nu=lambda t: [nu1, nu2_func(t)], tf=tf, rho=rho, theta=theta)
#
#             ld_stats = ld_stats.split(1)
#             m = [[0, m12, 0], [0, 0, 0], [0, 0, 0]]
#             ld_stats.integrate(nu=[nu1, nu2, nu3], tf=tf, m=m, rho=rho, theta=theta)
#
#             ld_stats = ld_stats.split(2)
#
#             ld_stats.integrate(nu=[nu1, nu2, nu3, nu4], tf=tf, rho=rho, theta=theta)
#             return ld_stats
#
#         nu1 = PopulationSizeVariable('nu1')
#         nu2 = PopulationSizeVariable('nu2')
#         nu3 = PopulationSizeVariable('nu3')
#         nu4 = PopulationSizeVariable('nu4')
#         t = TimeVariable('t')
#         m12 = MigrationVariable('m12')
#         dyn = DynamicVariable('dyn_exp')
#
#         values = {'nu1': nu1.resample(),
#                   'nu2': nu2.resample(),
#                   'nu3': nu3.resample(),
#                   'nu4': nu4.resample(),
#                   't': t.resample(),
#                   'm12': m12.resample(),
#                   'dyn_exp': 'Exp'}
#         list_for_moments_ld = ['nu1', 'nu2', 'nu3', 'nu4', 't', 'm12']
#
#         rhos = 4 * 10000 * np.logspace(-6, -3, 7)
#         theta = 4 * 10000 * 6.0e-5
#
#         simulated = moments_ld_func([values[x] for x in list_for_moments_ld], rhos, theta)
#         simulated = moments.LD.LDstats(
#             [(y_l + y_r) / 2 for y_l, y_r in zip(simulated[:-2], simulated[1:-1])]
#             + [simulated[-1]],
#             num_pops=simulated.num_pops,
#             pop_ids=simulated.pop_ids,
#         )
#         simulated = moments.LD.Inference.sigmaD2(simulated)
#
#         dm = EpochDemographicModel()
#         dm.add_epoch(t, [nu1])
#         dm.add_split(0, [nu1, nu1])
#         dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])
#         dm.add_split(1, [nu2, nu3])
#         dm.add_epoch(t, [nu1, nu2, nu3], mig_args=[[0, m12, 0], [0, 0, 0], [0, 0, 0]])
#         dm.add_split(2, [nu3, nu4])
#         dm.add_epoch(t, [nu1, nu2, nu3, nu4])
#
#         return values, simulated, dm
#
#     def test_4pops_moments_ld(self):
#         values, simulated_by_moments, dm = self.model_4pops_moments_ld()
#         vals = [values[var.name] for var in dm.variables
#                 if (var.name in values)]
#         engine = get_engine('momentsLD')
#         engine.set_model(dm)
#         engine.model.Nanc_size = 10000
#         engine.model.mutation_rate = 6.0e-5
#         engine.data_holder = DATA_HOLDER_FOR_MODELS
#         simulated_by_gadma = engine.simulate(vals)
#         self.assertTrue(np.allclose(
#             simulated_by_gadma[0],
#             simulated_by_moments[0],
#         ),
#             msg='Simulations differs in '
#                 'engine and momentsLD')
#         self.assertTrue(np.allclose(
#             simulated_by_gadma[1],
#             simulated_by_moments[1],
#         ),
#             msg='Simulations differs in '
#                 'engine and momentsLD')
#
#
# class TestModelEvaluation(unittest.TestCase):
#
#     def tearDown(self):
#         if Path(f"{TEST_OUTPUT}/").exists():
#             shutil.rmtree(f"{TEST_OUTPUT}/")
#         os.makedirs(TEST_OUTPUT)
#
#     def model_for_gadma_moments_evaluation(self):
#         def model_moments_ld(params, rho, theta):
#             nu, nu1, nu2, tf, m12, m21 = params
#
#             ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
#             ld_stats = moments.LD.LDstats(ld, num_pops=1)
#
#             ld_stats.integrate(tf=tf, nu=[nu], rho=rho, theta=theta)
#             ld_stats = ld_stats.split(0)
#
#             nu2_func = lambda t: nu * (nu2 / nu) ** (t / tf)
#             nu1_func = lambda t: nu * (nu1 / nu) ** (t / tf)
#
#             m = [[0, m12], [m21, 0]]
#             ld_stats.integrate(nu=lambda t: [nu1_func(t), nu2_func(t)], m=m, tf=tf, rho=rho, theta=theta)
#
#             return ld_stats
#
#         nu = PopulationSizeVariable('nu')
#         tf = TimeVariable('tf')
#         m12 = MigrationVariable('m12')
#         m21 = MigrationVariable('m21')
#         nu1 = PopulationSizeVariable('nu1')
#         nu2 = PopulationSizeVariable('nu2')
#         dyn1 = DynamicVariable('dyn1')
#         dyn2 = DynamicVariable('dyn2')
#
#         dm = EpochDemographicModel()
#         dm.add_epoch(tf, [nu])
#         dm.add_split(0, [nu, nu])
#         dm.add_epoch(
#             tf, [nu1, nu2],
#             dyn_args=[dyn1, dyn2],
#             mig_args=[[0, m12], [m21, 0]])
#
#         values = {'nu': nu.resample(),
#                   'nu1': nu1.resample(),
#                   'nu2': nu2.resample(),
#                   'tf': tf.resample(),
#                   'm12': m12.resample(),
#                   'm21': m21.resample(),
#                   'dyn1': 'Exp',
#                   'dyn2': 'Exp'}
#         list_for_moments_ld = ['nu', 'nu1', 'nu2', 'tf', 'm12', 'm21']
#
#         rhos = 4 * 10000 * np.logspace(-6, -3, 7)
#         theta = 4 * 10000 * 6.0e-5
#
#         simulated = model_moments_ld([values[x] for x in list_for_moments_ld], rhos, theta)
#         simulated = moments.LD.LDstats(
#             [(y_l + y_r) / 2 for y_l, y_r in zip(
#                 simulated[:-2],
#                 simulated[1:-1])] + [simulated[-1]],
#             num_pops=simulated.num_pops,
#             pop_ids=simulated.pop_ids,
#         )
#         simulated = moments.LD.Inference.sigmaD2(simulated)
#
#         return values, simulated, dm
#
#     def test_evaluation(self):
#         # use data read with gadma
#         engine = get_engine('momentsLD')
#         engine.data_holder = VCFDataHolder(
#             vcf_file=VCF_DATA_LD, popmap_file=POP_MAP,
#             recombination_maps=REC_MAPS_DIR,
#             output_directory=TEST_OUTPUT
#         )
#         engine.set_data(engine.data_holder)
#         data_gadma = engine.data
#         data_moments = engine.data
#         self.assertEqual(len(data_gadma), len(data_moments))
#         for arr in range(len(data_moments['means'])):
#             self.assertTrue(all(((a == b) | (np.isnan(a) & np.isnan(b))) for a, b in zip(
#                 data_moments['means'][arr], data_gadma['means'][arr]
#             )))
#
#         # simulate ld_stats with moments
#         values, simulated_by_moments, dm = self.model_for_gadma_moments_evaluation()
#         # check simulation ld_stats with GADMA
#         vals = [values[var.name] for var in dm.variables
#                 if (var.name in values)]
#         engine.set_model(dm)
#         engine.model.Nanc_size = 10000
#         engine.model.mutation_rate = 6.0e-5
#         simulated_by_gadma = engine.simulate(values=vals)
#         for ii in range(len(simulated_by_gadma)):
#             self.assertTrue(np.allclose(
#                 simulated_by_gadma[ii],
#                 simulated_by_moments[ii],
#             ),
#                 msg='Simulations differs in '
#                     'engine and momentsLD')
#
#         means, varcovs = moments.LD.Inference.remove_normalized_data(
#             data_moments["means"],
#             data_moments["varcovs"],
#             num_pops=simulated_by_moments.num_pops)
#         simulated_by_moments = moments.LD.Inference.remove_normalized_lds(
#             simulated_by_moments, normalization=0)
#         ll_moments = moments.LD.Inference.ll_over_bins(
#             means,
#             simulated_by_moments,
#             varcovs
#         )
#         ll_gadma = engine.evaluate(vals)
#
#         self.assertEqual(ll_gadma, ll_moments)
#
#     def test_draw_curves(self):
#
#         engine = get_engine('momentsLD')
#         engine.data_holder = VCFDataHolder(
#             vcf_file=VCF_DATA_LD, popmap_file=POP_MAP,
#             recombination_maps=REC_MAPS_DIR,
#             output_directory=TEST_OUTPUT
#         )
#
#         values, simulated_by_moments, dm = self.model_for_gadma_moments_evaluation()
#         vals = [values[var.name] for var in dm.variables
#                 if (var.name in values)]
#         engine.set_model(dm)
#         engine.model.Nanc_size = 10000
#         engine.model.mutation_rate = 6.0e-5
#         engine.inner_data = engine._read_data(engine.data_holder)
#
#         engine.draw_data_comp_plot(values=vals, save_file=SAVE_IMAGE,)
#
#
# class BedFilesCreation(unittest.TestCase):
#
#     def tearDown(self):
#         if Path(f"{TEST_OUTPUT}/").exists():
#             shutil.rmtree(f"{TEST_OUTPUT}/")
#         os.makedirs(TEST_OUTPUT)
#
#     def test_create_bed_files(self):
#         chromosomes = create_bed_files(DATA_HOLDER_FOR_MODELS.filename, TEST_OUTPUT)
#
#         test_bed_files_reference_info = []
#         test_bed_files_check_info = []
#
#         for file in listdir(f"{TEST_BED_FILES}/"):
#             with open(f"{TEST_BED_FILES}/{file}", "r") as bed_file:
#                 test_bed_files_reference_info.append(bed_file.readline())
#
#         for file in listdir(f"{TEST_OUTPUT}/bed_files/"):
#             with open(f"{TEST_OUTPUT}/bed_files/{file}", "r") as bed_file:
#                 test_bed_files_check_info.append(bed_file.readline())
#
#         self.assertTrue(len(listdir(f"{TEST_OUTPUT}/bed_files/")), 15)
#         for ii, nums in enumerate(test_bed_files_reference_info):
#             self.assertEqual(test_bed_files_reference_info[ii],
#                              test_bed_files_check_info[ii])
