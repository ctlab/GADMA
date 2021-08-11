import unittest
import sys
import os
from os import listdir
import pickle
import moments.LD
import numpy as np
from gadma import *

try:
    import moments.LD

    MOMENTS_LD_NOT_AVAILABLE = False
except ImportError:
    MOMENTS_LD_NOT_AVAILABLE = True

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

POP_MAP = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "pop_map.txt")
REC_MAP = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "rec_map.txt")
VCF_DATA = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "vcf_data.vcf")
BED_FILE = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "solo_bed.bed")
BED_FILE_IN_ONE_BED_DIR = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "one_bed_dir", "solo_bed.bed")
BED_FILE_FIRST_FROM_THREE = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "bed_dir_1", 'region_1.bed')
BED_FILE_SECOND_FROM_THREE = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "bed_dir_1", 'region_2.bed')
BED_FILE_THIRD_FROM_THREE = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "bed_dir_1", 'region_3.bed')
FEW_BEDS = [BED_FILE_FIRST_FROM_THREE, BED_FILE_SECOND_FROM_THREE, BED_FILE_THIRD_FROM_THREE]

PREPROCESSED_DATA = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "preprocessed_data.bp")

BED_FILES_DIR = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "bed_dir_1")
EMPTY_BED_DIR = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "empty_bed_dir")
ONE_BED_DIR = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "one_bed_dir")
BED_15_DIR = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', 'bed_files_15')

SAVE_IMAGE = os.path.join(DATA_PATH, 'DATA', 'vcf_ld', "ld_curves.jpg")

DATA_HOLDER_FOR_MODELS = VCFDataHolder(
            vcf_file=VCF_DATA, popmap_file=POP_MAP,
            recombination_map=REC_MAP, bed_files_dir=BED_15_DIR,
            ld_kwargs={'r_bins': 'np.logspace(-6, -3, 7)',
                       'report': False,
                       'pops': ["deme0", "deme1"]}
        )

class TestVCFDataHolderLD(unittest.TestCase):

    def test_vcf_data_holder_ld_init(self):
        ld_data = VCFDataHolder(vcf_file=VCF_DATA, popmap_file=POP_MAP,
                                recombination_map=REC_MAP, bed_file=BED_FILE)
        self.assertEqual(ld_data.filename, VCF_DATA)
        self.assertEqual(ld_data.popmap_file, POP_MAP)
        self.assertEqual(ld_data.recombination_map, REC_MAP)
        self.assertEqual(ld_data.bed_file, BED_FILE)

        ld_data_2 = VCFDataHolder(vcf_file=VCF_DATA, popmap_file=POP_MAP,
                                  recombination_map=REC_MAP, bed_files_dir=BED_FILES_DIR)

        self.assertEqual(ld_data_2.bed_files_dir, BED_FILES_DIR)

        ld_data_3 = VCFDataHolder(vcf_file=VCF_DATA, popmap_file=POP_MAP,
                                  recombination_map=REC_MAP, bed_files_dir=EMPTY_BED_DIR)

        settings = SettingsStorage()
        settings.data_holder = ld_data_3
        settings.engine = 'momentsLD'
        self.assertRaises(ValueError, settings.read_data)

        ld_data_4 = VCFDataHolder(vcf_file=VCF_DATA, popmap_file=POP_MAP,
                                  recombination_map=REC_MAP, bed_files_dir=ONE_BED_DIR,
                                  ld_kwargs={'r_bins': 'np.logspace(-6, -3, 7)',
                                             'report': False,
                                             'pops': ["deme0", "deme1"]})

        settings = SettingsStorage()
        settings.data_holder = ld_data_4
        settings.engine = 'momentsLD'
        settings.read_data()
        self.assertTrue(ld_data_4.bed_files_dir is None)
        self.assertEqual(ld_data_4.bed_file, BED_FILE_IN_ONE_BED_DIR)


def get_settings_test():
    settings, args = get_settings()
    check_required_settings(settings)
    return settings, args


class TestSettingStorageLDStats(unittest.TestCase):

    def test_param_file_with_ld(self):
        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_with_one_bed_file')

        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        settings, _ = get_settings_test()
        settings.read_data()

        self.assertEqual(settings.output_directory, abspath('some_dir'))
        self.assertEqual(settings.data_holder.filename, VCF_DATA)
        self.assertEqual(settings.data_holder.popmap_file, POP_MAP)
        self.assertEqual(settings.data_holder.recombination_map, REC_MAP)
        self.assertEqual(settings.data_holder.bed_file, BED_FILE)

        settings.data_holder.ld_kwargs.pop('r_bins')
        self.assertRaises(ValueError, settings.read_data)

    def test_data_holder_bed_dir_adding_abspath(self):
        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_with_3_bed_files_in_dir')

        BED_FILE_ABS_1 = "region_1.bed"
        BED_FILE_ABS_2 = "region_2.bed"
        BED_FILE_ABS_3 = "region_3.bed"

        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        settings, _ = get_settings_test()
        settings.read_data()
        self.assertTrue(
            all(
                [
                    BED_FILE_ABS_1 in listdir(settings.data_holder.bed_files_dir),
                    BED_FILE_ABS_2 in listdir(settings.data_holder.bed_files_dir),
                    BED_FILE_ABS_3 in listdir(settings.data_holder.bed_files_dir)
                ]
            )
        )

    def test_errors_in_param_file(self):
        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_param_file_with_wrong_keys_in_dict')
        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        self.assertRaises(KeyError, get_settings_test)

        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_param_file_with_dict_and_wrong_engine')
        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        self.assertRaises(ValueError, get_settings_test)

        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_param_file_with_labels_twice')
        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        settings, _ = get_settings_test()
        self.assertRaises(KeyError, settings.read_data)

        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_with_wrong_bed_file_extension')
        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        settings, _ = get_settings_test()
        self.assertRaises(FileExistsError, settings.read_data)

    def test_correct_LD_small_data_processing(self):
        try:
            with open(PREPROCESSED_DATA, "rb") as fin:
                ld_stats_moments = pickle.load(fin)
        except FileNotFoundError:
            pops = ["deme0", "deme1"]
            r_bins = np.logspace(-6, -3, 7)
            ld_stats_moments = moments.LD.Parsing.compute_ld_statistics(
                VCF_DATA,
                rec_map_file=REC_MAP,
                pop_file=POP_MAP,
                pops=pops,
                r_bins=r_bins,
                report=True,
            )
            with open(f"./preprocessed_data.bp", "wb+") as fout:
                pickle.dump(ld_stats_moments, fout)
        ld_stats_moments = moments.LD.Parsing.means_from_region_data(
            {0: ld_stats_moments}, ld_stats_moments["stats"], norm_idx=0
        )

        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_test_correct_small_data_processing_1')
        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        settings, _ = get_settings_test()
        ld_stats_gadma = settings.read_data()
        self.assertEqual(len(ld_stats_moments), len(ld_stats_gadma))
        for arr in range(len(ld_stats_moments)):
            self.assertTrue(np.allclose(ld_stats_moments[arr], ld_stats_gadma[arr]))

    def test_correct_LD_data_processing_2_one_bed(self):
        r_bins = np.logspace(-6, -3, 7)
        pops = ["deme0", "deme1"]
        ld_stats_2 = moments.LD.Parsing.compute_ld_statistics(
            VCF_DATA,
            rec_map_file=REC_MAP,
            pop_file=POP_MAP,
            pops=pops,
            r_bins=r_bins,
            bed_file=BED_FILE,
            report=False
        )

        data_means_from_moments = moments.LD.Parsing.means_from_region_data(
            {0: ld_stats_2}, ld_stats_2["stats"], norm_idx=0
        )

        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_with_one_bed_file')

        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']

        settings, _ = get_settings_test()
        ld_stats_gadma_2 = settings.read_data()

        self.assertEqual(len(data_means_from_moments), len(ld_stats_gadma_2))
        for arr in range(len(data_means_from_moments)):
            self.assertTrue(all(((a == b) | (np.isnan(a) & np.isnan(b))) for a, b in zip(
                data_means_from_moments[arr], ld_stats_gadma_2[arr]
            )))

    def test_correct_LD_data_processing_3_few_bed(self):
        r_bins = np.logspace(-6, -3, 7)
        pops = ["deme0", "deme1"]
        region_stats = {}

        for region_num in range(len(FEW_BEDS)):
            region_stats.update(
                {
                    region_num: moments.LD.Parsing.compute_ld_statistics(
                        VCF_DATA, rec_map_file=REC_MAP,
                        pop_file=POP_MAP, pops=pops, r_bins=r_bins,
                        bed_file=FEW_BEDS[region_num], report=False)
                }
            )

        data_from_moments_3 = moments.LD.Parsing.bootstrap_data(region_stats)
        param_file = os.path.join(DATA_PATH, "PARAMS", 'ld_params_with_3_bed_files_in_dir')
        sys.argv = ['gadma', '-p', param_file, '-o', 'some_dir']
        settings, _ = get_settings_test()
        ld_stats_gadma_3 = settings.read_data()

        self.assertEqual(len(data_from_moments_3), len(ld_stats_gadma_3))
        for arr in range(len(data_from_moments_3['means'])):
            self.assertTrue(all((np.allclose(a, b) | (np.isnan(a) & np.isnan(b))) for a, b in zip(
                data_from_moments_3['means'][arr], ld_stats_gadma_3['means'][arr]
            )))


class TestModelSimulation(unittest.TestCase):

    def model_one_pop_moments_ld(self):
        import moments.LD

        def moments_one_pop_func(params, rho, theta):
            nu, tf = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)
            ld_stats.integrate(tf=tf, nu=[nu], rho=rho, theta=theta)

            return ld_stats

        nu = PopulationSizeVariable('nu')
        tf = TimeVariable('tf')
        rho = 4 * 10000 * np.logspace(-6, -3, 7)
        theta = 0.001

        values = {'nu': nu.resample(),
                  'tf': tf.resample()}

        data = moments_one_pop_func([values[x] for x in values], rho, theta)
        data = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(data[:-2], data[1:-1])]
            + [data[-1]],
            num_pops=data.num_pops,
            pop_ids=data.pop_ids,
        )
        data = moments.LD.Inference.sigmaD2(data)

        dm = EpochDemographicModel()
        dm.add_epoch(tf, [nu])

        return values, data, dm

    def test_one_pop_moments_ld(self):
        engine = get_engine('momentsLD')
        values, data, dm = self.model_one_pop_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_data(data)
        engine.set_model(dm)
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        model = engine.simulate(vals)

        self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
                                                            'engine and momentsLD')
        self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
                                                            'engine and momentsLD')

    def model_two_pops_moments_ld(self):
        import moments.LD

        def moments_two_pops_func(params, rho, theta):
            nu1, nu2, tf = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)
            ld_stats.integrate(tf=tf, nu=[nu1], rho=rho, theta=theta)
            ld_stats = ld_stats.split(0)
            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
            ld_stats.integrate(tf=tf, nu=lambda t: [nu1, nu2_func(t)], rho=rho, theta=theta)

            return ld_stats

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        t = TimeVariable('t')
        dyn = DynamicVariable('dyn_exp')

        values = {'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  't': t.resample(),
                  'dyn_exp': 'Exp'}

        list_for_moments_ld = ['nu1', 'nu2', 't']
        rho = 4 * 10000 * np.logspace(-6, -3, 7)
        theta = 0.001
        data = moments_two_pops_func([values[x] for x in list_for_moments_ld], rho, theta)
        data = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(data[:-2], data[1:-1])]
            + [data[-1]],
            num_pops=data.num_pops,
            pop_ids=data.pop_ids,
        )
        data = moments.LD.Inference.sigmaD2(data)

        dm = EpochDemographicModel()
        dm.add_epoch(t, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])

        return values, data, dm

    def test_two_pops_moments_ld(self):
        engine = get_engine('momentsLD')
        values, data, dm = self.model_two_pops_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        engine.set_model(dm)
        model = engine.simulate(vals)
        self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
                                                            'engine and momentsLD')
        self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
                                                            'engine and momentsLD')

    def model_4pops_moments_ld(self):
        import moments.LD

        def moments_ld_func(params, rho, theta):
            nu1, nu2, nu3, nu4, tf, m12 = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)

            ld_stats.integrate(tf=tf, nu=[nu1], rho=rho, theta=theta)
            ld_stats = ld_stats.split(0)

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
            ld_stats.integrate(nu=lambda t: [nu1, nu2_func(t)], tf=tf, rho=rho, theta=theta)

            ld_stats = ld_stats.split(1)
            m = [[0, m12, 0], [0, 0, 0], [0, 0, 0]]
            ld_stats.integrate(nu=[nu1, nu2, nu3], tf=tf, m=m, rho=rho, theta=theta)

            ld_stats = ld_stats.split(2)

            ld_stats.integrate(nu=[nu1, nu2, nu3, nu4], tf=tf, rho=rho, theta=theta)
            return ld_stats

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        nu4 = PopulationSizeVariable('nu4')
        t = TimeVariable('t')
        m12 = MigrationVariable('m12')
        dyn = DynamicVariable('dyn_exp')

        values = {'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  'nu3': nu3.resample(),
                  'nu4': nu4.resample(),
                  't': t.resample(),
                  'm12': m12.resample(),
                  'dyn_exp': 'Exp'}
        list_for_moments_ld = ['nu1', 'nu2', 'nu3', 'nu4', 't', 'm12']

        rho = 4 * 10000 * np.logspace(-6, -3, 7)
        theta = 0.001

        simulated = moments_ld_func([values[x] for x in list_for_moments_ld], rho, theta)
        simulated = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(simulated[:-2], simulated[1:-1])]
            + [simulated[-1]],
            num_pops=simulated.num_pops,
            pop_ids=simulated.pop_ids,
        )
        simulated = moments.LD.Inference.sigmaD2(simulated)

        dm = EpochDemographicModel()
        dm.add_epoch(t, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])
        dm.add_split(1, [nu2, nu3])
        dm.add_epoch(t, [nu1, nu2, nu3], mig_args=[[0, m12, 0], [0, 0, 0], [0, 0, 0]])
        dm.add_split(2, [nu3, nu4])
        dm.add_epoch(t, [nu1, nu2, nu3, nu4])

        return values, simulated, dm

    def test_4pops_moments_ld(self):
        values, simulated_by_moments, dm = self.model_4pops_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine = get_engine('momentsLD')
        engine.set_model(dm)
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        simulated_by_gadma = engine.simulate(vals)
        self.assertTrue(np.allclose(
            simulated_by_gadma[0],
            simulated_by_moments[0],
        ),
            msg='Simulations differs in '
                'engine and momentsLD')
        self.assertTrue(np.allclose(
            simulated_by_gadma[1],
            simulated_by_moments[1],
        ),
            msg='Simulations differs in '
                'engine and momentsLD')


class TestModelEvaluation(unittest.TestCase):

    def model_for_gadma_moments_evaluation(self):
        def model_moments_ld(params, rho, theta):
            nu, nu1, nu2, tf, m12, m21 = params

            ld = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)

            ld_stats.integrate(tf=tf, nu=[nu], rho=rho, theta=theta)
            ld_stats = ld_stats.split(0)

            nu2_func = lambda t: nu * (nu2 / nu) ** (t / tf)
            nu1_func = lambda t: nu * (nu1 / nu) ** (t / tf)

            m = [[0, m12], [m21, 0]]
            ld_stats.integrate(nu=lambda t: [nu1_func(t), nu2_func(t)], m=m, tf=tf, rho=rho, theta=theta)

            return ld_stats

        nu = PopulationSizeVariable('nu')
        tf = TimeVariable('tf')
        m12 = MigrationVariable('m12')
        m21 = MigrationVariable('m21')
        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        dyn1 = DynamicVariable('dyn1')
        dyn2 = DynamicVariable('dyn2')

        dm = EpochDemographicModel()
        dm.add_epoch(tf, [nu])
        dm.add_split(0, [nu, nu])
        dm.add_epoch(
            tf, [nu1, nu2],
            dyn_args=[dyn1, dyn2],
            mig_args=[[0, m12], [m21, 0]])

        values = {'nu': nu.resample(),
                  'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  'tf': tf.resample(),
                  'm12': m12.resample(),
                  'm21': m21.resample(),
                  'dyn1': 'Exp',
                  'dyn2': 'Exp'}
        list_for_moments_ld = ['nu', 'nu1', 'nu2', 'tf', 'm12', 'm21']

        rho = 4 * 10000 * np.logspace(-6, -3, 7)
        theta = 0.001

        simulated = model_moments_ld([values[x] for x in list_for_moments_ld], rho, theta)
        simulated = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(
                simulated[:-2],
                simulated[1:-1])] + [simulated[-1]],
            num_pops=simulated.num_pops,
            pop_ids=simulated.pop_ids,
        )
        simulated = moments.LD.Inference.sigmaD2(simulated)

        return values, simulated, dm

    def test_evaluation(self):
        # use data read with gadma
        engine = get_engine('momentsLD')
        engine.data_holder = VCFDataHolder(
            vcf_file=VCF_DATA, popmap_file=POP_MAP,
            recombination_map=REC_MAP, bed_files_dir=BED_15_DIR,
            ld_kwargs={'r_bins': 'np.logspace(-6, -3, 7)',
                       'report': False,
                       'pops': ["deme0", "deme1"]}
        )
        engine.set_data(engine.data_holder)
        data_gadma = engine.data
        data_moments = engine.data
        self.assertEqual(len(data_gadma), len(data_moments))
        for arr in range(len(data_moments['means'])):
            self.assertTrue(all(((a == b) | (np.isnan(a) & np.isnan(b))) for a, b in zip(
                data_moments['means'][arr], data_gadma['means'][arr]
            )))

        # simulate ld_stats with moments
        values, simulated_by_moments, dm = self.model_for_gadma_moments_evaluation()
        # check simulation ld_stats with GADMA
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_model(dm)
        simulated_by_gadma = engine.simulate(values=vals)
        for ii in range(len(simulated_by_gadma)):
            self.assertTrue(np.allclose(
                simulated_by_gadma[ii],
                simulated_by_moments[ii],
            ),
                msg='Simulations differs in '
                    'engine and momentsLD')

        means, varcovs = moments.LD.Inference.remove_normalized_data(
            data_moments["means"],
            data_moments["varcovs"],
            num_pops=simulated_by_moments.num_pops)
        simulated_by_moments = moments.LD.Inference.remove_normalized_lds(
            simulated_by_moments, normalization=0)
        ll_moments = moments.LD.Inference.ll_over_bins(
            means,
            simulated_by_moments,
            varcovs
        )
        ll_gadma = engine.evaluate(vals)

        self.assertEqual(ll_gadma, ll_moments)

    def test_draw_curves(self):

        engine = get_engine('momentsLD')
        engine.data_holder = VCFDataHolder(
            vcf_file=VCF_DATA, popmap_file=POP_MAP,
            recombination_map=REC_MAP, bed_files_dir=BED_15_DIR,
            ld_kwargs={'r_bins': 'np.logspace(-6, -3, 7)',
                       'report': False,
                       'pops': ["deme0", "deme1"]}
        )

        values, simulated_by_moments, dm = self.model_for_gadma_moments_evaluation()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_model(dm)

        engine.draw_ld_curves(values=vals, save_file=SAVE_IMAGE)
