import unittest
import gadma
from gadma import *
from gadma.engines import register_engine, Engine
from gadma.models import Model
from gadma.data import create_bed_files_and_extract_chromosomes
import os
import pickle
import numpy as np
from pathlib import Path
import shutil
import pytest


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

R_BINS = np.array(
    [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])

DATA_HOLDER_FOR_MODELS = VCFDataHolder(
            vcf_file=VCF_DATA_LD,
            popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
            population_labels=["deme0", "deme1"]
)


class TestEngines(unittest.TestCase):
    def tearDown(self):
        if Path("./plot.png").exists():
            os.remove("./plot.png")

    def test_existence(self):
        self.assertTrue(len(list(all_engines())) > 0)
        self.assertTrue(len(list(all_available_engines())) > 0)
        self.assertTrue(len(list(all_simulation_engines())) > 0)
        self.assertTrue(len(list(all_drawing_engines())) > 0)
        for engine in all_engines():
            options = {}
            if engine.id == 'dadi':
                options['pts'] = [10, 20, 30]
            if engine.can_evaluate:
                self.assertRaises(ValueError, engine.evaluate, [], **options)
            else:
                self.assertRaises(NotImplementedError,
                                  engine.evaluate, [], **options)

        self.assertRaises(ValueError, register_engine,
                          get_engine("dadi"))
        self.assertRaises(ValueError, get_engine, "some_strange_name")

        engine = Engine()
        self.assertRaises(NotImplementedError, engine._read_data, "some")
        self.assertRaises(NotImplementedError, engine.evaluate, [])
        self.assertRaises(NotImplementedError, engine.generate_code, [])
        self.assertRaises(NotImplementedError,
                          engine.update_data_holder_with_inner_data)

        dadi_engine = get_engine('dadi')
        model = Model()
        self.assertRaises(ValueError, dadi_engine.set_model, model)
        self.assertRaises(ValueError,
                          dadi_engine.set_and_evaluate, [], None, None)

        model = EpochDemographicModel()
        self.assertRaises(ValueError,
                          dadi_engine.set_and_evaluate, [], model, None)

        # the list of the class is not changed, it is common for engines
        engine.supported_models = [EpochDemographicModel]
        engine.model = EpochDemographicModel()
        self.assertRaises(NotImplementedError, engine.get_N_ancestral, [])

    def model_3pop_dadi(self, ns, pts):
        import dadi
        def dadi_func(params, ns, pts):
            nu1, nu2, nu3, t1, t2, t3, m12 = params
            xx = dadi.Numerics.default_grid(pts)
            phi = dadi.PhiManip.phi_1D(xx)

            phi = dadi.Integration.one_pop(phi, xx, nu=nu1, T=t1)
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / t2)
            phi = dadi.Integration.two_pops(phi, xx, T=t2, nu1=nu1,
                                            nu2=nu2_func)
            phi = dadi.PhiManip.phi_2D_to_3D_split_1(xx, phi)

            phi = dadi.Integration.three_pops(phi, xx, T=t3, nu1=nu1,
                                              nu2=nu2, nu3=nu3, m12=m12)

            sfs = dadi.Spectrum.from_phi(phi, ns, [xx] * len(ns))
            return sfs

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        t1 = TimeVariable('t1', domain=[1e-3, 5])
        t2 = TimeVariable('t2', domain=[1e-3, 5])
        t3 = TimeVariable('t3')
        m12 = MigrationVariable('m12')
        dyn = DynamicVariable('dyn_exp')

        values = {'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  'nu3': nu3.resample(),
                  't1': t1.resample(),
                  't2': t2.resample(),
                  't3': t3.resample(),
                  'm12': m12.resample(),
                  'dyn_exp': 'Exp'}
        list_in_dadi = ['nu1', 'nu2', 'nu3', 't1', 't2', 't3', 'm12']

        func_ex = dadi.Numerics.make_extrap_log_func(dadi_func)
        data = func_ex([values[x] for x in list_in_dadi], ns, pts)
        ll = dadi.Inference.ll_multinom(data, data)

        dm = EpochDemographicModel()
        dm.add_epoch(t1, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t2, [nu1, nu2], dyn_args=['Sud', dyn])
        dm.add_split(0, [nu1, nu3])
        dm.add_epoch(t3, [nu1, nu2, nu3], mig_args=[[0, m12, 0], [0, 0, 0], [0, 0, 0]])

        return data, dm, values, ll

    def model_4pop_moments(self, ns, dt_fac):
        import moments
        def moments_func(params, ns, dt_fac):
            nu1, nu2, nu3, nu4, tf, m12 = params

            sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
            fs = moments.Spectrum(sts)

            fs.integrate(Npop=[nu1], tf=tf, dt_fac=dt_fac)
            fs = moments.Manips.split_1D_to_2D(fs, ns[0], sum(ns[1:]))

            nu2_func = lambda t: nu1 * (nu2 / nu1) ** (t / tf)
            fs.integrate(Npop=lambda t: [nu1, nu2_func(t)], tf=tf, dt_fac=dt_fac)
            fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2] + ns[3])

            m = [[0, m12, 0], [0, 0, 0], [0, 0, 0]]
            fs.integrate(Npop=[nu1, nu2, nu3], tf=tf, m=m, dt_fac=dt_fac)
            fs = moments.Manips.split_3D_to_4D_3(fs, ns[2], ns[3])

            fs.integrate(Npop=[nu1, nu2, nu3, nu4], tf=tf, dt_fac=dt_fac)

            return fs

        nu1 = PopulationSizeVariable('nu1')
        nu2 = PopulationSizeVariable('nu2')
        nu3 = PopulationSizeVariable('nu3')
        nu4 = PopulationSizeVariable('nu4')
        t = TimeVariable('t', domain=[1e-3, 5])
        m12 = MigrationVariable('m12')
        dyn = DynamicVariable('dyn_exp')

        values = {'nu1': nu1.resample(),
                  'nu2': nu2.resample(),
                  'nu3': nu3.resample(),
                  'nu4': nu4.resample(),
                  't': t.resample(),
                  'm12': m12.resample(),
                  'dyn_exp': 'Exp'}
        list_in_dadi = ['nu1', 'nu2', 'nu3', 'nu4', 't', 'm12']

        data = moments_func([values[x] for x in list_in_dadi], ns, dt_fac)
        ll = moments.Inference.ll_multinom(data, data)

        dm = EpochDemographicModel()
        dm.add_epoch(t, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])
        dm.add_split(1, [nu2, nu3])
        dm.add_epoch(t, [nu1, nu2, nu3], mig_args=[[0, m12, 0], [0, 0, 0], [0, 0, 0]])
        dm.add_split(2, [nu3, nu4])
        dm.add_epoch(t, [nu1, nu2, nu3, nu4])
        return data, dm, values, ll

    def test_dadi_engine(self):
        engine = get_engine('dadi')

        ns = (4, 4, 4)
        # values are random and for some of them simulation is very long on
        # big grid, small grid is enough as the same one is used for dadi
        pts = [6, 10, 14]

        data, dm, values, ll = self.model_3pop_dadi(ns, pts)
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]

        engine.set_data(data)
        engine.set_model(dm)
        seq_len = 1e6
        pops = None
        model = engine.simulate(vals, ns, seq_len, pops, pts)
        self.assertTrue(np.allclose(model, data), msg="Simulations differs in"
                                                      " engine and in dadi.")
        self.assertEqual(engine.set_and_evaluate(
            data=data,
            model=dm,
            values=vals,
            options={"pts": pts}
        ), ll)

    def test_import_without_optional_packages(self):
        """
        GADMA is imported when some packages are not installed: engines of
        moments are not available without moments, pandas is needed only
        for the scripts of confidence intervals.
        """
        import ast
        import subprocess
        import sys
        code = (
            "import sys, warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "for name in {blocked!r}:\n"
            "    sys.modules[name] = None  # import of it fails\n"
            "import gadma\n"
            "print(sorted(eng.id for eng in gadma.all_engines()))\n"
        )
        env = dict(os.environ)
        repo_dir = os.path.dirname(os.path.dirname(gadma.__file__))
        env["PYTHONPATH"] = os.pathsep.join(
            [repo_dir] + [el for el in [env.get("PYTHONPATH")] if el])
        all_ids = sorted(eng.id for eng in all_engines())
        for blocked in [["moments"], ["pandas"]]:
            res = subprocess.run(
                [sys.executable, "-c", code.format(blocked=blocked)],
                capture_output=True, universal_newlines=True, env=env
            )
            self.assertEqual(res.returncode, 0,
                             msg=f"{blocked}: {res.stderr[-1000:]}")
            ids = ast.literal_eval(res.stdout.strip().split("\n")[-1])
            if blocked == ["moments"]:
                self.assertEqual(
                    ids,
                    [el for el in all_ids if not el.startswith("moments")]
                )
            else:
                # momi needs pandas itself
                self.assertEqual(
                    [el for el in ids if el != "momi2"],
                    [el for el in all_ids if el != "momi2"]
                )

    def test_dadi_extrapolation_in_generated_code(self):
        """
        Generated code of dadi uses the same extrapolation as the engine,
        so it gives the same likelihood.
        """
        settings = SettingsStorage()
        settings.engine = "dadi"
        settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                           "YRI_CEU.fs")
        settings.projections = [4, 4]
        settings.pts = [6, 8, 10]
        settings.initial_structure = [1, 1]
        settings.only_sudden = True
        settings.no_migrations = True
        model = settings.get_model()
        x = [0.2 if var.name.startswith("t") else 1.5
             for var in model.variables]
        ll_values = []
        for extrapolation in ["make_extrap_log_func", "make_extrap_func"]:
            engine = get_engine("dadi")
            engine.extrapolation = extrapolation
            engine.set_data(settings.inner_data)
            engine.set_model(model)
            engine.data_holder = settings.data_holder
            ll_values.append(engine.evaluate(x, settings.pts))
            code = engine.generate_code(x, None, settings.pts)
            self.assertTrue(f"dadi.Numerics.{extrapolation}(" in code,
                            msg=extrapolation)
            namespace = {}
            exec(code, namespace)
            self.assertTrue(
                np.isclose(ll_values[-1], namespace["ll_model"]),
                msg=f"{extrapolation}: {ll_values[-1]} != "
                    f"{namespace['ll_model']}"
            )
        self.assertFalse(np.isclose(ll_values[0], ll_values[1]))

    def test_data_in_generated_code(self):
        """
        Generated code prepares the data in the same way as the engine:
        population labels could be in other order than in the file or be
        only some of them, and projections are given for these labels. So
        the code has the same data and gives the same likelihood.
        """
        # file, population labels, projections, outgroup
        cases = [
            ("YRI_CEU.fs", ["CEU", "YRI"], [8, 12], True),
            ("YRI_CEU.fs", ["CEU", "YRI"], [8, 12], False),
            ("YRI_CEU_folded.fs", ["CEU", "YRI"], [8, 12], False),
            ("3d_sfs.fs", ["CEU", "CHB", "YRI"], [4, 6, 8], True),
            ("3d_sfs.fs", ["CHB", "CEU"], None, True),
            ("3d_sfs.fs", ["CEU", "CHB"], [6, 8], True),
            ("3d_sfs.fs", ["CEU", "CHB"], [6, 8], False),
        ]
        engine_ids = [engine.id for engine in all_available_engines()
                      if engine.id in ["dadi", "moments"]]
        for engine_id in engine_ids:
            for filename, labels, projections, outgroup in cases:
                msg = f"{engine_id}, {filename}, labels {labels}, " \
                      f"projections {projections}, outgroup {outgroup}"
                settings = SettingsStorage()
                settings.engine = engine_id
                settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                                   filename)
                settings.population_labels = labels
                if projections is not None:
                    settings.projections = projections
                settings.outgroup = outgroup
                settings.initial_structure = [1 for _ in labels]
                settings.only_sudden = True
                settings.no_migrations = True
                settings.read_data()
                model = settings.get_model()
                x = [0.2 if var.name.startswith("t") else 1.5
                     for var in model.variables]
                engine = get_engine(engine_id)
                engine.set_data(settings.inner_data)
                engine.set_model(model)
                engine.data_holder = settings.data_holder
                args = settings.get_engine_args()
                ll_model = engine.evaluate(x, *args)

                namespace = {}
                exec(engine.generate_code(x, None, *args), namespace)
                data = namespace["data"]
                self.assertEqual(list(data.pop_ids), labels, msg=msg)
                self.assertEqual(list(data.sample_sizes),
                                 list(engine.inner_data.sample_sizes),
                                 msg=msg)
                self.assertEqual(data.folded, not outgroup, msg=msg)
                self.assertTrue(
                    np.ma.allclose(data, engine.inner_data) and
                    np.all(data.mask == engine.inner_data.mask),
                    msg=msg
                )
                self.assertTrue(
                    np.isclose(ll_model, namespace["ll_model"]),
                    msg=f"{ll_model} != {namespace['ll_model']} : {msg}"
                )

    @pytest.mark.skipif(not gadma.momi_available, reason="No momi2")
    def test_data_in_generated_code_of_momi(self):
        """
        Generated code of momi2 reads a file that is not in SFS format of
        dadi with the same settings of data as the engine has: e.g. file in
        SNP format is read with projections. So the code has the same data
        and gives the same likelihood.
        """
        # population labels, projections, outgroup
        cases = [
            (["YRI", "CEU"], [8, 12], True),
            (["CEU", "YRI"], [8, 12], True),
            (["CEU", "YRI"], [8, 12], False),
        ]
        for labels, projections, outgroup in cases:
            msg = f"labels {labels}, projections {projections}, " \
                  f"outgroup {outgroup}"
            settings = SettingsStorage()
            settings.engine = "momi2"
            settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                               "data.txt")
            settings.population_labels = labels
            settings.projections = projections
            settings.outgroup = outgroup
            settings.mutation_rate = 1.25e-8
            settings.sequence_length = 4.04e6
            settings.initial_structure = [1, 1]
            settings.only_sudden = True
            settings.no_migrations = True
            settings.ancestral_size_as_parameter = True
            settings.read_data()
            model = settings.get_model()
            x = []
            for var in model.variables:
                if var.name == "Nanc":
                    x.append(10000)
                else:
                    x.append(0.2 if var.name.startswith("t") else 1.5)
            engine = get_engine("momi2")
            engine.set_data(settings.inner_data)
            engine.set_model(model)
            engine.data_holder = settings.data_holder
            args = settings.get_engine_args()
            ll_model = engine.evaluate(x, *args)

            code = engine.generate_code(x, None, *args)
            # we do not want to draw picture
            code = code[:code.index("from matplotlib import pyplot")]
            namespace = {}
            exec(code, namespace)
            data = namespace["data"]
            self.assertEqual(list(data.sampled_pops), labels, msg=msg)
            self.assertEqual(list(data.sampled_n), projections, msg=msg)
            self.assertEqual(data.folded, not outgroup, msg=msg)
            self.assertTrue(
                np.isclose(ll_model, namespace["ll_model"]),
                msg=f"{ll_model} != {namespace['ll_model']} : {msg}"
            )

    @pytest.mark.skipif(not gadma.momi_available, reason="No momi2")
    def test_momi_leaves_no_file_in_current_directory(self):
        """
        Neither momi2 engine nor its generated code write a file to the
        current directory when they read SFS data. By default momi writes
        `.tmp.sfs` there and reads it back, so two runs that are started in
        one directory at the same time could get data of each other.
        """
        import tempfile
        sfs_dir = os.path.abspath(os.path.join(DATA_PATH, "DATA", "sfs"))
        current_dir = os.getcwd()
        dirname = tempfile.mkdtemp()
        try:
            os.chdir(dirname)
            # engine reads file of dadi and file that is read through dadi
            for filename in ["YRI_CEU.fs", "data.txt"]:
                data = get_engine("momi2").read_data(
                    SFSDataHolder(os.path.join(sfs_dir, filename)))
                self.assertEqual(list(data.sampled_pops), ["YRI", "CEU"],
                                 msg=filename)
                self.assertEqual(os.listdir(dirname), [], msg=filename)

            # generated code
            settings = SettingsStorage()
            settings.engine = "momi2"
            settings.input_data = os.path.join(sfs_dir, "YRI_CEU.fs")
            settings.mutation_rate = 1.25e-8
            settings.sequence_length = 4.04e6
            settings.initial_structure = [1, 1]
            settings.only_sudden = True
            settings.no_migrations = True
            settings.ancestral_size_as_parameter = True
            settings.read_data()
            model = settings.get_model()
            x = []
            for var in model.variables:
                if var.name == "Nanc":
                    x.append(10000)
                else:
                    x.append(0.2 if var.name.startswith("t") else 1.5)
            engine = get_engine("momi2")
            engine.set_data(settings.inner_data)
            engine.set_model(model)
            engine.data_holder = settings.data_holder
            ll_model = engine.evaluate(x)
            code = engine.generate_code(x, None)
            # we do not want to draw picture
            code = code[:code.index("from matplotlib import pyplot")]
            namespace = {}
            exec(code, namespace)
            self.assertTrue(np.isclose(ll_model, namespace["ll_model"]),
                            msg=f"{ll_model} != {namespace['ll_model']}")
            self.assertEqual(os.listdir(dirname), [])
        finally:
            os.chdir(current_dir)
            shutil.rmtree(dirname)

    @pytest.mark.skipif(not gadma.momi_available, reason="No momi2")
    def test_values_in_generated_code_of_momi(self):
        """
        Values are written in generated code of momi2 as code of Python:
        path of the file is escaped (a quote here, backslashes of paths of
        Windows) and names of non-ascertained populations are strings.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as dirname:
            sfs_file = os.path.join(dirname, "it's_YRI_CEU.fs")
            shutil.copy(os.path.join(DATA_PATH, "DATA", "sfs", "YRI_CEU.fs"),
                        sfs_file)
            settings = SettingsStorage()
            settings.engine = "momi2"
            settings.input_data = sfs_file
            settings.non_ascertained_pops = ["CEU"]
            settings.mutation_rate = 1.25e-8
            settings.sequence_length = 4.04e6
            settings.initial_structure = [1, 1]
            settings.only_sudden = True
            settings.no_migrations = True
            settings.ancestral_size_as_parameter = True
            settings.read_data()
            model = settings.get_model()
            x = []
            for var in model.variables:
                if var.name == "Nanc":
                    x.append(10000)
                else:
                    x.append(0.2 if var.name.startswith("t") else 1.5)
            engine = get_engine("momi2")
            engine.set_data(settings.inner_data)
            engine.set_model(model)
            engine.data_holder = settings.data_holder
            ll_model = engine.evaluate(x)
            code = engine.generate_code(x, None)
            # we do not want to draw picture
            code = code[:code.index("from matplotlib import pyplot")]
            namespace = {}
            exec(code, namespace)
            self.assertTrue(np.isclose(ll_model, namespace["ll_model"]),
                            msg=f"{ll_model} != {namespace['ll_model']}")

    def test_generated_code_with_theta0_and_few_parameters(self):
        """
        Generated code is valid and gives the same likelihood as the engine
        when size of ancestral population is a parameter together with
        theta0, and when function of the model has no parameters or has only
        one parameter.
        """
        sfs_dir = os.path.join(DATA_PATH, "DATA", "sfs")
        # file, projections, structure, ancestral size as parameter, theta0
        cases = [
            ("YRI_CEU.fs", [4, 4], [1, 1], True, 0.37),
            # model has no parameters
            ("small_1pop.fs", [4], [1], False, None),
            # the only parameter is the size of ancestral population
            ("small_1pop.fs", [4], [1], True, None),
            ("small_1pop.fs", [4], [1], True, 0.37),
        ]
        engine_ids = [engine.id for engine in all_available_engines()
                      if engine.id in ["dadi", "moments"]]
        for engine_id in engine_ids:
            for filename, projections, structure, anc_size, theta0 in cases:
                msg = f"{engine_id}, structure {structure}, ancestral size " \
                      f"as parameter {anc_size}, theta0 {theta0}"
                settings = SettingsStorage()
                settings.engine = engine_id
                settings.input_data = os.path.join(sfs_dir, filename)
                settings.projections = projections
                settings.pts = [8, 10, 12]
                settings.initial_structure = structure
                settings.only_sudden = True
                settings.no_migrations = True
                settings.ancestral_size_as_parameter = anc_size
                if theta0 is not None:
                    settings.theta0 = theta0
                settings.read_data()
                model = settings.get_model()
                x = []
                for var in model.variables:
                    if var.name == "Nanc":
                        x.append(10000)
                    else:
                        x.append(0.2 if var.name.startswith("t") else 1.5)
                engine = get_engine(engine_id)
                engine.set_data(settings.inner_data)
                engine.set_model(model)
                engine.data_holder = settings.data_holder
                args = settings.get_engine_args()
                ll_model = engine.evaluate(x, *args)
                namespace = {}
                exec(engine.generate_code(x, None, *args), namespace)
                self.assertTrue(
                    np.isclose(ll_model, namespace["ll_model"]),
                    msg=f"{ll_model} != {namespace['ll_model']} : {msg}"
                )

            # function of the model has one parameter
            model = EpochDemographicModel()
            model.add_epoch(0.3, [PopulationSizeVariable("nu")])
            engine = get_engine(engine_id)
            engine.data = SFSDataHolder(
                os.path.join(sfs_dir, "small_1pop.fs"), projections=[4])
            engine.model = model
            ll_model = engine.evaluate([1.5], *args)
            namespace = {}
            exec(engine.generate_code([1.5], None, *args), namespace)
            self.assertTrue(
                np.isclose(ll_model, namespace["ll_model"]),
                msg=f"{ll_model} != {namespace['ll_model']} : {engine_id}, "
                    f"one parameter"
            )

        if gadma.momi_available:
            # one population with the size of ancestral population only
            settings = SettingsStorage()
            settings.engine = "momi2"
            settings.input_data = os.path.join(sfs_dir, "YRI_CEU.fs")
            settings.population_labels = ["YRI"]
            settings.mutation_rate = 1.25e-8
            settings.sequence_length = 4.04e6
            settings.initial_structure = [1]
            settings.ancestral_size_as_parameter = True
            settings.read_data()
            model = settings.get_model()
            self.assertEqual([var.name for var in model.variables], ["Nanc"])
            engine = get_engine("momi2")
            engine.set_data(settings.inner_data)
            engine.set_model(model)
            engine.data_holder = settings.data_holder
            ll_model = engine.evaluate([10000])
            code = engine.generate_code([10000], None)
            # we do not want to draw picture
            code = code[:code.index("from matplotlib import pyplot")]
            namespace = {}
            exec(code, namespace)
            self.assertTrue(np.isclose(ll_model, namespace["ll_model"]),
                            msg=f"{ll_model} != {namespace['ll_model']}")

    def test_generated_code_of_custom_model_without_loaded_module(self):
        """
        Code for a custom model is generated when the module with its
        function is not among loaded modules (`sys.modules`). It is so in
        the processes of runs, and code of moments was not saved in the
        directories of runs for such models.
        """
        import importlib.util
        import sys
        from gadma.code_generator.moments_ld_generator import \
            _print_momentsLD_func
        variables = [PopulationSizeVariable("nuB"),
                     PopulationSizeVariable("nuF"),
                     TimeVariable("TB"), TimeVariable("TF")]
        x = [1.5, 1.2, 0.2, 0.3]

        def get_custom_model(engine_id):
            filename = os.path.join(
                DATA_PATH, "MODELS", f"small_1pop_dem_model_{engine_id}.py")
            spec = importlib.util.spec_from_file_location(
                f"not_loaded_module_of_{engine_id}", filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(module.model_func.__module__ not in sys.modules)
            model = CustomDemographicModel(module.model_func, variables)
            return filename, model

        _, model = get_custom_model("moments")
        engine = get_engine("moments")
        engine.data = SFSDataHolder(
            os.path.join(DATA_PATH, "DATA", "sfs", "small_1pop.fs"),
            projections=[4])
        engine.model = model
        ll_model = engine.evaluate(x)
        namespace = {}
        exec(engine.generate_code(x, None), namespace)
        self.assertTrue(np.isclose(ll_model, namespace["ll_model"]),
                        msg=f"{ll_model} != {namespace['ll_model']}")

        if gadma.moments_LD_available:
            filename, model = get_custom_model("momentsLD")
            engine = get_engine("momentsLD")
            engine.model = model
            engine.data_holder = VCFDataHolder(
                vcf_file=os.path.join(DATA_PATH, "DATA", "vcf_ld",
                                      "vcf_data.vcf"),
                popmap_file=os.path.join(DATA_PATH, "DATA", "vcf_ld",
                                         "pop_map.txt"),
            )
            namespace = {}
            exec(_print_momentsLD_func(engine, x), namespace)
            # it is the function of our file
            code_file = namespace["model_func"].__code__.co_filename
            self.assertTrue(os.path.samefile(code_file, filename))

    def test_selection_and_dominance_of_each_population(self):
        """
        Selection and dominance coefficients of every population are used
        during simulation, not of the last population only.
        """
        nu1, nu2, nu3 = [PopulationSizeVariable(f'nu{i}') for i in (1, 2, 3)]
        t1, t2 = TimeVariable('t1'), TimeVariable('t2')
        gammas = [SelectionVariable(f'gamma{i}') for i in (1, 2, 3)]
        doms = [FractionVariable(f'h{i}') for i in (1, 2, 3)]

        # 1. Arguments for dadi integration functions
        dadi_engine = get_engine('dadi')
        dm = EpochDemographicModel()
        dm.add_epoch(t1, [nu1], sel_args=gammas[:1], dom_args=doms[:1])
        kwargs = dadi_engine._get_kwargs(dm.events[-1], {})
        self.assertTrue(kwargs['gamma'] is gammas[0])
        self.assertTrue(kwargs['h'] is doms[0])
        dm.add_split(0, [nu1, nu2])
        dm.add_epoch(t1, [nu1, nu2], sel_args=gammas[:2], dom_args=doms[:2])
        dm.add_split(1, [nu2, nu3])
        dm.add_epoch(t2, [nu1, nu2, nu3], sel_args=gammas, dom_args=doms)
        for event, n_pop in [(dm.events[-3], 2), (dm.events[-1], 3)]:
            kwargs = dadi_engine._get_kwargs(event, {})
            for i in range(n_pop):
                self.assertTrue(kwargs[f'gamma{i + 1}'] is gammas[i],
                                msg=f"{n_pop} populations: {kwargs}")
                self.assertTrue(kwargs[f'h{i + 1}'] is doms[i],
                                msg=f"{n_pop} populations: {kwargs}")
            self.assertEqual(
                len([key for key in kwargs if key[0] in "gh"]), 2 * n_pop
            )

        # 2. Two populations that are the same except selection. When
        # coefficients of populations are swapped, SFS is transposed.
        dm = EpochDemographicModel()
        dm.add_split(0, [nu1, nu2])
        dm.add_epoch(t1, [nu1, nu2], sel_args=gammas[:2], dom_args=doms[:2])
        ns = (6, 6)

        def simulate(engine, args, **new_values):
            values = {'nu1': 1.0, 'nu2': 1.0, 't1': 0.3, 'gamma1': 0.0,
                      'gamma2': 0.0, 'h1': 0.5, 'h2': 0.5}
            values.update(new_values)
            sfs = engine.simulate([values[var.name] for var in dm.variables],
                                  ns, None, None, *args)
            return np.ma.filled(sfs, 0.0)

        total = {}
        for engine_id, args in [('dadi', ([20, 30, 40],)), ('moments', ())]:
            engine = get_engine(engine_id)
            engine.set_model(dm)
            neutral = simulate(engine, args)
            for name, values in [('gamma', (5.0, 0.0)), ('h', (0.1, 0.9))]:
                msg = f"engine {engine_id}, {name} {values}"
                other = {}
                if name == 'h':
                    other = {'gamma1': 5.0, 'gamma2': 5.0}
                sfs = simulate(engine, args, **other,
                               **{f"{name}1": values[0],
                                  f"{name}2": values[1]})
                sfs_swapped = simulate(engine, args, **other,
                                       **{f"{name}1": values[1],
                                          f"{name}2": values[0]})
                self.assertTrue(np.allclose(sfs, sfs_swapped.T, rtol=1e-2),
                                msg=msg)
                self.assertFalse(np.allclose(sfs, sfs_swapped, rtol=1e-2),
                                 msg=msg)
                self.assertFalse(np.allclose(sfs, neutral, rtol=1e-2),
                                 msg=msg)
                total[(engine_id, name)] = sfs.sum()
        # two engines agree
        for name in ['gamma', 'h']:
            self.assertTrue(
                np.isclose(total[('dadi', name)], total[('moments', name)],
                           rtol=3e-2),
                msg=f"{name}: dadi {total[('dadi', name)]}, "
                    f"moments {total[('moments', name)]}"
            )

    def test_sfs_of_evaluation_is_used_in_comp_plot(self):
        """
        Best model is drawn right after its evaluation. SFS for comparison
        plot should be taken from that evaluation and not simulated again.
        """
        engine = get_engine('dadi')
        ns = (4, 4, 4)
        pts = [6, 10, 14]
        data, dm, values, ll = self.model_3pop_dadi(ns, pts)
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_data(data)
        engine.set_model(dm)

        simulate = engine.simulate
        n_calls = []

        def simulate_and_count(*args, **kwargs):
            n_calls.append(1)
            return simulate(*args, **kwargs)

        engine.simulate = simulate_and_count
        self.assertEqual(engine.evaluate(vals, pts), ll)
        self.assertEqual(len(n_calls), 1)
        engine.draw_data_comp_plot(vals, pts, save_file="plot.png")
        self.assertEqual(len(n_calls), 1)
        # values that were not evaluated are simulated: the first value is
        # changed so that it stays in the domain of its variable
        other_vals = list(vals)
        other_vals[0] = 1.5 * vals[0]
        if not dm.variables[0].correct_value(other_vals[0]):
            other_vals[0] = vals[0] / 1.5
        self.assertTrue(dm.variables[0].correct_value(other_vals[0]))
        engine.draw_data_comp_plot(other_vals, pts, save_file="plot.png")
        self.assertEqual(len(n_calls), 2)
        # saved SFS is not valid for new model or data
        engine.set_model(dm)
        engine.draw_data_comp_plot(vals, pts, save_file="plot.png")
        self.assertEqual(len(n_calls), 3)

    def test_sfs_in_comp_plot_is_scaled_as_for_likelihood(self):
        """
        Expected SFS in the plot of comparison with data is scaled by the
        same theta as for the likelihood: by the optimal one, by the one
        that is restricted by bounds of the split time or by the one that
        corresponds to the size of ancestral population.
        """
        import tempfile
        Nanc = 12000.
        out_dir = tempfile.mkdtemp("gadma_test_dir")
        cases = ["implicit", "implicit with bound", "parameter", "fixed"]
        try:
            for engine_id, case in [(engine_id, case)
                                    for engine_id in ["dadi", "moments"]
                                    for case in cases]:
                msg = f"engine: {engine_id}, ancestral size: {case}"
                settings = SettingsStorage()
                settings.engine = engine_id
                settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                                   "YRI_CEU.fs")
                settings.projections = [4, 4]
                if engine_id == "dadi":
                    settings.pts = [8, 10, 12]
                settings.mutation_rate = 1.25e-8
                settings.sequence_length = 4.04e6
                settings.initial_structure = [1, 1]
                settings.only_sudden = True
                settings.no_migrations = True
                if case in ["parameter", "fixed"]:
                    settings.ancestral_size_as_parameter = True
                if case == "fixed":
                    settings.fixed_ancestral_size = Nanc
                if case == "implicit with bound":
                    settings.upper_bound_of_first_split = 1000

                model = settings.get_model()
                engine = get_engine(engine_id)
                engine.set_data(settings.inner_data)
                engine.set_model(model)
                engine.data_holder = settings.data_holder
                args = settings.get_engine_args()
                x = [{"Nanc": Nanc, "t1": 0.2}.get(var.name, 1.5)
                     for var in model.variables]

                module = engine.base_module
                ll = engine.evaluate(x, *args)
                sfs = engine.simulate(x, engine.data.sample_sizes, None, None,
                                      *args)
                ll_optimal = module.Inference.ll_multinom(sfs, engine.data)
                if case == "implicit":
                    self.assertTrue(np.isclose(ll, ll_optimal), msg=msg)
                else:
                    # theta is not optimal in other cases
                    self.assertTrue(ll < ll_optimal - 1,
                                    msg=f"{ll}, {ll_optimal}. {msg}")

                # SFS that is drawn
                drawn = []
                n_calls = []
                plot_func = module.Plotting.plot_2d_comp_Poisson
                simulate = engine.simulate

                def save_model(model, data, **kwargs):
                    drawn.append(model)

                def simulate_and_count(*args, **kwargs):
                    n_calls.append(1)
                    return simulate(*args, **kwargs)

                module.Plotting.plot_2d_comp_Poisson = save_model
                engine.simulate = simulate_and_count
                try:
                    engine.draw_data_comp_plot(
                        x, *args, save_file=os.path.join(out_dir, "plot.png")
                    )
                finally:
                    module.Plotting.plot_2d_comp_Poisson = plot_func
                self.assertEqual(len(drawn), 1, msg=msg)
                ll_drawn = module.Inference.ll(drawn[0], engine.data)
                self.assertTrue(np.isclose(ll, ll_drawn),
                                msg=f"{ll}, {ll_drawn}. {msg}")
                # SFS of the evaluation is used, it is not simulated again
                self.assertEqual(len(n_calls), 0, msg=msg)
        finally:
            shutil.rmtree(out_dir)

    def test_moments_engine(self):
        engine = get_engine('moments')

        ns = (4, 4, 4, 4)
        dt_fac = 0.1

        data, dm, values, ll = self.model_4pop_moments(ns, dt_fac)
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]

        engine.set_data(data)
        engine.set_model(dm)
        seq_len = 1e6
        pops = None
        model = engine.simulate(vals, ns, seq_len, pops, dt_fac)
        self.assertTrue(np.allclose(model, data), msg="Simulations differs in"
                                                      " engine and in dadi.")
        eng_ll = engine.set_and_evaluate(
            data=data,
            model=dm,
            values=vals,
            options={"dt_fac": dt_fac}
        )
        self.assertTrue(np.allclose(eng_ll, ll),
                        msg=f"{eng_ll} != {ll}")
        # check that constrain is used when ancestral size is known (not
        # multinom): solution that does not satisfy constrain is skipped
        dm = EpochDemographicModel(Nanc_size=10000)
        dm.add_epoch(TimeVariable("t"), [PopulationSizeVariable("nu")])
        engine.model = dm
        engine.data = engine.simulate(
            values=[1, 1],
            ns=(10,),
            sequence_length=1e6,
            population_labels=["Pop1"]
        )
        ll = engine.evaluate(values=[1, 1])
        self.assertTrue(ll is not None)
        # time is multiplied by ancestral size and compared with bounds
        A = [[int(isinstance(var, TimeVariable)) for var in dm.variables]]
        engine.model.linear_constrain = optimizers.LinearConstrain(
            A, [5000], [20000])
        self.assertEqual(engine.evaluate(values=[1, 1]), ll)
        engine.model.linear_constrain = optimizers.LinearConstrain(
            A, [0], [5000])
        self.assertTrue(engine.evaluate(values=[1, 1]) is None)
        engine.model.linear_constrain = optimizers.LinearConstrain(
            A, [20000], [np.inf])
        self.assertTrue(engine.evaluate(values=[1, 1]) is None)

    def test_upper_bounds_on_split_time(self):
        import dadi
        settings = SettingsStorage()
        settings.initial_structure = [2, 1, 1]
        settings.upper_bound_of_first_split = 100
        settings.upper_bound_of_second_split = 50

        data = dadi.Spectrum.from_file(os.path.join(DATA_PATH, "DATA",
                                                    "sfs", "3d_sfs.fs"))
        engine = get_engine("moments")
        engine.set_data(data)
        engine.set_model(settings.get_model())

        self.assertRaises(AttributeError, engine.generate_code, [], "filename")

        values = [var.resample() for var in engine.model.variables]
        engine.get_theta(values)

        engine.model.linear_constrain.lb = [51, -1]
        engine.set_model(engine.model)
        self.assertRaises(ValueError, engine.get_theta, values)

    def test_bounds_of_split_time_with_known_ancestral_size(self):
        """
        Bounds of split times are checked when size of ancestral population
        is parameter of the model or constant: solution that is out of bounds
        is skipped, other solutions have the same value as without bounds.
        """
        engine_ids = ["moments", "dadi"]
        if gadma.momi_available:
            engine_ids.append("momi2")
        for engine_id, is_fixed in [(engine_id, is_fixed)
                                    for engine_id in engine_ids
                                    for is_fixed in [False, True]]:
            if engine_id == "momi2" and is_fixed:
                continue
            msg = f"engine: {engine_id}, fixed ancestral size: {is_fixed}"
            settings = SettingsStorage()
            settings.engine = engine_id
            settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                               "YRI_CEU.fs")
            settings.projections = [4, 4]
            if engine_id == "dadi":
                settings.pts = [8, 10, 12]
            settings.mutation_rate = 1.25e-8
            settings.sequence_length = 4.04e6
            settings.initial_structure = [1, 1]
            settings.only_sudden = True
            settings.no_migrations = True
            settings.ancestral_size_as_parameter = True
            if is_fixed:
                settings.fixed_ancestral_size = 10000
            settings.lower_bound_of_first_split = 2000
            settings.upper_bound_of_first_split = 6000

            model = settings.get_model()
            names = [var.name for var in model.variables]
            self.assertEqual("Nanc" in names, not is_fixed, msg=msg)
            # constrain is built for the variables of the model
            constrain = model.linear_constrain
            self.assertTrue(constrain is not None, msg=msg)
            self.assertTrue(
                np.all(constrain.A == [[name == "t1" for name in names]]),
                msg=f"{constrain}. {msg}"
            )
            engine = get_engine(engine_id)
            engine.set_data(settings.inner_data)
            engine.set_model(model)
            engine.data_holder = settings.data_holder
            args = settings.get_engine_args()

            # Ancestral size is 10000, so split is 20000 * t1 generations ago
            for time, is_inside in [(0.05, False), (0.2, True), (0.5, False)]:
                x = [{"Nanc": 10000., "t1": time}.get(name, 1.0)
                     for name in names]
                ll = engine.evaluate(x, *args)
                model.linear_constrain = None
                ll_without_bounds = engine.evaluate(x, *args)
                model.linear_constrain = constrain
                self.assertTrue(ll_without_bounds is not None, msg=msg)
                if is_inside:
                    self.assertEqual(ll, ll_without_bounds, msg=msg)
                else:
                    self.assertTrue(ll is None,
                                    msg=f"time {time}, value {ll}. {msg}")
                # size of ancestral population is available for all values
                self.assertEqual(engine.get_N_ancestral(x, *args), 10000,
                                 msg=msg)

            # Optimizers get the same constrain for values of the variables,
            # size of ancestral population is its factor. Solution that is
            # out of bounds is moved to the bound: time is scaled.
            opt_constrain = engine.get_linear_constrain_for_optimizers()
            self.assertTrue(opt_constrain is not None, msg=msg)
            if is_fixed:
                self.assertEqual(opt_constrain.factor, 10000, msg=msg)
                self.assertEqual(opt_constrain.factor_index, None, msg=msg)
            else:
                self.assertEqual(opt_constrain.factor_index,
                                 names.index("Nanc"), msg=msg)
            for Nanc, time, new_time in [(10000., 0.05, 0.1),
                                         (10000., 0.2, 0.2),
                                         (10000., 0.5, 0.3),
                                         (5000., 0.5, 0.5),
                                         (5000., 0.1, 0.2),
                                         (50000., 0.1, 0.06)]:
                if is_fixed and Nanc != 10000:
                    continue
                x = [{"Nanc": Nanc, "t1": time}.get(name, 1.0)
                     for name in names]
                is_inside = (time == new_time)
                self.assertEqual(engine.fits_linear_constrain(x), is_inside,
                                 msg=f"{x}. {msg}")
                self.assertEqual(opt_constrain.fits(x), is_inside,
                                 msg=f"{x}. {msg}")
                x_tr, success = opt_constrain.try_to_transform(x)
                self.assertTrue(success, msg=f"{x}. {msg}")
                x_expected = [{"Nanc": Nanc, "t1": new_time}.get(name, 1.0)
                              for name in names]
                self.assertTrue(np.allclose(x_tr, x_expected),
                                msg=f"{x_tr}. {msg}")
                self.assertTrue(engine.fits_linear_constrain(x_tr),
                                msg=f"{x_tr}. {msg}")
                self.assertTrue(engine.evaluate(list(x_tr), *args)
                                is not None, msg=f"{x_tr}. {msg}")
            # The same answer as engine has for random values
            np.random.seed(11)
            for _ in range(50):
                x = [var.resample() for var in model.variables]
                self.assertEqual(opt_constrain.fits(x),
                                 engine.fits_linear_constrain(x),
                                 msg=f"{x}. {msg}")

        # If size of ancestral population is not known from the model then
        # engine deals with bounds itself, optimizers do not get constrain
        settings = SettingsStorage()
        settings.engine = "moments"
        settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                           "YRI_CEU.fs")
        settings.mutation_rate = 1.25e-8
        settings.sequence_length = 4.04e6
        settings.initial_structure = [1, 1]
        settings.upper_bound_of_first_split = 6000
        model = settings.get_model()
        self.assertTrue(model.linear_constrain is not None)
        engine = get_engine("moments")
        engine.set_data(settings.inner_data)
        engine.set_model(model)
        self.assertTrue(engine.get_linear_constrain_for_optimizers() is None)
        settings.ancestral_size_as_parameter = True
        model = settings.get_model()
        engine.set_model(model)
        self.assertTrue(
            engine.get_linear_constrain_for_optimizers() is not None)
        model.linear_constrain = None
        self.assertTrue(engine.get_linear_constrain_for_optimizers() is None)

    def test_fixed_ancestral_size(self):
        """
        Likelihood of the model with fixed size of ancestral population is
        equal to the likelihood of the model where this size is a parameter
        with the same value. Generated code gives the same likelihood.
        """
        Nanc = 10000
        engine_ids = ["moments", "dadi"]
        if gadma.momi_available:
            engine_ids.append("momi2")
        for engine_id in engine_ids:
            ll_values = {}
            for is_fixed in [False, True]:
                msg = f"engine: {engine_id}, fixed ancestral size: {is_fixed}"
                settings = SettingsStorage()
                settings.engine = engine_id
                settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                                   "YRI_CEU.fs")
                settings.projections = [4, 4]
                if engine_id == "dadi":
                    settings.pts = [8, 10, 12]
                settings.mutation_rate = 1.25e-8
                settings.sequence_length = 4.04e6
                # Sizes of populations depend on the size of ancestral
                # population in this model: size is changed from it during
                # the first interval and populations are its fractions
                # right after the split.
                settings.initial_structure = [2, 1]
                settings.split_fractions = True
                settings.no_migrations = True
                settings.ancestral_size_as_parameter = True
                if is_fixed:
                    settings.fixed_ancestral_size = Nanc
                model = settings.get_model()
                names = [var.name for var in model.variables]
                self.assertEqual("Nanc" in names, not is_fixed, msg=msg)

                # the same values for both models
                x = []
                n_sizes = 0
                for var in model.variables:
                    if var.name == "Nanc":
                        x.append(Nanc)
                    elif isinstance(var, DynamicVariable):
                        x.append("Exp")
                    elif isinstance(var, TimeVariable):
                        x.append(0.4)
                    elif isinstance(var, FractionVariable):
                        x.append(0.3)
                    else:
                        self.assertTrue(
                            isinstance(var, PopulationSizeVariable),
                            msg=f"{var}. {msg}"
                        )
                        n_sizes += 1
                        x.append(0.5 * n_sizes)
                engine = get_engine(engine_id)
                engine.set_data(settings.inner_data)
                engine.set_model(model)
                engine.data_holder = settings.data_holder
                args = settings.get_engine_args()
                ll_values[is_fixed] = engine.evaluate(x, *args)
                self.assertTrue(ll_values[is_fixed] is not None, msg=msg)

                # generated code gives the same value
                code = engine.generate_code(x, None, *args)
                if engine_id == "momi2":
                    # we do not want to draw picture
                    code = code[:code.index("from matplotlib import pyplot")]
                namespace = {}
                exec(code, namespace)
                self.assertTrue(
                    np.isclose(ll_values[is_fixed], namespace["ll_model"]),
                    msg=f"{ll_values[is_fixed]} != {namespace['ll_model']}"
                        f" : {msg}"
                )

                # Size of ancestral population is equal to 1 in genetic
                # units, so simulation is the same for values of the model
                # and for values translated to genetic units
                if engine_id != "momi2" and not is_fixed:
                    x_gen = model.translate_values(units="genetic", values=x)
                    self.assertEqual(x_gen[0], 1.0, msg=msg)
                    ns = engine.data.sample_sizes
                    self.assertTrue(
                        np.allclose(engine.simulate(x, ns, None, None, *args),
                                    engine.simulate(x_gen, ns, None, None,
                                                    *args)),
                        msg=msg
                    )
            self.assertTrue(np.isclose(ll_values[False], ll_values[True]),
                            msg=f"{ll_values}, engine: {engine_id}")

    def test_demes_engine(self):
        data_holder = DataHolder(
            filename=None,
            projections=[10, 10, 10],
            outgroup=False,
            population_labels=["YRI", "CEU", "CHB"],
            sequence_length=None
        )

        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        m13 = MigrationVariable('m13')
        Tp = TimeVariable('Tp', domain=[1e-3, 5])
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')

        dm = EpochDemographicModel(Nanc_size=1000)
        dm.add_epoch(T, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(Tp, [nu1F, nu2F], dyn_args=['Sud', Dyn])
        dm.add_split(1, [nu2F, nu2F])
        migs = [[0, m, m13], [m, 0, 0], [0, 0, 0]]
        dm.add_epoch(T, [nu1F, nu2F, nu2F], mig_args=migs)

        values = {'nu1F': 1.880, nu2B: 0.0724, 'f': 0.9, 'nu2F': 1.764,
                  'm': 0.930, 'Tp': 0.363, 'T': 0.112, 'Dyn': 'Lin',
                  'SudDyn': 'Sud', 's': 0.1, 'dom': 0.5, 'm13': 1.5}

        engine = gadma.engines.demes_engine.DemesEngine()
        engine.data = data_holder
        engine.model = dm

        engine.generate_code(values=values)
        engine.draw_schematic_model_plot(values, save_file="plot.png")

        engine.generate_code(values=values,
                             gen_time=25, gen_time_units="years")
        engine.draw_schematic_model_plot(values, save_file="plot.png",
                                         gen_time=25, gen_time_units="years")

        # now check for genetic units
        engine.generate_code(values=values, gen_time=0.5,
                             gen_time_units="genetic units")
        engine.draw_schematic_model_plot(values, save_file="plot.png",
                                         gen_time=0.5, gen_time_units="genetic units")

        # check that demographic units are genetic if we do not set nanc
        dm.has_anc_size = False
        gr = engine.build_demes_graph(values=values, nanc=None, gen_time_units="generations")
        self.assertEqual(gr.time_units, "genetic units")

        dm = EpochDemographicModel(Nanc_size=1000)
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(Tp, [nu1F, nu2F], dyn_args=['Sud', Dyn])
        dm.add_split(1, [nu2F, nu2F])
        migs = [[0, m, m13], [m, 0, 0], [0, 0, 0]]
        dm.add_epoch(T, [nu1F, nu2F, nu2F], mig_args=migs)

        engine.model = dm
        engine.generate_code(values=values,
                             gen_time=25, gen_time_units="years")
        engine.draw_schematic_model_plot(values, save_file="plot.png",
                                         gen_time=25, gen_time_units="years")
        engine.draw_schematic_model_plot(values, save_file=None,
                                         gen_time=25, gen_time_units="years")

        # error when we give model that has several ancestor populations
        # data still has three populations
        dm = EpochDemographicModel()
        engine.model = dm
        self.assertRaises(AssertionError, engine.build_demes_graph,
                          values=values, nanc=None)

    def test_moments_drawing(self):
        data_holder = SFSDataHolder(
            os.path.join(DATA_PATH, "DATA", "sfs", "YRI_CEU.fs"),
        )

        engine = get_engine("moments")

        nu = PopulationSizeVariable('nu')
        T = TimeVariable('T', domain=[1e-3, 5])
        Dyn = DynamicVariable('Dyn')

        dm = EpochDemographicModel(Nanc_size=1000)
        dm.add_split(0, [nu, nu])
        dm.add_epoch(T, [nu, nu], dyn_args=['Sud', Dyn])

        values = {'nu': 1.880, 'T': 0.112, 'Dyn': 'Exp'}

        engine.data = engine.read_data(data_holder)
        assert engine.data_holder is None
        engine.model = dm

        engine.draw_schematic_model_plot(values, save_file="plot.png")
        engine.draw_data_comp_plot(values=values, save_file=None)

        dm = EpochDemographicModel()
        dm.add_epoch(T, [nu])
        engine.model = dm
        engine.data = engine.simulate(
            values=values,
            ns=[20],
            sequence_length=1e6,
            population_labels=["Pop1"]
        )
        engine.draw_data_comp_plot(values=values, save_file=None)

    def get_data_holder(self):
        # we take the most common data
        vcf_path = os.path.join(DATA_PATH, "DATA", "vcf")
        data_holder = VCFDataHolder(
            os.path.join(vcf_path, "out_of_africa_chr22_sim.vcf"),
            popmap_file=os.path.join(vcf_path, "out_of_africa_chr22_sim_3pop.popmap"),
            projections=[4, 4, 4],
            population_labels=["YRI", "CEU", "CHB"],
            sequence_length=50818468,  # chr22
        )
        return data_holder

    def get_model(self):
        dm = StructureDemographicModel(
            initial_structure=[2, 1, 1],
            final_structure=[2, 1, 1],
            has_anc_size=True,
            has_migs=False,
            has_sels=False,
            has_dom=False,
            has_dyns=True,
            sym_migs=False,
            frac_split=False,
            has_p_misid=False,
            has_inbr=False,
        )
        dm.mutation_rate = 1e-8
        dm.recombination_rate = 1e-8
        return dm

    @pytest.mark.skipif(not gadma.demes_available, reason="No demes")
    def test_demes_engine_genetic_units(self):
        """
        In genetic units size of ancestral population is equal to 1 and time
        is the same as values of time parameters.
        """
        for sym_migs in [False, True]:
            dm = StructureDemographicModel(
                initial_structure=[1, 1],
                final_structure=[1, 1],
                has_migs=True,
                has_sels=False,
                has_dom=False,
                has_dyns=False,
                sym_migs=sym_migs,
                frac_split=True,
                has_p_misid=False,
            )
            values = {"s1": 0.5, "t1": 0.3, "nu11": 1.5, "nu12": 0.8,
                      "m1_12": 0.5, "m1_21": 0.2}
            x = [values[var.name] for var in dm.variables]
            engine = get_engine("demes")
            engine.set_model(dm)

            # 1. Units are genetic if there is no size of ancestral
            # population or if they are required
            graphs = [
                engine.build_demes_graph(x),
                engine.build_demes_graph(x, gen_time_units="genetic units"),
                engine.build_demes_graph(x, nanc=1.0, gen_time=0.5,
                                         gen_time_units="genetic units"),
                engine.build_demes_graph(x, nanc=12345, gen_time=25,
                                         gen_time_units="genetic units"),
            ]
            for graph in graphs:
                self.assertEqual(graph.time_units, "genetic units")
                ancestral = graph["ancestral"]
                # time of split is equal to t1
                self.assertTrue(np.isclose(ancestral.end_time, 0.3),
                                msg=f"Time of split: {ancestral.end_time}")
                self.assertTrue(
                    np.isclose(ancestral.epochs[0].start_size, 1.0)
                )
                self.assertTrue(
                    np.isclose(graph["pop0"].epochs[0].start_size, 1.5)
                )
                self.assertTrue(
                    np.isclose(graph["pop1"].epochs[0].start_size, 0.8)
                )
                self.assertEqual(graphs[0].asdict(), graph.asdict())

            # 2. Migration rates that are greater than 1 are divided, each
            # rate is divided one time
            values["m1_12"] = 5.0
            x = [values[var.name] for var in dm.variables]
            for kwargs in [{}, {"gen_time_units": "genetic units"}]:
                graph = engine.build_demes_graph(x, **kwargs)
                self.assertTrue("divided by 100" in graph.description)
                rates = sorted([mig.rate for mig in graph.migrations])
                if sym_migs:
                    expected = [5.0 / 2 / 100, 5.0 / 2 / 100]
                else:
                    expected = [0.2 / 2 / 100, 5.0 / 2 / 100]
                self.assertTrue(np.allclose(rates, expected),
                                msg=f"{rates} != {expected}")

            # 3. Time is in generations if size of ancestral population is
            # given and other units are not required
            values["m1_12"] = 0.5
            x = [values[var.name] for var in dm.variables]
            graph = engine.build_demes_graph(x, nanc=1000)
            self.assertEqual(graph.time_units, "generations")
            self.assertTrue(np.isclose(graph["ancestral"].end_time,
                                       2 * 1000 * 0.3))

    def test_drawing_engines(self):
        data_holder = self.get_data_holder()
        dm = self.get_model()

        values = []
        for var in dm.variables:
            if isinstance(var, ContinuousVariable):
                if var.units == "physical":
                    values.append(10000)
                else:
                    values.append(np.random.uniform(0.5, 1.5))
            else:
                values.append(np.random.choice(["Sud", "Exp"]))

        for engine in all_available_engines():
            if not engine.can_draw_comp:
                continue
            if engine.id == "momentsLD":
                # drawing test for momentsLD is separate
                continue
            engine.model = dm
            engine.data = data_holder
            kwargs = {}
            if engine.id == "dadi":
                kwargs["pts"] = [5, 10, 15]  # pts

            engine.draw_data_comp_plot(values, **kwargs, save_file=None)

            if engine.can_simulate:
                engine.data = engine.simulate(
                    values=values,
                    ns=engine.data_holder.projections,
                    sequence_length=engine.data_holder.sequence_length,
                    population_labels=engine.data_holder.population_labels,
                    **kwargs,
                )
                engine.draw_data_comp_plot(values, **kwargs, save_file=None)

        for engine in all_available_engines():
            if not engine.can_draw_model:
                continue
            engine.model = dm
            engine.data_holder = data_holder
            engine.draw_schematic_model_plot(
                values=values,
                gen_time=10,
                gen_time_units='Years',
                save_file="plot.png"
            )

    def test_evaluating_engines(self):
        data_holder = self.get_data_holder()
        dm = self.get_model()

        values = [var.resample() for var in dm.variables]
        values = [el if el != "Lin" else "Exp" for el in values]

        for engine in all_available_engines():
            if engine.id == "momentsLD":
                # Evaluation test for momentsLD is separate
                continue
            kwargs = {}
            if engine.id == "dadi":
                kwargs["pts"] = [5, 10, 15]  # pts

            if engine.can_evaluate:
                engine.data = data_holder
                engine.model = dm
                engine.evaluate(values, **kwargs)


class TestModelSimulation(unittest.TestCase):

    def tearDown(self):
        if Path(f"{TEST_OUTPUT}/_bed_files/").exists():
            shutil.rmtree(f"{TEST_OUTPUT}/_bed_files/")

    def model_one_pop_moments_ld(self):
        import moments.LD

        def moments_one_pop_func(params, rho, theta):
            nu, tf = params

            ld = moments.LD.Numerics.steady_state(nus=[1], rho=rho, theta=theta)
            ld_stats = moments.LD.LDstats(ld, num_pops=1)
            ld_stats.integrate(tf=tf, nu=[nu], rho=rho, theta=theta)

            return ld_stats

        nu = PopulationSizeVariable('nu')
        tf = TimeVariable('tf')
        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5

        values = {'nu': nu.resample(),
                  'tf': tf.resample()}

        data = moments.LD.Inference.bin_stats(
            model_func=moments_one_pop_func,
            params=[values[x] for x in values],
            rho=rhos,
            theta=theta
        )
        data = moments.LD.Inference.sigmaD2(data)

        dm = EpochDemographicModel()
        dm.add_epoch(tf, [nu])

        return values, data, dm

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_one_pop_moments_ld(self):
        engine = get_engine('momentsLD')
        values, data, dm = self.model_one_pop_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_data(data)
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        model = engine.simulate(vals)

        self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
                                                            'engine and momentsLD')
        self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
                                                            'engine and momentsLD')

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def model_two_pops_moments_ld(self):
        import moments.LD

        def moments_two_pops_func(params, rho, theta):
            nu1, nu2, tf = params

            ld = moments.LD.Numerics.steady_state(nus=[1], rho=rho, theta=theta)
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
        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5
        data = moments.LD.Inference.bin_stats(
            model_func=moments_two_pops_func,
            params=[values[x] for x in list_for_moments_ld],
            rho=rhos,
            theta=theta
        )
        data = moments.LD.Inference.sigmaD2(data)

        dm = EpochDemographicModel()
        dm.add_epoch(t, [nu1])
        dm.add_split(0, [nu1, nu1])
        dm.add_epoch(t, [nu1, nu2], dyn_args=['Sud', dyn])

        return values, data, dm

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_two_pops_moments_ld(self):
        engine = get_engine('momentsLD')
        values, data, dm = self.model_two_pops_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.data_holder = DATA_HOLDER_FOR_MODELS
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        model = engine.simulate(vals)
        self.assertTrue(np.allclose(model[0], data[0]), msg='Simulations differs in '
                                                            'engine and momentsLD')
        self.assertTrue(np.allclose(model[1], data[1]), msg='Simulations differs in '
                                                            'engine and momentsLD')

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def model_4pops_moments_ld(self):
        import moments.LD

        def moments_ld_func(params, rho, theta):
            nu1, nu2, nu3, nu4, tf, m12 = params

            ld = moments.LD.Numerics.steady_state(nus=[1], rho=rho, theta=theta)
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

        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5

        simulated = moments.LD.Inference.bin_stats(
            model_func=moments_ld_func,
            params=[values[x] for x in list_for_moments_ld],
            rho=rhos,
            theta=theta
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

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def test_4pops_moments_ld(self):
        values, simulated_by_moments, dm = self.model_4pops_moments_ld()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine = get_engine('momentsLD')
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
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


@pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
class TestMomentsLDCode(unittest.TestCase):
    """
    Generated code of momentsLD gives the same likelihood as the engine.
    """
    def setUp(self):
        import tempfile
        self.dirname = tempfile.mkdtemp()

    def tearDown(self):
        moments.LD.Inference._varcov_inv_cache = {}
        shutil.rmtree(self.dirname, ignore_errors=True)

    def get_settings(self, vcf_file, popmap, structure, lengths=1000000,
                     preprocessed_data=None, split_fractions=True):
        import tempfile
        settings = SettingsStorage()
        settings.output_directory = os.path.join(
            tempfile.mkdtemp(dir=self.dirname), "out")
        settings.input_data = f"{vcf_file}, {popmap}"
        settings.engine = "momentsLD"
        settings.sequence_length = lengths
        settings.mutation_rate = 2.35e-8
        settings.recombination_rate = 1e-8
        settings.ld_kwargs = {"report": False}
        settings.ancestral_size_as_parameter = True
        settings.initial_structure = structure
        settings.final_structure = structure
        settings.split_fractions = split_fractions
        if preprocessed_data is not None:
            settings.preprocessed_data = preprocessed_data
        return settings

    def run_script(self, filename):
        with open(filename) as fl:
            code = fl.read()
        # we do not want to draw LD curves
        code = code[:code.index("stats_to_plot = [")]
        namespace = {}
        moments.LD.Inference._varcov_inv_cache = {}
        exec(code, namespace)
        return namespace

    def test_code_for_data_computed_in_run(self):
        """
        Code is generated when LD statistics were computed during the run:
        it computes them for the same regions with the same arguments.
        """
        settings = self.get_settings(VCF_DATA_LD, POP_MAP, [1, 1])
        # bootstrap of statistics is random, the same seed gives the same
        np.random.seed(1)
        data = settings.read_data()
        model = settings.get_model()
        x = [10000 if var.name == "Nanc" else
             ("Exp" if var.name.startswith("dyn") else var.resample())
             for var in model.variables]
        engine = get_engine("momentsLD")
        engine.set_data(data)
        engine.data_holder = settings.data_holder
        engine.set_model(model)
        ll_model = engine.evaluate(x)

        filename = os.path.join(self.dirname, "code.py")
        engine.generate_code(x, filename, None, 10000)
        np.random.seed(1)
        namespace = self.run_script(filename)
        self.assertTrue(np.isclose(ll_model, namespace["ll_model"]),
                        msg=f"{ll_model} != {namespace['ll_model']}")
        # file for confidence intervals says that data is not saved
        ci_file = os.path.join(self.dirname, "codedata_for_CI.py")
        with open(ci_file) as fl:
            self.assertTrue("data_file = None" in fl.read())

    def test_code_with_sudden_children_of_split(self):
        """
        When both populations after a split change their size suddenly,
        their sizes right after the split do not change the model. They are
        not parameters of the code, but the code gives the same likelihood.
        """
        vcf_file = os.path.join(DATA_PATH, "DATA", "vcf_ld", "fake_data.vcf")
        popmap = os.path.join(DATA_PATH, "DATA", "vcf_ld",
                              "fake_data_pop_map.txt")
        preprocessed_data = os.path.join(DATA_PATH, "DATA", "vcf_ld",
                                         "preprocessed_data_model1.bp")
        lengths = {str(i): 200000 for i in range(1, 21)}
        # dynamics: only the populations after the second split are sudden
        cases = [(True, ["Exp", "Exp", "Sud", "Sud", "Sud"], ["s2"]),
                 (True, ["Sud", "Sud", "Exp", "Exp", "Exp"], ["s1"]),
                 (False, ["Exp", "Exp", "Sud", "Sud", "Sud"],
                  ["nu12_1", "nu12_2"])]
        for split_fractions, dyns, fixed in cases:
            msg = f"split fractions {split_fractions}, dynamics {dyns}"
            settings = self.get_settings(
                vcf_file, popmap, [1, 1, 1], lengths=lengths,
                preprocessed_data=preprocessed_data,
                split_fractions=split_fractions)
            data = settings.read_data()
            model = settings.get_model()
            np.random.seed(2)
            x, dyn_values = [], list(dyns)
            for var in model.variables:
                if var.name == "Nanc":
                    x.append(10000)
                elif var.name.startswith("dyn"):
                    x.append(dyn_values.pop(0))
                else:
                    x.append(var.resample())
            engine = get_engine("momentsLD")
            engine.set_data(data)
            engine.data_holder = settings.data_holder
            engine.set_model(model)
            moments.LD.Inference._varcov_inv_cache = {}
            ll_model = engine.evaluate(x)

            filename = os.path.join(self.dirname, "code.py")
            engine.generate_code(x, filename, None, 10000)
            namespace = self.run_script(filename)
            self.assertTrue(np.isclose(ll_model, namespace["ll_model"]),
                            msg=f"{ll_model} != {namespace['ll_model']}. "
                                f"{msg}")
            # model of engine is not changed
            self.assertEqual(len(model.variables), len(x), msg=msg)
            # parameters for confidence intervals
            namespace = {}
            with open(os.path.join(self.dirname,
                                   "codedata_for_CI.py")) as fl:
                exec(fl.read(), namespace)
            names = [var.name for var in model.variables
                     if not isinstance(var, DiscreteVariable)]
            expected = [name for name in names
                        if name != "Nanc" and name not in fixed] + ["Nanc"]
            self.assertEqual(namespace["param_names"], expected, msg=msg)
            self.assertEqual(len(namespace["opt_params"]), len(expected),
                             msg=msg)

    def test_ci_file_of_custom_model(self):
        """
        Values of custom models are not translated, so their size of
        ancestral population is physical. The file for confidence intervals
        has it as it is in the last place, the tool for confidence intervals
        takes Ne from there.
        """
        preprocessed_data = os.path.join(DATA_PATH, "DATA", "vcf_ld",
                                         "preprocessed_data_YRI_CEU.bp")
        settings = self.get_settings(VCF_DATA_LD, POP_MAP, [1, 1],
                                     preprocessed_data=preprocessed_data)
        settings.custom_filename = os.path.join(
            DATA_PATH, "MODELS", "demographic_model_momentsLD_YRI_CEU.py")
        settings.parameter_identifiers = ["nu1F", "nu2B", "nu2F", "m", "Tp",
                                          "T"]
        settings.lower_bound = [1e-2, 1e-2, 1e-2, 0, 1e-3, 1e-3]
        settings.upper_bound = [100, 100, 100, 10, 5, 5]
        data = settings.read_data()
        model = settings.get_model()
        self.assertTrue(isinstance(model, CustomDemographicModel))
        values = {"Nanc": 10000, "nu1F": 2.0, "nu2B": 0.5, "nu2F": 3.0,
                  "m": 1.0, "Tp": 0.2, "T": 0.1}
        x = [values[var.name] for var in model.variables]
        engine = get_engine("momentsLD")
        engine.set_data(data)
        engine.data_holder = settings.data_holder
        engine.set_model(model)
        engine.generate_code(x, os.path.join(self.dirname, "code.py"), None,
                             10000)
        namespace = {}
        with open(os.path.join(self.dirname, "codedata_for_CI.py")) as fl:
            exec(fl.read(), namespace)
        names = ["nu1F", "nu2B", "nu2F", "m", "Tp", "T", "Nanc"]
        self.assertEqual(namespace["param_names"], names)
        self.assertEqual(namespace["opt_params"],
                         [values[name] for name in names])

    def test_code_of_iterations(self):
        """
        Code of momentsLD is written at iterations of the run when it is
        asked.
        """
        from gadma.core import SharedDictForCoreRun
        from gadma.core.core_run import CoreRun
        preprocessed_data = os.path.join(DATA_PATH, "DATA", "vcf_ld",
                                         "preprocessed_data_YRI_CEU.bp")
        settings = self.get_settings(VCF_DATA_LD, POP_MAP, [1, 1],
                                     preprocessed_data=preprocessed_data)
        settings.print_models_code_every_n_iteration = 1
        settings.draw_models_every_n_iteration = 0
        settings.silence = True
        core_run = CoreRun(1, SharedDictForCoreRun(multiprocessing=False),
                           settings)
        self.assertTrue("momentsLD" in os.listdir(core_run.code_dir))
        model = core_run.model
        x = [10000 if var.name == "Nanc" else
             ("Sud" if var.name.startswith("dyn") else var.resample())
             for var in model.variables]
        core_run.code_iter_callback(x, -100)
        code_dir = os.path.join(core_run.code_dir, "momentsLD")
        self.assertTrue("iteration_0.py" in os.listdir(code_dir),
                        msg=f"{os.listdir(code_dir)}")


class TestModelEvaluation(unittest.TestCase):

    def tearDown(self):
        moments.LD.Inference._varcov_inv_cache = {}
        if Path(f"{TEST_OUTPUT}/").exists():
            shutil.rmtree(f"{TEST_OUTPUT}/")
        os.mkdir(TEST_OUTPUT)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    def model_for_gadma_moments_evaluation(self):
        def model_moments_ld(params, rho, theta):
            nu, nu1, nu2, tf, m12, m21 = params

            ld = moments.LD.Numerics.steady_state(nus=[1], rho=rho, theta=theta)
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

        rhos = 4 * 10000 * R_BINS
        theta = 4 * 10000 * 6.0e-5

        simulated = moments.LD.Inference.bin_stats(
            model_func=model_moments_ld,
            params=[values[x] for x in list_for_moments_ld],
            rho=rhos,
            theta=theta
        )
        simulated = moments.LD.Inference.sigmaD2(simulated)

        return values, simulated, dm

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    @pytest.mark.timeout(0)
    def test_evaluation_ld(self):
        # use data read with gadma
        engine = get_engine('momentsLD')

        preprocessed_test_data = os.path.join(
            DATA_PATH, 'DATA', 'vcf_ld',
            'preprocessed_data_test_engine.bp'
        )
        with open(preprocessed_test_data, 'rb') as file:
            region_stats_moments_ld = pickle.load(file)
        engine.set_data(region_stats_moments_ld)
        self.assertRaises(AttributeError, engine.generate_code, [], "filename")

        engine.data_holder = VCFDataHolder(
            vcf_file=VCF_DATA_LD, popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
            population_labels=["deme0", "deme1"],
            sequence_length=1000000
        )
        # create bed files
        bed_files_dir = os.path.join(TEST_OUTPUT, "_bed_files")
        create_bed_files_and_extract_chromosomes(
            data_holder=engine.data_holder,
            output_dir=bed_files_dir
        )
        engine.data_holder.bed_files_dir = bed_files_dir
        engine.ld_kwargs = {"cM": True, "report": "False"}
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
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
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

        theta = 4 * 10000 * 6.0e-5
        self.assertEqual(theta, engine.get_theta(vals))
        self.assertEqual(10000, engine.get_N_ancestral_from_theta(theta))

        # constrain is checked as ancestral size is known: time is multiplied
        # by it and solution that is out of bounds is skipped
        A = [[int(var.name == "tf") for var in dm.variables]]
        time = 10000 * values["tf"]
        engine.model.linear_constrain = optimizers.LinearConstrain(
            A, [0.5 * time], [2 * time])
        self.assertEqual(engine.evaluate(vals), ll_moments)
        engine.model.linear_constrain = optimizers.LinearConstrain(
            A, [0], [0.5 * time])
        self.assertTrue(engine.evaluate(vals) is None)
        # optimizers get constrain with ancestral size as its factor, they
        # move such solution to the bound
        opt_constrain = engine.get_linear_constrain_for_optimizers()
        self.assertEqual(opt_constrain.factor, 10000)
        self.assertFalse(opt_constrain.fits(vals))
        vals_tr, success = opt_constrain.try_to_transform(vals)
        self.assertTrue(success)
        self.assertTrue(engine.evaluate(list(vals_tr)) is not None)

    @pytest.mark.skipif(not gadma.moments_LD_available, reason="No momentsLD")
    @pytest.mark.timeout(0)
    def test_draw_curves(self):
        preprocessed_test_data = os.path.join(
            DATA_PATH, 'DATA', 'vcf_ld',
            'preprocessed_data_YRI_CEU.bp'
        )
        engine = get_engine('momentsLD')
        engine.data_holder = VCFDataHolder(
            vcf_file=VCF_DATA_LD, popmap_file=POP_MAP,
            recombination_maps=REC_MAPS_DIR,
            population_labels=["deme0", "deme1"],
            preprocessed_data=preprocessed_test_data
        )

        values, simulated_by_moments, dm = self.model_for_gadma_moments_evaluation()
        vals = [values[var.name] for var in dm.variables
                if (var.name in values)]
        engine.set_model(dm)
        engine.model.Nanc_size = 10000
        engine.model.mutation_rate = 6.0e-5
        engine.inner_data = engine._read_data(engine.data_holder)

        engine.draw_data_comp_plot(values=vals, save_file=SAVE_IMAGE,)
