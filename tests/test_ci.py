import unittest
import gadma
import sys
import os
import gadma
from gadma import *
import itertools
import shutil
import warnings
import copy
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.stats', lineno=21604)


DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


class TestConfidenceIntervals(unittest.TestCase):
    def test_run_boots(self):
        dir_boot = os.path.join(DATA_DIR, "DATA", "sfs",
                                "small_1_pop_bootstrap")
        params_file = os.path.join(DATA_DIR, "PARAMS", "ls_params")
        if os.path.exists(params_file):
            os.remove(params_file)

        for engine in all_engines():
            # momentsLD has different method for CI evaluation
            if engine.id == "momentsLD":
                continue
            model_name = f"small_1pop_dem_model_{engine.id}.py"
            model_name = os.path.join(DATA_DIR, "MODELS", model_name)
            no_ids_model_name = os.path.join(DATA_DIR, "MODELS",
                                             "small_1pop_dem_model_no_ids.py")
            p0 = [2, 0.1, 0.5, 2]  # nuB, nuF, TB, TF
            if engine.id not in ["dadi", "moments"]:
                # translate units to physical
                p0 = [20000, 1000, 10000, 40000]
            output_dir = os.path.join(DATA_DIR, f"run_ls_out_{engine.id}")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            jobs = 1
            with open(params_file, 'w') as f:
                f.write(f"p0 = {p0}\n")
                if engine.id == 'dadi':
                    f.write(f"pts = [4, 6, 8]")
                    jobs = 4
            basic_argv = ["gadma-run_ls_on_boot_data",  "-b", dir_boot,
                        "-d", model_name, "-o", output_dir, "-p", params_file,
                        "-j", str(jobs)]
            try:
                for opt in ['optimize_lbfgsb', 'log']: #all_local_optimizers():
                    sys.argv = copy.copy(basic_argv)
                    sys.argv.extend(["--opt", opt])
                    # for dadi and moments engine is defined automatically
                    if engine.id in ["dadi", "moments"]:
                        gadma.run_ls_on_boot_data.main()
                        gadma.run_ls_on_boot_data.main()
                    else:
                        # error about engine specification
                        self.assertRaises(ValueError,
                                          gadma.run_ls_on_boot_data.main)
                        sys.argv.extend(["--engine", engine.id])
                        gadma.run_ls_on_boot_data.main()
                        gadma.run_ls_on_boot_data.main()
            finally:
                if os.path.exists(params_file):
                    os.remove(params_file)

            # Failures
            sys.argv = ["gadma-run_ls_on_boot_data",  "-b", dir_boot,
                        "-d", model_name, "-o", output_dir]
            self.assertRaises(ValueError, gadma.run_ls_on_boot_data.main)
            sys.argv = ["gadma-run_ls_on_boot_data",  "-b", dir_boot,
                        "-d", no_ids_model_name, "-o", output_dir]
            self.assertRaises(ValueError, gadma.run_ls_on_boot_data.main)

    def test_dadi_extrapolation_in_run_boots(self):
        """
        Local search on bootstrapped data uses the extrapolation of dadi
        that is used in the file with demographic model.
        """
        import dadi
        import numpy as np
        import pandas as pd
        import tempfile
        from gadma.engines import DadiEngine
        from gadma.run_ls_on_boot_data import load_module, \
            get_dadi_extrapolation_from_python_file

        dir_boot = os.path.join(DATA_DIR, "DATA", "sfs",
                                "small_1_pop_bootstrap")
        base_model = os.path.join(DATA_DIR, "MODELS",
                                  "small_1pop_dem_model_dadi.py")
        with open(base_model) as f:
            base_code = f.read()
        p0, pts = [2, 0.1, 0.5, 2], [4, 6, 8]
        log_name, lin_name = "make_extrap_log_func", "make_extrap_func"
        codes = {
            "no": base_code,
            "log": base_code + f"\nfunc_ex = dadi.Numerics.{log_name}("
                               "model_func)\n",
            "lin": base_code + f"\nfunc_ex = dadi.Numerics.{lin_name}("
                               "model_func)\n",
            "lin_import": "from dadi.Numerics import make_extrap_func\n" +
                          base_code + "\n# not make_extrap_log_func\n"
                          "func_ex = make_extrap_func(model_func)\n",
            "both": base_code + f"\nfunc_ex = dadi.Numerics.{lin_name}("
                                f"model_func)\nfunc_ex_2 = dadi.Numerics."
                                f"{log_name}(model_func)\n",
            "not_python": "some text (",
        }
        in_file = {"no": None, "log": log_name, "lin": lin_name,
                   "lin_import": lin_name, "both": None, "not_python": None}

        work_dir = tempfile.mkdtemp("gadma_test_dir")
        try:
            files = {}
            for key, code in codes.items():
                files[key] = os.path.join(work_dir, f"model_{key}.py")
                with open(files[key], "w") as f:
                    f.write(code)
                self.assertEqual(
                    get_dadi_extrapolation_from_python_file(files[key]),
                    in_file[key], msg=key
                )
            self.assertEqual(get_dadi_extrapolation_from_python_file(None),
                             None)
            params_file = os.path.join(work_dir, "ls_params.py")
            with open(params_file, "w") as f:
                f.write(f"p0 = {p0}\npts = {pts}\n")

            # Local search is switched off, so the result is theta of p0
            # for each bootstrapped data. It depends on the extrapolation.
            boot_names = sorted(os.listdir(dir_boot))
            model_func = load_module(base_model).model_func
            expected = {}
            for name in [log_name, lin_name]:
                func_ex = getattr(dadi.Numerics, name)(model_func)
                expected[name] = []
                for boot_name in boot_names:
                    fs = dadi.Spectrum.from_file(
                        os.path.join(dir_boot, boot_name))
                    expected[name].append(dadi.Inference.optimal_sfs_scaling(
                        func_ex(p0, fs.sample_sizes, pts), fs))
            self.assertFalse(np.allclose(expected[log_name],
                                         expected[lin_name]))

            for key, jobs, used in [("lin", 1, lin_name),
                                    ("no", 1, log_name),
                                    ("lin_import", 2, lin_name),
                                    ("log", 1, log_name)]:
                output_dir = os.path.join(work_dir, f"out_{key}")
                sys.argv = ["gadma-run_ls_on_boot_data", "-b", dir_boot,
                            "-d", files[key], "-o", output_dir,
                            "-p", params_file, "-j", str(jobs),
                            "--opt", "None"]
                gadma.run_ls_on_boot_data.main()
                table = pd.read_csv(
                    os.path.join(output_dir, "result_table.csv"), index_col=0)
                thetas = [table.loc[boot_name, "Theta"]
                          for boot_name in boot_names]
                self.assertTrue(np.allclose(thetas, expected[used]),
                                msg=f"file {key}, jobs {jobs}: {thetas}")
        finally:
            shutil.rmtree(work_dir)
            DadiEngine.extrapolation = log_name

    def test_run_ci_evaluation(self):
        moments.LD.Inference._varcov_inv_cache = {}
        warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.stats', lineno=1604)
        for engine in all_engines():
            if engine.id == "momentsLD":
                results = os.path.join(DATA_DIR, "ci_test_ld", "ci_results.xlsx")
                incorrect_data_for_ci_ld = os.path.join(
                    DATA_DIR, "ci_test_ld", "ci_test_incorrect_file.py"
                )
                try:
                    data_for_ci_ld = os.path.join(DATA_DIR, "ci_test_ld", "data_for_ci_test.py")
                    sys.argv = ["gadma-get_confidence_intervals_for_ld", data_for_ci_ld]
                    gadma.get_confidence_intervals_for_ld.main()
                    self.assertTrue(os.path.exists(results))

                    not_exist_data = os.path.join(DATA_DIR, "ci_test_ld", "not_exist.py")
                    sys.argv = ["gadma-get_confidence_intervals_for_ld", not_exist_data]
                    self.assertRaises(ValueError, gadma.get_confidence_intervals_for_ld.main)

                    line_to_ignore = 'data_file = ' \
                                     'os.path.join(os.path.dirname(__file__), ' \
                                     '\"preprocessed_data.bp\")'
                    with open(data_for_ci_ld, 'r') as original_data:
                        lines = original_data.readlines()

                    with open(incorrect_data_for_ci_ld, 'w') as file:
                        for line in lines:
                            if line.rstrip('\n') != line_to_ignore:
                                file.write(f'{line}')
                    sys.argv = ["gadma-get_confidence_intervals_for_ld", incorrect_data_for_ci_ld]
                    self.assertRaises(ValueError, gadma.get_confidence_intervals_for_ld.main)

                finally:
                    if os.path.exists(results):
                        os.remove(results)
                    if os.path.exists(incorrect_data_for_ci_ld):
                        os.remove(incorrect_data_for_ci_ld)

            else:
                output_dir = os.path.join(DATA_DIR, f"run_ls_out_{engine.id}")
                table = os.path.join(output_dir, 'result_table')
                try:
                    for is_log, is_tex, has_acc in itertools.product([False, True],
                                                                     repeat=3):
                        for ext in [".csv", ".pkl"]:
                            sys.argv = ["gadma-get_confidence_intervals",
                                        table + ext]
                            if is_log:
                                sys.argv.extend(["--log"])
                            if is_tex:
                                sys.argv.extend(["--tex"])
                            if has_acc:
                                sys.argv.extend(["--acc", "10"])
                            gadma.get_confidence_intervals.main()
                finally:
                    shutil.rmtree(output_dir)

    def test_missed_lines_and_failures(self):
        load_parameters_from_python_file(None)
        not_valid_file = os.path.join(DATA_DIR, 'params_file')
        self.assertRaises(ValueError,
                          load_parameters_from_python_file, not_valid_file)
        not_existed_file = "not_existing_filename"
        sys.argv = ["gadma-get_confidence_intervals", not_existed_file]
        self.assertRaises(ValueError, gadma.get_confidence_intervals.main)
        bad_extension = not_valid_file
        open(bad_extension, 'a').close()
        sys.argv = ["gadma-get_confidence_intervals", bad_extension]
        self.assertRaises(ValueError, gadma.get_confidence_intervals.main)
        os.remove(bad_extension)
