import unittest
import gadma
import sys
import gadma
from gadma import *
import itertools
import shutil
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.stats', lineno=21604)


DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


class TestConfidenceIntervals(unittest.TestCase):
    def test_run_boots(self):
        dir_boot = os.path.join(DATA_DIR, "small_1_pop_bootstrap")
        params_file = os.path.join(DATA_DIR, "ls_params")
        if os.path.exists(params_file):
            os.remove(params_file)

        for engine in all_engines():
            model_name = f"small_1pop_dem_model_{engine.id}.py"
            model_name = os.path.join(DATA_DIR, model_name)
            no_ids_model_name = os.path.join(DATA_DIR,
                                             "small_1pop_dem_model_no_ids.py")
            p0 = [2, 0.1, 0.5, 2]  # nuB, nuF, TB, TF
            output_dir = os.path.join(DATA_DIR, f"run_ls_out_{engine.id}")
            jobs = 1
            with open(params_file, 'w') as f:
                f.write(f"p0 = {p0}\n")
                if engine.id == 'dadi':
                    f.write(f"pts = [5, 10, 15]")
                    jobs = 4
            sys.argv = ["gadma-run_ls_on_boot_data",  "-b", dir_boot,
                        "-d", model_name, "-o", output_dir, "-p", params_file,
                        "-j", str(jobs)]
            try:
                for opt in ['optimize_lbfgsb', 'log', 'powell']: #all_local_optimizers():
                    sys.argv.extend(["--opt", opt])
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

    def test_run_ci_evaluation(self):
        for engine in all_engines():
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
        sys.argv = ["gadma-get_confidence_intervals", bad_extension]
        self.assertRaises(ValueError, gadma.get_confidence_intervals.main)
