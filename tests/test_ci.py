import unittest
import gadma
import sys
import gadma
from gadma import *
import itertools
import shutil

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
            p0 = [2, 0.1, 0.5, 2]  # nuB, nuF, TB, TF
            output_dir = os.path.join(DATA_DIR, f"run_ls_out_{engine.id}")
            with open(params_file, 'w') as f:
                f.write(f"p0 = {p0}\n")
                if engine.id == 'dadi':
                    f.write(f"pts = [5, 10, 15]")
            sys.argv = ["gadma-run_ls_on_boot_data",  "-b", dir_boot,
                        "-d", model_name, "-o", output_dir, "-p", params_file,
                        "-j", "4"]
            try:
                for opt in ['optimize_lbfgsb', 'log', 'powell']: #all_local_optimizers():
                    sys.argv.extend(["--opt", opt])
                    gadma.run_ls_on_boot_data.main()
                    gadma.run_ls_on_boot_data.main()
            finally:
                if os.path.exists(params_file):
                    os.remove(params_file)

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


