from .. import matplotlib, moments, dadi, Image
from .. import matplotlib_available, PIL_available, moments_available
from ..models import EpochDemographicModel, Split
from ..engines import get_engine, all_engines
from ..utils import bcolors, WeightedMetaArray

import warnings
import os
import warnings
import io
from datetime import datetime
import copy
import numpy as np


def draw_plots_to_file(x, engine, settings, filename, fig_title):
    """
    Draws plots of data (SFS) and model from ``engine`` with parameters
    ``x``.

    :param x: Values of the parameters in model.
    :type x: list or dict
    :param engine: Engine with specified model and data.
    :type engine: :class:`gadma.engines.engine.Engine`
    :param filename: File name to save picture.
    :type filename: str
    :param fig_title: Title of the schematic model plot.
    :type fig_title: str

    :note: print warnings if schematic model plot was not drawn.
    """
    if not matplotlib_available:
        raise ValueError("Matplotlib is required to draw models.")
        return
    warnings.filterwarnings(
            'ignore', category=matplotlib.cbook.MatplotlibDeprecationWarning)

    # 0. Get place of dot in filename - where file extention starts
    pos = filename.rfind('.')
    if pos == -1:
        pos = len(filename)

    # 1. Draw sfs
    # 1.1 Set file or buffer to save plot
    if PIL_available:  # then we will concatenate plots later
        if not moments_available:  # then we won't draw model plot
            save_file_sfs = filename
        else:  # future concatenation
            save_file_sfs = io.BytesIO()
    else:
        save_file_sfs = filename[:pos] + '_sfs' + filename[pos:]
    # 1.2 Draw plot to save_file
    engine.draw_sfs_plots(x, *settings.get_engine_args(),
                          save_file=save_file_sfs, vmin=settings.vmin)

    # 2 Draw schematic model plot
    # 2.0 Check that moments is available, it not we return
    if not moments_available:
        raise ValueError("Moments is required to draw schematic model plots.")

    # 2.1 Set file or buffer to save plot
    if PIL_available:
        save_file_model = io.BytesIO()
    else:
        save_file_model = filename[:pos] + '_model' + filename[pos:]
    # 2.2 Create all arguments for drawing function
    nref = engine.get_N_ancestral(x, *settings.get_engine_args())
    gen_time = settings.time_for_generation
    if gen_time is None:
        gen_time = 1.0
        gen_time_units = "Generations"
    else:
        gen_time *= settings.const_of_time_in_drawing
        gen_time_units = settings.units_of_time_in_drawing
    # 2.3 Draw model plot with moments engine
    # We use try except to be carefull
    try:
        engine.draw_schematic_model_plot(x, save_file_model, fig_title, nref,
                                         gen_time, gen_time_units)
    except Exception as e:
        save_file_sfs.seek(0)
        with open(filename, 'wb') as fl:
            fl.write(save_file_sfs.read())
        raise e

    # 3. Concatenate plots if PIL is available
    if PIL_available:
        save_file_sfs.seek(0)
        save_file_model.seek(0)

        img1 = Image.open(save_file_model)
        img2 = Image.open(save_file_sfs)

        if img2.size[1] < img1.size[1]:
            img2 = img2.resize(
                (int(img1.size[1] * img2.size[0] / img2.size[1]),
                 img1.size[1]))

        width = img1.size[0] + img2.size[0]
        height = max(img1.size[1], img2.size[1])

        new_img = Image.new('RGB', (width, height))

        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.size[0], 0))

        new_img.save(filename)


def generate_code_to_file(x, engine, settings, filename):
    """
    Generates code of demographic model to file. Settings are required to get
    ``engine`` arguments in :func:`evaluation` function.

    :param x: Values of the parameters in model.
    :type x: list or dict
    :param engine: Engine with specified model and data.
    :type engine: :class:`gadma.engines.engine.Engine`
    :param settings: Settings of the run.
    :type settings: :class:`gadma.cli.settings_storage.SettingsStorage`
    :param filename: File name to save picture.
    :type filename: str
    """
    pos = filename.rfind('.')
    if pos == -1 or filename[pos:] != '.py':
        pos = len(filename)
    prefix = filename[:pos]

    nref = engine.get_N_ancestral(x, *settings.get_engine_args())
    gen_time = None
    if settings.time_for_generation is not None:
        gen_time = settings.time_for_generation *\
                   settings.const_of_time_in_drawing
    gen_time_units = settings.units_of_time_in_drawing
    # Generate code
    if isinstance(engine.model, EpochDemographicModel):
        engines = all_engines()
    else:
        engines = [copy.deepcopy(engine)]
    for other_engine in engines:
        save_file = prefix + f"_{other_engine.id}_code.py"
        other_engine.set_data(engine.data)
        other_engine.data_holder = copy.deepcopy(engine.data_holder)
        other_engine.set_model(engine.model)
        args = settings.get_engine_args(other_engine.id)
        other_engine.generate_code(x, save_file, *args, nref, gen_time,
                                   gen_time_units)


def print_runs_summary(start_time, shared_dict, settings):
    """
    Prints best demographic model by logLL among all processes.

    :param start_time: Time when equation was started.
    :type start_time: float
    :param shared_dict: Dictionary to share information between processes.
    :type shared_dict: :class:`gadma.core.shared_dict.SharedDict`
    :param settings: Settings of run.
    :type settings: :class:`gadma.cli.settings_storage.SettingsStorage`
    """
    s = (datetime.now() - start_time).total_seconds()
    time_str = f"\n[{int(s//3600):03}:{int(s%3600//60):02}:{int(s%60):02}]"
    print(time_str)
    metric_names = shared_dict.get_available_groups()
    if len(metric_names) == 0:
        print("No models yet")
    for best_by in metric_names:
        models = shared_dict.get_models_in_group(best_by, align_y_dict=True)
        local_metrics = models[0][1][2].keys()
        sorted_models = models
        metrics = local_metrics

        print(f"All best by {best_by} models")
        print("Number", *metrics, "Model", sep='\t')
        for model in sorted_models:
            index, info = model
            engine, x, y_vals = info
            # Get theta and N ancestral
            theta = engine.get_theta(x, *settings.get_engine_args())
            Nanc = engine.get_N_ancestral_from_theta(theta)
            addit_str = f"(theta = {theta: .2f})"
            if Nanc is not None or Nanc == 0:
                if settings.relative_parameters:
                    addit_str += f" (Nanc = {int(Nanc)})"
                    model_str = engine.model.as_custom_string(x)
                else:
                    model_str = f" [Nanc = {int(Nanc)}] "
#                    Tg = settings.time_for_generation or 1.0
                    x_translated = engine.model.translate_units(x, Nanc)
                    model_str += engine.model.as_custom_string(x_translated)
            else:
                model_str = engine.model.as_custom_string(x)
            if hasattr(x, "metadata"):
                model_str += f"\t{x.metadata}"
            # Begin to print
            metric_vals = []
            for metr in metrics:
                if metr not in y_vals:
                    metric_vals.append("None")
                else:
                    if isinstance(y_vals[metr], tuple):
                        if y_vals[metr][0] is None:
                            val_str = "None"
                        else:
                            val_str = f"{y_vals[metr][0]:.2f} "\
                                      f"(eps={y_vals[metr][1]:.1e})"
                    else:
                        val_str = f"{y_vals[metr]:.2f}"
                    metric_vals.append(val_str)
            print(f"Run {index}", *metric_vals, model_str,
                  addit_str, sep='\t')
            fig_title = f"Best by {best_by} model. "
            ind = 0
            for metr in metrics:
                if metr not in y_vals:
                    continue
                val_str = metric_vals[ind]
                ind += 1
                fig_title += f"{metr}: {val_str}"
        # Draw and generate code for best model
        _, (engine, x, y_vals) = sorted_models[0]
        prefix = (settings.BASE_OUTPUT_DIR_PREFIX +
                  settings.LONG_NAME_2_SHORT.get(best_by.lower(),
                                                 best_by.lower()))
        out_dir = settings.output_directory
        save_plot_file = os.path.join(out_dir, prefix + "_model.png")
        save_code_file = os.path.join(out_dir, prefix + "_model.py")
        drawn = True
        gener = True
        try:
            draw_plots_to_file(x, engine, settings, save_plot_file, fig_title)
        except Exception as e:
            drawn = False
            print(f"{bcolors.FAIL}Run {index}: failed to draw model due to "
                  f"the following exception: {e}{bcolors.ENDC}")
        try:
            generate_code_to_file(x, engine, settings, save_code_file)
        except Exception as e:
            gener = False
            print(f"{bcolors.FAIL}Run {index}: failed to generate code due to"
                  f" the following exception: {e}{bcolors.ENDC}")
        if drawn and gener:
            print("\nYou can find the picture and the Python code of the best "
                  "model in the output directory.\n")
        elif drawn:
            print("\nYou can find the picture of the best model in the output "
                  "directory.\n")
        elif gener:
            print("\nYou can find the Python code of the best model in the "
                  "output directory.\n")
