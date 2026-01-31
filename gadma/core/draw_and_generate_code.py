from .. import matplotlib, Image
from .. import matplotlib_available, PIL_available
from ..models import EpochDemographicModel, CustomDemographicModel
from ..utils import DiscreteVariable
from ..engines import all_available_engines, Engine, get_engine
from ..utils import bcolors

import warnings
import os
import io
from datetime import datetime
import copy
import numpy as np
if matplotlib_available:
    from matplotlib import pyplot as plt


def get_Nanc_gen_time_and_units(x, engine, settings):
    """
    Returns size of ancestral size, time for one generation and units of time
    in plot drawing.
    """
    Nanc = engine.get_N_ancestral(x, *settings.get_engine_args())
    if settings.time_for_generation is not None:
        gen_time = settings.time_for_generation *\
                   settings.const_of_time_in_drawing
        gen_time_units = settings.units_of_time_in_drawing
    else:
        if settings.units_of_time_in_drawing == "genetic units":
            gen_time = 0.5
            gen_time_units = settings.units_of_time_in_drawing
            Nanc = 1.0
        gen_time = 1.0
        gen_time_units = "generations"
    return Nanc, gen_time, gen_time_units


def draw_plots_to_file(x, engine, settings, filename, fig_title):
    """
    Draws plots of data (SFS) and model from ``engine`` with parameters
    ``x``.

    :param x: Values of the parameters in model.
    :type x: list or dict
    :param engine: Original engine with specified model and data.
    :type engine: :class:`gadma.engines.Engine`
    :param filename: File name to save picture.
    :type filename: str
    :param fig_title: Title of the schematic model plot.
    :type fig_title: str

    :note: print warnings if schematic model plot was not drawn.
    """
    if not matplotlib_available:
        raise ValueError("Matplotlib is required to draw models.")
    # close all other figures that were accidentally created before
    plt.close()
    model_plot_engine = get_engine(settings.model_plot_engine)
    model_plot_engine.data_holder = engine.data_holder
    model_plot_engine.model = engine.model
    comp_plot_engine = engine

    Nanc, gen_time, gen_time_units = get_Nanc_gen_time_and_units(
        x=x,
        engine=engine,
        settings=settings,
    )
    warnings.filterwarnings(
            'ignore', category=matplotlib.cbook.MatplotlibDeprecationWarning)

    # 0. Get place of dot in filename - where file extention starts
    pos = filename.rfind('.')
    if pos == -1:
        pos = len(filename)

    # 0. Check that we can draw plot
    bad_model = False
    if engine.id != model_plot_engine.id:
        bad_model = isinstance(engine.model, CustomDemographicModel)
        good_case = engine.id == "dadi" and model_plot_engine.id == "demes"
        if bad_model and good_case:
            bad_model = False

    # Check that plots will be drawn
    draw_comp_plot = comp_plot_engine.can_draw_comp
    draw_model_plot = not bad_model and model_plot_engine.can_draw_model

    if not draw_comp_plot and not draw_model_plot:
        raise ValueError(f"Plot is missed: engine {engine.id} cannot "
                         "draw data comparison and engine "
                         f"{model_plot_engine.id} cannot draw models.")

    fig1 = None
    fig2 = None
    # 1. Draw data comparison
    if draw_comp_plot:
        # 1.1 Set file or buffer to save plot
        save_file_comp = filename[:pos] + '_data_comp.pdf'
        # 1.2 Draw plot to save_file
        fig1 = comp_plot_engine.draw_data_comp_plot(
            x,
            *settings.get_engine_args(comp_plot_engine.id),
            save_file=save_file_comp,
            vmin=settings.vmin
        )

    # 2 Draw schematic model plot
    if draw_model_plot:
        # 2.1 Set file or buffer to save plot
        save_file_model = filename[:pos] + '_model.pdf'
        # 2.2 Draw model plot with moments engine
        # We use try except to be careful
        fig2 = model_plot_engine.draw_schematic_model_plot(
            values=x,
            save_file=save_file_model,
            fig_title=fig_title,
            nref=Nanc,
            gen_time=gen_time,
            gen_time_units=gen_time_units
        )

    # 3. Concatenate plots if PIL is available
    if PIL_available and draw_comp_plot and draw_model_plot:
        save_file_comp = io.BytesIO()
        save_file_model = io.BytesIO()

        fig1.savefig(save_file_comp, dpi=300)
        fig2.savefig(save_file_model, dpi=300)

        save_file_comp.seek(0)
        save_file_model.seek(0)

        img1 = Image.open(save_file_model)
        img2 = Image.open(save_file_comp)

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
    if fig1 is not None:
        plt.close(fig1)
    if fig2 is not None:
        plt.close(fig2)


def generate_code_to_file(
    x, engine, settings, filename, available_engines=None,
):
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
    :param available_engines: list of engines to generate code for.
                              If None then for all available engines.
    """
    if available_engines is None:
        available_engines = [eng.id for eng in all_available_engines()]

    pos = filename.rfind('.')
    if pos == -1 or filename[pos:] != '.py':
        pos = len(filename)
    prefix = filename[:pos]

    Nanc, gen_time, gen_time_units = get_Nanc_gen_time_and_units(
        x=x,
        engine=engine,
        settings=settings,
    )
    # Generate code
    if isinstance(engine.model, EpochDemographicModel):
        # we now can use even demes when we do not have Nanc
        engines_ids = available_engines
        engines = [get_engine(engine_id) for engine_id in engines_ids]
    else:
        engines = [copy.deepcopy(engine)]
        # we can still use demes if we have dadi's custom model
        if engine.id == "dadi" and "demes" in available_engines:
            engines.append(get_engine("demes"))
    failes = {}  # engine.id: reason

    for other_engine in engines:
        save_file = prefix + f"_{other_engine.id}_code.py"
        # other_engine.set_data(engine.data)
        other_engine.data_holder = copy.deepcopy(engine.data_holder)
        other_engine.set_model(engine.model)
        args = settings.get_engine_args(other_engine.id)
        if other_engine.id == "momentsLD":
            args = ["Void"]
        try:
            other_engine.generate_code(x, save_file, *args, Nanc, gen_time,
                                       gen_time_units)
        except Exception as e:
            failes[other_engine.id] = str(e)

    if len(failes) > 0:
        raise ValueError("; ".join([f"{id}: {failes[id]}" for id in failes]))


def get_string_for_metric_value(value):
    if isinstance(value, tuple):
        if value[0] is None:
            return "None"
        return f"{value[0]:.2f} (eps={value[1]:.1e})"
    return f"{value:.2f}"


def get_fig_title(best_by, metrics_names, metrics_vals_dict):
    """
    Returns a string for figure title. Best by {best_by} + metric: value

    :param best_by: String telling by which metric this model is the best
    :type best_by: string
    :param metrics_names: Names of metrics that should be mentioned in the
                          second part of the title
    :type metrics_names: list of strings
    :param metrics_vals_dict: Dictionary with metric_name: value
    :type matrics_vals_dict: dict
    """
    fig_title_items = [f"Best by {best_by} model"]
    for metric_name in metrics_names:
        value = metrics_vals_dict.get(metric_name, None)
        fig_title_items.append(
            f"{metric_name}: {get_string_for_metric_value(value)}"
        )
    return fig_title_items[0] + "\n" + ", ".join(fig_title_items[1:])


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
        metrics_names = [metric[0].upper() + metric[1:] for metric in metrics]
        print("Number", *metrics_names, "Model", "Units", sep='\t')
        # save fig titles to use in drawing of best model at the end
        for model in sorted_models:
            index, info = model
            engine, x, y_vals = info
            # Check if engine has model, if not then it was unpickleable and
            # we should restore it.
            if engine.model is None:
                default_model = settings.get_model()
                assert (
                    isinstance(default_model, CustomDemographicModel)
                ), f"default model is instance of {default_model.__class__}"
                super(Engine, engine).__setattr__("_model", default_model)
            # Get theta and N ancestral
            addit_str = ""
            Nanc = None
            is_custom = isinstance(engine.model, CustomDemographicModel)
            if (hasattr(engine.model, "fixed_anc_size") and
                    engine.model.fixed_anc_size is not None):
                Nanc = engine.model.fixed_anc_size
            elif engine.id in ["dadi", "moments"]:
                # dadi and moments has the same function get_N_ancestral
                # but we want to print theta
                theta = engine.get_theta(x, *settings.get_engine_args())
                # get_N_anc_from_theta will return None if it is impossible to
                # translate units
                Nanc = engine.get_N_ancestral_from_theta(theta)
                if theta is None or theta < 0:
                    theta_str = "None"
                else:
                    theta_str = f"{theta: .2f}"
                addit_str += f"(theta = {theta_str})"
            elif not is_custom:
                Nanc = engine.get_N_ancestral(x, *settings.get_engine_args())
            # Nanc can be None if we have custom demographic model and
            # we cannot get Nanc size from theta
            model_str = ""
            units = "genetic"  # default units
            if settings.relative_parameters:
                x_translated = engine.model.translate_values(
                    units="genetic", values=x, Nanc=Nanc
                )
                if Nanc is not None:
                    addit_str += f" (Nanc = {int(Nanc)})"
                units = "genetic"
            else:
                if not engine.model.has_anc_size and Nanc is not None:
                    model_str += f" [Nanc = {int(Nanc)}] "
                # In some cases dadi and moments fail to generate SFS
                # and Nanc size become <0. In that case we ignore such
                # solutions (ll is None) and do not translate values
                if engine.id in ['dadi', 'moments'] and Nanc < 0:
                    model_str += "[WARNING Nanc < 0!]"
                    x_translated = x
                    units = "genetic"
                else:
                    x_translated = engine.model.translate_values(
                        units="physical",
                        values=x,
                        Nanc=Nanc,
                        time_in_generations=False
                    )
                    units = "physical"

            model_str += engine.model.as_custom_string(x_translated)
            if hasattr(x, "metadata"):
                model_str += f"\t{x.metadata}"
            if len(addit_str) > 0:
                model_str += f"\t{addit_str}"

            if units == "physical":
                if engine.model.gen_time is not None:
                    time_units = "years"
                else:
                    time_units = "generations"
                units_str = f"{units}, time in {time_units}"
            else:
                units_str = units

            # Begin to print
            metric_vals = [
                get_string_for_metric_value(y_vals[met]) for met in metrics
            ]
            print(f"Run {index}", *metric_vals, model_str,
                  units_str, sep='\t')

        # Check bounds for the best model
        index, (engine, x, y_vals) = sorted_models[0]
        variables = engine.model.variables
        hit_lower_bound = []
        hit_upper_bound = []
        for val, var in zip(x, variables):
            if isinstance(var, DiscreteVariable):
                continue
            if np.isclose(val, var.domain[0]):
                hit_lower_bound.append(var.name)
            if np.isclose(val, var.domain[1]):
                hit_upper_bound.append(var.name)
        if len(hit_lower_bound) > 0 or len(hit_upper_bound) > 0:
            msg = "\nINFO: Some parameters of the best"\
                  " model hit their bounds: "
            if len(hit_lower_bound) > 0:
                msg += ", ".join(hit_lower_bound) + " hit lower bounds"
            if len(hit_lower_bound) > 0 and len(hit_upper_bound) > 0:
                msg += "; "
            if len(hit_upper_bound) > 0:
                msg += ", ".join(hit_upper_bound) + " hit upper bounds"
            print(msg)

        # Check if the best model has numerics failings in case of
        # dadi and moments
        if engine.id in ["dadi", "moments"]:
            key = engine._get_key(
                x,
                *settings.get_engine_args(engine_id=engine.id)
            )
            failed_f = engine.saved_add_info[key]["failed_f"] * 100
            if failed_f != 0:
                print(
                    f"\nINFO: Some numerics ({failed_f:.2f}%) for the best"
                    " model were failed. If this is the end of GADMA run then"
                    " it is better to run more repeats", end=""
                )
                if engine.id == "dadi":
                    print(" or increase Pts option", end="")
                print(".")

        # Draw and generate code for best model
        index, (engine, x, y_vals) = sorted_models[0]
        prefix = (settings.BASE_OUTPUT_DIR_PREFIX +
                  settings.LONG_NAME_2_SHORT.get(best_by.lower(),
                                                 best_by.lower()))
        out_dir = settings.output_directory
        save_plot_file = os.path.join(out_dir, prefix + "_model.png")
        save_code_file = os.path.join(out_dir, prefix + "_model.py")
        drawn = True
        gener = True
        fig_title = get_fig_title(
            best_by=best_by,
            metrics_names=metrics,
            metrics_vals_dict=y_vals
        )
        try:
            draw_plots_to_file(x, engine, settings, save_plot_file, fig_title)
        except Exception as e:
            drawn = False
            print(f"{bcolors.WARNING}Run {index} warning: failed to draw model"
                  f" due to the following exception: {e}{bcolors.ENDC}.")
        try:
            generate_code_to_file(
                x,
                engine,
                settings,
                save_code_file,
                settings.get_available_engines(),
            )
        except Exception as e:
            gener = False
            print(
                f"{bcolors.WARNING}Run {index} warning: failed to generate"
                f" some code due to the following exception: {e}{bcolors.ENDC}"
            )
        if drawn and gener:
            print("\nYou can find the picture and the Python code of the best "
                  "model in the output directory.\n")
        elif drawn:
            print("\nYou can find the picture of the best model in the output "
                  "directory.\n")
        elif gener:
            print("\nYou can find the Python code of the best model in the "
                  "output directory.\n")
