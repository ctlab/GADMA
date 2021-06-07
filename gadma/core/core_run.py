from ..engines import get_engine, all_available_engines
from ..utils import sort_by_other_list, ensure_dir_existence,\
                    ensure_file_existence, check_file_existence,\
                    check_dir_existence
from ..utils import WeightedMetaArray
from ..optimizers import GlobalOptimizerAndLocalOptimizer
from ..utils import get_aic_score, get_claic_score, ident_transform, bcolors
from ..models import EpochDemographicModel, StructureDemographicModel
from .draw_and_generate_code import draw_plots_to_file, generate_code_to_file
from .draw_and_generate_code import get_Nanc_gen_time_and_units
from ..cli import SettingsStorage
import os
import numpy as np
from collections import OrderedDict
import copy
import warnings
from functools import partial


class CoreRun(object):
    """
    Class of main run in GADMA.
    Has a :meth:`run` method to start launch.
    Runs creates new directory named by its `index` in the output directory
    then all log, code and pictures will be saved there. AIC and CLAIC is
    calculated here.

    :param index: Index of the run. Like id.
    :type index: int
    :param shared_dict: Dictionary to save results in callbacks. Will be saved
                        with key equal to `index`. Is used for
                        multiprocessing cooperation.
    :type shared_dict: :class:`gadma.cor.shared_dict.SharedDictForCoreRun`
    :param settings: Settings of the run. Information to form output directory
                     and so on will be taken from settings.
    :type settings: :class:`gadma.cli.settings_storage.SettingsStorage`
    """
    REPORT_FILENAME = 'GADMA_GA.log'
    EVAL_FILENAME = 'eval_file'
    SAVE_FILENAME = 'save_file'

    def __init__(self, index, shared_dict, settings):
        # 1. Save all init arguments
        self.index = index
        self.shared_dict = shared_dict
        self.settings = settings

        # 2. Extract other information from settings
        # 2.1 Take engine, data and model for the first start.
        self.engine = get_engine(self.settings.engine)
        self.data = self.settings.inner_data
        self.model = self.settings.get_model()
        # We save data_holder in engine for good code generation
        self.engine.data_holder = self.settings.data_holder
        # 2.2 Get optimizers and their kwargs that will be used.
        self.global_optimizer = self.settings.get_global_optimizer()
        self.local_optimizer = self.settings.get_local_optimizer()
        self.optimize_kwargs = self.settings.get_optimizers_kwargs()
        self.optimize_kwargs['callback'] = self.callback

        # 2.3 Check if we need to calculate and keep additional functions
        # values in callbacks during optimization.
        self.aic_score = not self.settings.linked_snp_s
        has_boots = self.settings.directory_with_bootstrap is not None
        self.claic_score = not self.aic_score and has_boots
        self.boots = self.settings.bootstrap_data

        # 2.4 Create all necessary output dirs and files
        if settings.output_directory is not None:
            self.output_dir = os.path.join(settings.output_directory,
                                           str(self.index))

            self.resume_dir = None
            if self.settings.resume_from is not None:
                self.resume_dir = os.path.join(settings.resume_from,
                                               str(self.index))
                if not check_dir_existence(self.resume_dir):
                    self.resume_dir = None

            self.output_dir = ensure_dir_existence(self.output_dir)
            self.eval_file = os.path.join(self.output_dir, self.EVAL_FILENAME)
            self.eval_file = ensure_file_existence(self.eval_file)
            self.report_file = os.path.join(self.output_dir,
                                            self.REPORT_FILENAME)
            self.report_file = ensure_file_existence(self.report_file)
            self.save_file = os.path.join(self.output_dir, self.SAVE_FILENAME)

            # Tell optimizers about output files
            self.optimize_kwargs['eval_file'] = self.eval_file
            self.optimize_kwargs['report_file'] = self.report_file
            self.optimize_kwargs['save_file'] = self.save_file
            self.optimize_kwargs['global_maxiter'] = settings.global_maxiter
            self.optimize_kwargs['global_maxeval'] = settings.global_maxeval
            self.optimize_kwargs['local_maxiter'] = settings.local_maxiter
            self.optimize_kwargs['local_maxeval'] = settings.local_maxeval
            # Create directory for pictures if needed
            if self.settings.draw_models_every_n_iteration != 0:
                self.pictures_dir = os.path.join(self.output_dir, 'pictures')
                ensure_dir_existence(self.pictures_dir)
            # Create directory for generated code if needed
            if self.settings.print_models_code_every_n_iteration != 0:
                self.code_dir = os.path.join(self.output_dir, 'code')
                ensure_dir_existence(self.code_dir)
                # If our model is built via gadma then we can write code for
                # all engines.
                if isinstance(self.model, EpochDemographicModel):
                    demes_will_not_work = False
                    mu_and_L = self.engine.model.mu is not None and \
                        self.settings.sequence_length is not None
                    if not (self.engine.model.has_anc_size or
                            self.engine.model.theta0 is not None or mu_and_L):
                        demes_will_not_work = True
                    for engine in all_available_engines():
                        if engine.id == "demes" and demes_will_not_work:
                            continue
                        engine_dir = os.path.join(self.code_dir, engine.id)
                        ensure_dir_existence(engine_dir)
        # Set counters to zero for callbacks to count number of their calls
        self.draw_iter_callback_counter = 0
        self.code_iter_callback_counter = 0

        self.x_best = None
        self.y_best = None

    @property
    def model(self):
        """
        Returns current demographic model.
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Sets new demographic model and updates model in the engine.
        """
        self._model = model
        self.engine.set_model(model)

    @property
    def data(self):
        """
        Returns current data.
        """
        return self._data

    @data.setter
    def data(self, data):
        """
        Sets new data and updates data in the engine.
        """
        self._data = data
        self.engine.set_data(data)

    def base_callback(self, x, y):
        """
        Base callback:

        1) Updates values of best solution in :attr:`shared_dict`.
        2) If new best values are received then draws and generates code to\
        the `output_dir` of this run.

        :param x: Vector of values for model parameters.
        :param y: Value of log-likelihood for this values.
        """
        best_by = 'log-likelihood'
        if self.aic_score:
            n_params = self.engine.model.get_number_of_parameters(x)
            aic = get_aic_score(n_params, y)
            y_dict = {best_by: y, 'AIC score': aic}
        else:
            y_dict = y

        sign = self.global_optimizer.sign
        equal_x = False
        if self.x_best is not None:
            equal_x = list(self.x_best) == list(x)
            x_has_w = hasattr(x, 'weights')
            x_best_has_w = hasattr(self.x_best, 'weights')
            if x_has_w != x_best_has_w:
                equal_x = False
            elif x_has_w and x.metadata != self.x_best.metadata:
                equal_x = False
        if (self.x_best is None or
                sign * self.y_best > sign * y or
                (self.y_best == y and not equal_x)):
            self.x_best = x
            self.y_best = y
            self.shared_dict._put_new_model_for_process(
                self.index, best_by, (self.engine, x, y_dict))
            prefix = (self.settings.LOCAL_OUTPUT_DIR_PREFIX +
                      self.settings.LONG_NAME_2_SHORT.get(best_by, best_by))
            save_code_file = os.path.join(self.output_dir,
                                          prefix + "_model")
            try:
                generate_code_to_file(x, self.engine,
                                      self.settings, save_code_file)
            except Exception:
                pass

    def draw_iter_callback(self, x, y):
        """
        Draws best model on current iteration in the pictures directory.
        Pictures directory is located in the output directory of this run.
        Drawing happens every
        :attr:`self.settings.draw_models_every_n_iteration` iteration.

        :param x: Vector of values for model parameters.
        :param y: Value of log-likelihood for this values.
        """
        n_iter = self.draw_iter_callback_counter
        verbose = self.settings.draw_models_every_n_iteration
        if verbose != 0 and n_iter % verbose == 0:
            fig_title = f"Iteration {n_iter}, "\
                        f"Log-likelihood: {y:.2f}"
            save_file = os.path.join(self.pictures_dir,
                                     f"iteration_{n_iter}.png")
            draw_plots_to_file(x, self.engine, self.settings,
                               save_file, fig_title)
        self.draw_iter_callback_counter += 1

    def code_iter_callback(self, x, y):
        """
        Generates code of best model on current iteration to the code
        directory. Code directory is located in the output directory of this
        run. Generation happens every
        :attr:`self.settings.print_models_code_every_n_iteration` iteration.

        :param x: Vector of values for model parameters.
        :param y: Value of log-likelihood for this values.
        """
        n_iter = self.code_iter_callback_counter
        verbose = self.settings.print_models_code_every_n_iteration
        filename = f"iteration_{n_iter}"
        if verbose != 0 and n_iter % verbose == 0:
            Nanc, gen_time, gen_time_units = get_Nanc_gen_time_and_units(
                x=x,
                engine=self.engine,
                settings=self.settings,
            )
            if isinstance(self.model, EpochDemographicModel):
                for engine_id in os.listdir(self.code_dir):
                    engine = get_engine(engine_id)
                    save_file = os.path.join(self.code_dir, engine.id,
                                             filename)
                    args = self.settings.get_engine_args(engine.id)
                    engine.set_data(self.engine.data)
                    engine.data_holder = self.engine.data_holder
                    engine.set_model(self.engine.model)
                    try:
                        engine.generate_code(x, save_file, *args, Nanc,
                                             gen_time=gen_time,
                                             gen_time_units=gen_time_units)
                    except Exception as e:
                        pass
            else:
                save_file = os.path.join(self.code_dir, filename)
                args = self.settings.get_engine_args()
                self.engine.generate_code(x, save_file, *args,  Nanc,
                                          gen_time=gen_time,
                                          gen_time_units=gen_time_units)
        self.code_iter_callback_counter += 1

    def callback(self, x, y):
        """
        Main callback for optimizers to get. It is combination of three
        callbacks:

        1) :meth:`base_callback`
        2) :meth:`draw_iter_callback`
        3) :meth:`code_iter_callback`

        :param x: Vector of values for model parameters.
        :param y: Value of log-likelihood for this values.

        """
        self.base_callback(x, y)
        try:
            self.draw_iter_callback(x, y)
        except Exception:
            pass
        self.code_iter_callback(x, y)

    def intermediate_callback(self, x, y):
        """
        Almost final callback that is called after each global + local
        optimization.

        Saves AIC and CLAIC values in the :attr:`shared_dict` if needed.
        If new best by AIC or CLAIC vector then code and picture are generated.

        :param x: Vector of values for model parameters.
        :param y: Value of log-likelihood for this values.
        """
        if self.aic_score:
            best_by = "AIC score"
            n_params = self.engine.model.get_number_of_parameters(x)
            value = get_aic_score(n_params, y)
        elif self.claic_score:
            best_by = "CLAIC score"
            args = self.settings.get_engine_args()
            value = get_claic_score(self.engine, x, self.boots,
                                    args, y, return_eps=True)
        else:
            return

        y_dict = OrderedDict()
        y_dict['log-likelihood'] = y
        y_dict[best_by] = value

        if self.claic_score:
            self.shared_dict.add_model_for_process(self.index, best_by,
                                                   self.engine, x, y_dict)
        else:
            self.shared_dict.update_best_model_for_process(self.index,
                                                           best_by,
                                                           self.engine,
                                                           x, y_dict)

        # Draw and generate code for best_by model
        if self.aic_score:
            value_str = f"{value: .2f}" if value is not None else "None"
        if self.claic_score:
            if value[0] is None:
                value_str = "None"
            else:
                value_str = f"{value[0]:.2f} (eps={value[1]: .1e})"
        fig_title = f"Best by {best_by}. Log-likelihood: {y:.2f}, "\
                    f"{best_by}: {value_str}."
        prefix = (self.settings.LOCAL_OUTPUT_DIR_PREFIX +
                  self.settings.LONG_NAME_2_SHORT.get(best_by.lower(),
                                                      best_by.lower()))
        save_plot_file = os.path.join(self.output_dir, prefix + "_model.png")
        save_code_file = os.path.join(self.output_dir, prefix + "_model")
        try:
            draw_plots_to_file(x, self.engine, self.settings,
                               save_plot_file, fig_title)
        except Exception as e:
            print(f"{bcolors.WARNING}Run {self.index}: failed to draw model "
                  f"due to the following exception: {e}{bcolors.ENDC}")
        try:
            generate_code_to_file(x, self.engine,
                                  self.settings, save_code_file)
        except Exception as e:
            print(f"{bcolors.WARNING}Run {self.index}: failed to generate code"
                  f" due to the following exception: {e}{bcolors.ENDC}")

    def draw_model_in_output_dir(self, x, y,
                                 best_by='log-likelihood', final=True):
        """
        Draws picture of demographic model with `x` as parameters to the
        output directory of run.

        :param x: Vector of values for model parameters.
        :param y: Value of log-likelihood for this values.
        :param best_by: By what function (log-likelihood, AIC, CLAIC) this
                        colution is best.
        :param final: If True then solution is final and it will be saved by
                      `final` name.
        """
        fig_title = f"Best by {best_by} model. {best_by}: {y: .2f}"
        if final:
            prefix = self.settings.LOCAL_OUTPUT_DIR_PREFIX_FINAL
        else:
            prefix = self.settings.LOCAL_OUTPUT_DIR_PREFIX
        prefix += self.settings.LONG_NAME_2_SHORT.get(best_by, best_by)
        save_plot_file = os.path.join(self.output_dir, prefix + "_model.png")
        save_code_file = os.path.join(self.output_dir, prefix + "_model")
        try:
            generate_code_to_file(x, self.engine,
                                  self.settings, save_code_file)
        except Exception:
            pass
        try:
            draw_plots_to_file(x, self.engine, self.settings,
                               save_plot_file, fig_title)
        except Exception:
            pass

    def get_save_file(self):
        """
        Returns filename to save optimization run. If demographic model does
        not have structure then returns `self.save_file` else adds suffix about
        structure at the end of `self.save_file`.
        """
        if isinstance(self.model, StructureDemographicModel):
            suffix = "_".join([str(x) for x in self.model.get_structure()])
            suffix = "_" + suffix
            return self.save_file + suffix
        return self.save_file

    def run_without_increase(self, initial_kwargs={}):
        """
        Run one launch without any increase of demographic model structure.
        Runs one global+local optimization for the current model.

        :param initial_kwargs: Initial kwargs for optimization.
        """
        np.random.seed()
        self.optimize_kwargs['callback'] = self.callback
        self.optimize_kwargs['save_file'] = self.get_save_file()
        # We set some kwargs if they were not set in run_with_increase
        if self.settings.initial_structure is None:
            options = list(self.get_run_options())
            assert len(options) > 0
            restore_file, _, only_models, x_transform = options[0]
            self.optimize_kwargs['restore_file'] = restore_file
            self.optimize_kwargs['restore_points_only'] = only_models
            self.optimize_kwargs['global_x_transform'] = x_transform[0]
            self.optimize_kwargs['local_x_transform'] = x_transform[1]
        f = self.engine.evaluate
        variables = self.model.variables

        optimizer = GlobalOptimizerAndLocalOptimizer(self.global_optimizer,
                                                     self.local_optimizer)

        opt_kwargs = {**self.optimize_kwargs, **initial_kwargs}
        result = optimizer.optimize(f, variables, **opt_kwargs)
        self.intermediate_callback(result.x, result.y)
        return result

    def run_with_increase(self, initial_kwargs={}):
        """
        Run launch with increase of the demographic model structure.
        Structure of the model will be increased up to final structure. Then
        the final solution will be returned.
        Runs :meth:`run_without_increase` and ``increase_structure()`` in the
        loop.

        :param initial_kwargs: Initial kwargs for optimization.
        """
        np.random.seed()
        # Simple checks
        assert self.settings.initial_structure is not None
        assert self.settings.final_structure is not None

        options = self.get_run_options()

        restore_file, struct, only_models, x_transform = next(options)
        if struct is not None and struct != self.model.get_structure():
            warnings.warn(f"Initial structure ({struct}) with saved file "
                          f"of optimization from restored dir looks different "
                          f"to current ({self.settings.initial_structure}). "
                          f"It will be restored.")
            self.settings.initial_structure = struct
            self.model = self.model.from_structure(struct)
        self.optimize_kwargs['restore_file'] = restore_file
        self.optimize_kwargs['restore_points_only'] = only_models
        self.optimize_kwargs['global_x_transform'] = x_transform[0]
        self.optimize_kwargs['local_x_transform'] = x_transform[1]

        result = self.run_without_increase(initial_kwargs)
        for restore_file, structure, only_models, x_transform in options:
            X_init = result.X_out
            if self.model.get_structure() != structure:
                self.model, X_init = self.model.increase_structure(structure,
                                                                   X=X_init)
            Y_init = copy.copy(result.Y_out)
            self.optimize_kwargs['X_init'] = X_init
            self.optimize_kwargs['Y_init'] = Y_init
            self.optimize_kwargs['restore_file'] = restore_file
            self.optimize_kwargs['restore_points_only'] = only_models
            self.optimize_kwargs['global_x_transform'] = x_transform[0]
            self.optimize_kwargs['local_x_transform'] = x_transform[1]
            result = self.run_without_increase()
#        self.final_callback()
        return result

    def get_run_options(self):
        """
        Returns iterator of run options, each element has four options:

        1) File to restore optimization from. If None then no resume.
        2) Structure of the demographic model for the run.
        3) Bool `points_only` - if True then resumed run uses old points as\
        initial points only and optimization is run from the beginning.
        4) Function of x transformation - for case when restored vectors\
        should be transformed somehow before they will be used in the\
        optimization.

        """
        restore_files = []
        only_models = []
        structures = []
        if self.resume_dir is not None:
            for filename in os.listdir(self.resume_dir):
                file_path = os.path.join(self.resume_dir, filename)
                if filename.startswith(self.SAVE_FILENAME):
                    if self.settings.initial_structure is None:
                        if filename == self.SAVE_FILENAME:
                            restore_files.append(file_path)
                            break
                        continue
                    if len(filename) == len(self.SAVE_FILENAME):
                        warnings.warn(f"File {file_path} has name like saved"
                                      f" file of optimization but has no "
                                      f"structure at the end of the name. "
                                      f"So it is ignored.")
                        continue
                    strct_str = filename[len(self.SAVE_FILENAME)+1:]
                    structure = [int(x) for x in strct_str.split('_')]
                    if np.all(np.array(structure) >=
                              np.array(self.settings.initial_structure)):
                        restore_files.append(file_path)
                        structures.append(structure)

            if self.settings.initial_structure is not None:
                restore_files, structures = sort_by_other_list(
                    restore_files, structures, key=lambda x: sum(x))
            else:
                x_transform = (None, None)
                if self.settings.generate_x_transform:
                    x_transform = (ident_transform, ident_transform)
                return iter([(restore_files[0], None,
                              self.settings.only_models, x_transform)])

            some_file_not_valid = False
            for i in range(len(restore_files)):
                if some_file_not_valid:
                    restore_files[i] = None
                    only_models.append(False)
                    continue
                gs = self.global_optimizer.valid_restore_file(restore_files[i])
                ls = self.local_optimizer.valid_restore_file(restore_files[i])
                only_models.append(False)
                if not gs or not ls:
                    some_file_not_valid = True
            if self.settings.only_models:
                only_models.append(True)
                structures.append(structures[-1])

        if self.settings.initial_structure is None:
            return iter([(None, None, False, (None, None))])

        final_sum = sum(self.settings.final_structure)
        initial_sum = sum(self.settings.initial_structure)
        res_files, res_strct, res_bools, res_trans = [], [], [], []
        if self.settings.generate_x_transform:
            old_params = os.path.join(self.settings.resume_from, 'params_file')
            old_extra = os.path.join(self.settings.resume_from,
                                     'extra_params_file')
            if not check_file_existence(old_extra):
                old_extra = None
            old_settings = SettingsStorage.from_file(old_params, old_extra)
            old_init_model = old_settings.get_model()
        # additional one is when true only models is used
        addit_one = int(self.settings.only_models)
        if addit_one == 1:
            addit_one -= int(self.settings.resume_from is None)
        for i in range(final_sum - initial_sum + 1 + addit_one):
            if i >= len(restore_files):
                restore_file = None
            else:
                restore_file = restore_files[i]
            if i >= len(structures):
                structure = None
            else:
                structure = structures[i]
            if i >= len(only_models):
                restore_points_only = False
            else:
                restore_points_only = only_models[i]
            if not self.settings.generate_x_transform or restore_file is None:
                gs_x_transform = None
                ls_x_transform = None
            else:
                old_init_model = old_init_model.from_structure(structure)
                gs_x_transform = partial(
                    self.model.transform_values_from_other_model,
                    copy.deepcopy(old_init_model))
                copy_old_model = copy.deepcopy(old_init_model)
                copy_self_model = copy.deepcopy(self.model)
                copy_old_model.has_dyns = False
                copy_old_model = copy_old_model.from_structure(structure)
                copy_self_model.has_dyns = False
                copy_self_model = copy_self_model.from_structure(structure)
                ls_x_transform = partial(
                    copy_self_model.transform_values_from_other_model,
                    copy_old_model)

            res_files.append(restore_file)
            res_strct.append(structure)
            res_bools.append(restore_points_only)
            res_trans.append((gs_x_transform, ls_x_transform))
        return zip(res_files, res_strct, res_bools, res_trans)

    def run(self, initial_kwargs={}):
        """
        Main method of the class to run optimization. Runs
        :meth:`run_without_increase` or :meth:`run_with_increase` according to
        `settings`.
        """
        print(f'Run launch number {self.index}\n', end='')
        if self.index in self.shared_dict.dict:
            del self.shared_dict.dict[self.index]
        if self.settings.initial_structure is None:
            result = self.run_without_increase(initial_kwargs)
        else:
            result = self.run_with_increase(initial_kwargs)
        result.x = WeightedMetaArray(result.x)
        result.x.metadata = 'f'
        self.base_callback(result.x, result.y)
        self.intermediate_callback(result.x, result.y)
        # draw final model in output dir
        self.draw_model_in_output_dir(result.x, result.y)
        print(f'Finish genetic algorithm number {self.index}\n', end='')
        return result
