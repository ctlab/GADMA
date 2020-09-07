from ..engines import get_engine, all_engines
from ..utils import sort_by_other_list, ensure_dir_existence,\
ensure_file_existence
from ..utils import TimeVariable, PopulationSizeVariable, SelectionVariable,\
DynamicVariable
from ..optimizers import GlobalOptimizerAndLocalOptimizer
from ..utils import get_aic_score, get_claic_score
from ..models import EpochDemographicModel
from .draw_and_generate_code import draw_plots_to_file, generate_code_to_file
import os
import numpy as np
from collections import defaultdict, OrderedDict
import copy

class CoreRun(object):
    """
    Class of main run in GADMA.
    Has a :meth:`run` method to start launch.

    :param index: Index of the run. Like id.
    :type index: int
    :param shared_dict: Dictionary to save results in callbacks. Will be saved
                        with key equal to :param:`index`. Is used for
                        multiprocessing cooperation.
    :type shared_dict: dict
    :param settings: Settings of the run. Information to form output directory
                     and so on will be taken from settings.
    :type settings: :class:`gadma.SettingsStorage`
    """
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
#        self.optimize_kwargs['linear_constrain'] = self.settings.get_linear_constrain(self.engine)
        self.optimize_kwargs['callback'] = self.callback

        # 2.3 Check if we need to calculate and keep additional functions
        # values in callbacks during optimization.
        self.aic_score = not self.settings.linked_snp_s
        self.claic_score = (not self.aic_score and
            (self.settings.directory_with_bootstrap is not None))
        self.boots = self.settings.bootstrap_data

        # 2.4 Create all necessary output dirs and files
        if settings.output_directory is not None:
            self.output_dir = os.path.join(settings.output_directory,
                                           str(self.index))
            self.output_dir = ensure_dir_existence(self.output_dir)
            self.eval_file = os.path.join(self.output_dir, 'eval_file')
            self.eval_file = ensure_file_existence(self.eval_file)
            self.report_file = os.path.join(self.output_dir, 'GADMA_GA.log')
            self.report_file = ensure_file_existence(self.report_file)
            # Tell optimizers about output files
            self.optimize_kwargs['eval_file'] = self.eval_file
            self.optimize_kwargs['report_file'] = self.report_file
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
                    for engine in all_engines():
                        engine_dir = os.path.join(self.code_dir, engine.id)
                        ensure_dir_existence(engine_dir)
        # Set counters to zero for callbacks to count number of their calls
        self.draw_iter_callback_counter = 0
        self.code_iter_callback_counter = 0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.engine.set_model(model)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.engine.set_data(data)

    def base_callback(self, x, y):
        """
        Base callback:

        1) Updates values of best solution in :attr:`shared_dict`.

        2)If new values are received then draws and generates code to the
        :attr:`output_dir`.
        """
        best_by = 'log-likelihood'
        if self.aic_score:
            n_params = self.engine.model.get_number_of_parameters(x)
            aic = get_aic_score(n_params, y)
            y_dict = {best_by: y, 'AIC score': aic}
        else:
            y_dict = y
        updated = self.shared_dict.update_best_model_for_process(self.index,
                                                                 best_by,
                                                                 self.engine,
                                                                 x, y_dict)
#        if self.index in self.shared_dict:
#            engine, x_best, y_dict = self.shared_dict[self.index][best_by]
#            y_best = y_dict[best_by]
#        if (self.index not in self.shared_dict or
#                np.allclose(y, y_best) or y > y_best):
#            if self.index not in self.shared_dict:   
#                new_dict = OrderedDict()
#            else:
#                new_dict = OrderedDict(self.shared_dict[self.index])
#            new_dict[best_by] = (copy.deepcopy(self.engine), x,
#                                 OrderedDict({best_by: y}))
#            print("base", new_dict[best_by][1])
#            if self.aic_score:
#                n_params = self.engine.model.get_number_of_parameters(x)
#                new_dict[best_by][2]['AIC score'] = get_aic_score(n_params, y)
#            self.shared_dict[self.index] = new_dict
#            print("base", self.shared_dict[self.index][best_by][1])
            # Draw and generate code for current best model
        if updated:
#            fig_title = f"Best by {best_by} model. {best_by}: {y: .2f}"
            prefix = (self.settings.LOCAL_OUTPUT_DIR_PREFIX +
                      self.settings.LONG_NAME_2_SHORT.get(best_by, best_by))
#            save_plot_file = os.path.join(self.output_dir, prefix + "_model.png")
            save_code_file = os.path.join(self.output_dir, prefix + "_model.py")
#            try:
#               draw_plots_to_file(x, self.engine, self.settings, save_plot_file, fig_title)
#            except Exception as e:
#                pass
            try:
                generate_code_to_file(x, self.engine, self.settings, save_code_file)
            except Exception as e:
                pass

    def draw_iter_callback(self, x, y):
        """
        Draws best model on current iteration in the pictures directory.
        It happens every :attr:`self.settings.draw_models_every_n_iteration`
        iteration.
        """
        n_iter = self.draw_iter_callback_counter
        verbose = self.settings.draw_models_every_n_iteration
        if verbose != 0 and n_iter % verbose == 0:
            fig_title = f"Iteration {n_iter}, "\
                        f"Log-likelihood: {y:.2f}"
            save_file = os.path.join(self.pictures_dir,
                                     f"iteration_{n_iter}.png")
            draw_plots_to_file(x, self.engine, self.settings, save_file, fig_title)
        self.draw_iter_callback_counter += 1

    def code_iter_callback(self, x, y):
        """
        Generates code of best model on current iteration to the code
        directory. It happens every
        :attr:`self.settings.print_models_code_every_n_iteration` iteration.
        """
        n_iter = self.code_iter_callback_counter
        verbose = self.settings.print_models_code_every_n_iteration
        filename = f"iteration_{n_iter}.py"
        if verbose != 0 and n_iter % verbose == 0:
            if isinstance(self.model, EpochDemographicModel):
                for engine in all_engines():
                    save_file = os.path.join(self.code_dir, engine.id,
                                             filename)
                    args = self.settings.get_engine_args(engine.id)
                    engine.set_data(self.engine.data)
                    engine.data_holder = self.engine.data_holder
                    engine.set_model(self.engine.model)
                    engine.generate_code(x, save_file, *args)
            else:
                save_file = os.path.join(self.code_dir, filename)
                args = self.settings.get_engine_args()
                self.engine.generate_code(x, save_file, *args)
        self.code_iter_callback_counter += 1

    def callback(self, x, y):
        """
        Main callback for optimizers to get.
        """
        self.base_callback(x, y)
        try:
            self.draw_iter_callback(x, y)
        except Exception as e:
            pass
        try:
            self.code_iter_callback(x, y)
        except Exception as e:
            pass

    def intermediate_callback(self, x, y):
        """
        Almost final callback that is called after each global + local
        optimization.

        Saves AIC and CLAIC values in the :attr:`shared_dict` if needed.
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
#        if self.index not in self.shared_dict:
#            self.shared_dict[self.index] = OrderedDict()
#        new_dict = OrderedDict(self.shared_dict[self.index])
#        if best_by not in self.shared_dict[self.index]:
#            new_dict[best_by] = OrderedDict()
#        else:
#            if not self.claic_score:                
#                engine, x_best, y_dict = new_dict[best_by]
#                if y_dict[best_by] < value:
#                    return
#        y_dict = OrderedDict()
#        y_dict['log-likelihood'] = y
#        y_dict[best_by] = value
#        element = (copy.deepcopy(self.engine), x, y_dict) 
#        if not self.claic_score:
#            new_dict[best_by] = element
#        else:
#            if best_by not in self.shared_dict[self.index]:
#                new_list = list()
#            else:
#                new_list = copy.deepcopy(
#                    self.shared_dict[self.index][best_by])
#            new_list.append(element)
#            new_dict[best_by] = new_list
#        self.shared_dict[self.index] = new_dict

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
        save_code_file = os.path.join(self.output_dir, prefix + "_model.py")
        try:
            draw_plots_to_file(x, self.engine, self.settings, save_plot_file, fig_title)
        except Exception as e:
            print(f"{bcolors.FAIL}Run {index}: failed to draw model due to "
                  f"the following exception: {e}{bcolors.ENDC}")
        try:
            generate_code_to_file(x, self.engine, self.settings, save_code_file)
        except Exception as e:
            print(f"{bcolors.FAIL}Run {index}: failed to generate code due to"
                  f" the following exception: {e}{bcolors.ENDC}")

    def draw_model_in_output_dir(self, x, y,
                                 best_by='log-likelihood', final=True):
        fig_title = f"Best by {best_by} model. {best_by}: {y: .2f}"
        if final:
            prefix = self.settings.LOCAL_OUTPUT_DIR_PREFIX_FINAL
        else:
            prefix = self.settings.LOCAL_OUTPUT_DIR_PREFIX
        prefix += self.settings.LONG_NAME_2_SHORT.get(best_by, best_by)
        save_plot_file = os.path.join(self.output_dir, prefix + "_model.png")
        save_code_file = os.path.join(self.output_dir, prefix + "_model.py")
        try:
            generate_code_to_file(x, self.engine, self.settings, save_code_file)
        except Exception as e:
            pass
        try:
           draw_plots_to_file(x, self.engine, self.settings, save_plot_file, fig_title)
        except Exception as e:
            pass

    def run_without_increase(self, initial_kwargs={}):
        np.random.seed()
        self.optimize_kwargs['callback'] = self.callback
        f = self.engine.evaluate
        variables = self.model.variables

        optimizer = GlobalOptimizerAndLocalOptimizer(self.global_optimizer,
                                                     self.local_optimizer)

        result = optimizer.optimize(f, variables, **self.optimize_kwargs)
        self.intermediate_callback(result.x, result.y)
        return result

    def run_with_increase(self, initial_kwargs={}):
        np.random.seed()
        # Simple checks
        assert self.settings.initial_structure is not None
        assert self.settings.final_structure is not None

        result = self.run_without_increase(initial_kwargs)
        while (self.model.get_structure() != self.settings.final_structure):
            self.model, X_init = self.model.increase_structure(result.X_out)
            Y_init = copy.copy(result.Y_out)
            self.optimize_kwargs['X_init'] = X_init
            self.optimize_kwargs['Y_init'] = Y_init
            result = self.run_without_increase()
#        self.final_callback()
        return result

    def run(self, initial_kwargs={}):
        print(f'Run launch number {self.index}\n', end='')
        if self.settings.initial_structure is None:
            result = self.run_without_increase(initial_kwargs)
        else:
            result = self.run_with_increase(initial_kwargs)
        # draw final model in output dir
        self.draw_model_in_output_dir(result.x, result.y)
        print(f'Finish genetic algorithm number {self.index}\n', end='')
        return result

