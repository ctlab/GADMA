
from ..models import DemographicModel, Split
from ..cli import arg_parser
from ..utils import parallel_wrap, StdAndFileLogger, get_aic_score
from ..utils import sort_by_other_list, ensure_dir_existence,\
ensure_file_existence
from ..utils import TimeVariable, PopulationSizeVariable, SelectionVariable,\
DynamicVariable
from ..engines import get_engine
from ..optimizers import GlobalOptimizerAndLocalOptimizer

from functools import partial
import numpy as np
import os
import sys
import copy

from datetime import datetime
import operator
from collections import defaultdict, OrderedDict
import multiprocessing
import signal
from multiprocessing import Manager, Pool
import math


class CoreRun(object):
    def __init__(self, index, shared_dict, settings):
        self.index = index
        self.shared_dict = shared_dict
        self.settings = settings

        # Take something from settings
        self.engine = get_engine(self.settings.engine)
        self.data = self.settings.inner_data
        self.model = self.settings.get_model()
        # We save data_holder in engine for good code generation
        self.engine.data_holder = self.settings.data_holder

        self.global_optimizer = self.settings.get_global_optimizer()
        self.local_optimizer = self.settings.get_local_optimizer()
        self.optimize_kwargs = self.settings.get_optimizers_kwargs()
        self.optimize_kwargs['callback'] = self.callback

        self.aic_score = not self.settings.linked_snp_s
        self.claic_score = False#not self.aic_score and self.settings.bootstrap_directory

        # Create all neccessary output dirs and files
        if settings.output_directory is not None:
            self.output_dir = os.path.join(settings.output_directory,
                                           str(self.index))
            self.output_dir = ensure_dir_existence(self.output_dir)
            self.eval_file = os.path.join(self.output_dir, 'eval_file')
            self.eval_file = ensure_file_existence(self.eval_file)
            self.report_file = os.path.join(self.output_dir, 'GADMA_GA.log')
            self.report_file = ensure_file_existence(self.report_file)
            self.optimize_kwargs['eval_file'] = self.eval_file
            self.optimize_kwargs['report_file'] = self.report_file

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

    def callback(self, x, y):
        best_by = 'log-likelihood'
        if self.index in self.shared_dict:
            engine, x_best, y_dict = self.shared_dict[self.index][best_by]
            y_best = y_dict[best_by]
        if (self.index not in self.shared_dict or
                np.allclose(y, y_best) or y > y_best):
            if self.index not in self.shared_dict:   
                new_dict = OrderedDict()
            else:
                new_dict = OrderedDict(self.shared_dict[self.index])
            new_dict[best_by] = (copy.deepcopy(self.engine), x,
                                 OrderedDict({best_by: y}))
            if self.aic_score:
                n_params = self.engine.model.get_number_of_parameters(x)
                new_dict[best_by][2]['AIC score'] = get_aic_score(n_params, y)
            self.shared_dict[self.index] = new_dict

    def final_callback(self, x, y):
        if False:#self.aic_score:
            best_by = "AIC score"
            n_params = self.engine.model.get_number_of_parameters(x)
            value = get_aic_score(n_params, y)
        elif self.claic_score:
            pass
            # TODO
        else:
            return
        if self.index not in self.shared_dict:
            self.shared_dict[self.index] = OrderedDict()
        new_dict = OrderedDict(self.shared_dict[self.index])
        if best_by not in self.shared_dict[self.index]:
            new_dict[best_by] = OrderedDict()
        else:
            engine, x_best, y_dict = new_dict[best_by]
            if y_dict[best_by] < value:
                return
        y_dict = OrderedDict()
        y_dict['log-likelihood'] = y
        y_dict[best_by] = value
        new_dict[best_by] = (copy.deepcopy(self.engine), x, y_dict)
        self.shared_dict[self.index] = new_dict

    def run_without_increase(self):
        np.random.seed()
        self.optimize_kwargs['callback'] = self.callback
        f = self.engine.evaluate
        variables = self.model.variables

        optimizer = GlobalOptimizerAndLocalOptimizer(self.global_optimizer,
                                                     self.local_optimizer)

        result = optimizer.optimize(f, variables, **self.optimize_kwargs)
        self.final_callback(result.x, result.y)
        return result

    def run_with_increase(self):
        np.random.seed()
        # Simple checks
        assert self.settings.initial_structure is not None
        assert self.settings.final_structure is not None

        print(f'Run launch number {self.index}\n', end='')
        result = self.run_without_increase()
        while (self.model.get_structure() != self.settings.final_structure):
            self.model, X_init = self.model.increase_structure(result.X_out)
            Y_init = copy.copy(result.Y_out)
            self.optimize_kwargs['X_init'] = X_init
            self.optimize_kwargs['Y_init'] = Y_init
            result = self.run_without_increase()
        print(f'Finish genetic algorithm number {self.index}\n', end='')
        return result



def job(index, shared_dict, settings):
    obj = CoreRun(index, shared_dict, settings)
    obj.run_with_increase()

def worker_init():
    """Graceful way to interrupt all processes by Ctrl+C."""
    # ignore the SIGINT in sub process
    def sig_int(signal_num, frame):
        pass

    signal.signal(signal.SIGINT, sig_int)


def print_best_solution_now(start_time, shared_dict, settings,
                            log_file, precision, draw_model):
    """Prints best demographic model by logLL among all processes.

    start_time :    time when equation was started.
    shared_dict :   dictionary to share information between processes.
    log_file :      file to write logs.
    draw_model :    plot model best by logll and best by AIC (if needed).
    """
    s = (datetime.now() - start_time).total_seconds()
    time_str = f"\n[{int(s//3600):03}:{int(s%3600//60):02}:{int(s%60):02}]" 
    print(time_str)
    metric_names = list()  # ordered set
    for index in shared_dict:
        for name in shared_dict[index]:
            if name not in metric_names:
                metric_names.append(name)
    for best_by in metric_names:
        models = [(index, shared_dict[index][best_by])
                  for index in shared_dict]
        sorted_models = sorted(models, key=lambda x: x[1][2][best_by])
        if best_by == 'log-likelihood':
            sorted_models = list(reversed(sorted_models))
        metrics = list()  # ordered set
        for model in sorted_models:
            for key in model[1][2]:
                if key not in metrics:
                    metrics.append(key)
        print(f"All best by {best_by} models")
        print("Number", *metrics, "Model", sep='\t')
        for model in sorted_models:
            index, info = model
            engine, x, y_vals = info
            # Get theta and N ancestral
            theta = engine.get_theta(x, *settings.get_engine_kwargs()['args'])
            Nanc = engine.get_N_ancestral(theta)
            addit_str = f"(theta = {theta: .2f})"
            if Nanc is not None:
                addit_str += f" (Nanc = {Nanc: .0f})"
            # Begin to print
            metric_vals = []
            for metr in metrics:
                if metr not in y_vals:
                    metric_vals.append("None")
                else:
                    metric_vals.append(f"{y_vals[metr]: .5f}")
            print(f"Run {index}", *metric_vals,
                  engine.model.as_custom_string(x),
                  addit_str, sep='\t')
            #print(engine.generate_code(x, None, *settings.get_engine_kwargs()['args']))
        
        
#    def write_func(string): return support.write_log(log_file, string,
#                                                     write_to_stdout=not params.silence)
#
#    def my_str(x): return support.float_representation(x, precision)
#
#
#    has_aic_or_claic = params.model_func_file is None and (params.final_structure != params.initial_structure).any()
#    has_aic = not params.linked_snp and has_aic_or_claic
#    has_claic =  params.linked_snp and has_aic_or_claic and params.boot_dir is not None
#    
#
#    all_models_data = dict(shared_dict)
#    if not all_models_data:
#        return
#    all_models = [(i, all_models_data[i][0]) for i in all_models_data]
#    all_models = sorted(all_models, key=lambda x: x[
#                        1].get_fitness_func_value())
#
#    s = (datetime.now() - start_time).total_seconds()
#    write_func('\n[%(hours)03d:%(minutes)02d:%(seconds)02d]' % {
#        'hours': s / 3600,
#        'minutes': s % 3600 / 60,
#        'seconds': s % 60
#    })
#
#    support.print_set_of_models(log_file, all_models, 
#                params, first_col='GA number', heading='All best logLL models:', silence=params.silence)
#
#    if has_aic:
#        all_aic_models = []
#        for i in all_models_data:
#            best_model, final_models = all_models_data[i]
#            all_aic_models.append((i, best_model))
#            for model in final_models:
#                if model.get_aic_score() < best_model.get_aic_score():
#                    all_aic_models[-1] = (i, model)
#
#        all_aic_models = sorted(all_aic_models, key=lambda x: x[1].get_aic_score())
#        support.print_set_of_models(log_file, all_aic_models, 
#                params, first_col='GA number', heading='\nAll best AIC models:', silence=params.silence)
#    if has_claic:
#        all_claic_models = []
#        for i in all_models_data:
#            best_model, final_models = all_models_data[i]
#            for final_model in final_models:
#                all_claic_models.append((i, final_model))
##            if len(final_models) > 0:
##                all_claic_models.append((i, final_models[0]))
##            for model in final_models:
##                if model.get_claic_score() < best_model.get_claic_score():
##                    all_claic_models[-1] = (i, model)
#
#        if len(all_claic_models) != 0:
#            all_claic_models = sorted(all_claic_models, key=lambda x: x[1].get_claic_score())
#            support.print_set_of_models(log_file, all_claic_models, 
#                    params, first_col='GA number', heading='\nAll intermediate and final models (with CLAIC):', silence=params.silence)
#
#    support.print_best_logll_model_long(log_file, all_models[0][1], params, silence=params.silence)
#
#    if has_aic:
#        support.print_best_aic_model_long(log_file, all_aic_models[0][1], params, silence=params.silence)
#
#    if draw_model:
#        support.save_model_plot(os.path.join(params.output_dir, 'best_logLL.png'), all_models[0][1], params, title='')
#
#        if has_aic:
#            support.save_model_plot(os.path.join(params.output_dir, 'best_aic.png'), all_aic_models[0][1], params, title='')
#
#    support.print_model_code(params.output_dir, all_models[0][1], params, prefix='best_logLL_model')
#    
#    if has_aic:
#        support.print_model_code(params.output_dir, all_aic_models[0][1], params, prefix='best_aic_model')
#
#    if not params.test:
#        write_func(
#            '\nYou can find its picture and python code in output directory')


def main():
    settings_storage, args = arg_parser.get_settings()

    # Form output directory
    log_file = os.path.join(settings_storage.output_directory, 'GADMA.log')
    params_file = os.path.join(settings_storage.output_directory,
                              'params_file')
    extra_params_file = os.path.join(settings_storage.output_directory,
                              'extra_params_file')

    # Change output stream both to stdout and log file
    saved_stdout = sys.stdout
    sys.stdout = StdAndFileLogger(log_file)
#    sys.stderr = StdAndFileLogger(log_file)

    print("--Successful arguments parsing--")

    # Data reading
    print("Data reading")
    data = settings_storage.read_data()
    print("--Successful data reading--")

    # Save parameters
    settings_storage.to_files(params_file, extra_params_file)
    if not args.test:
        print(f"Parameters of launch are saved in output directory: "
              f"{params_file}")
        print(f"All output is saved in output directory: {log_file}")

    print("--Start pipeline--")

    # Change output stream both to stdout and log file
    sys.stdout = saved_stdout
    sys.stdout = StdAndFileLogger(log_file, settings_storage.silence)

    # Create shared dictionary
    m = Manager()
    shared_dict = m.dict()

    # Start pool of processes
    start_time = datetime.now()

    # For debug
#    run_pipeline_with_increase(0, None, model, data, settings_storage.engine,
#              settings_storage.final_structure, global_optimizer,
#              local_optimizer, {}, common_kwargs, None, os.path.join(settings_storage.output_directory, '0'))
#
    pool = Pool(processes=settings_storage.number_of_processes,
                initializer=worker_init)
    try:
        pool_map = pool.map_async(
            partial(parallel_wrap, job),
            [(i + 1, shared_dict, settings_storage)
             for i in range(settings_storage.number_of_repeats)])
        pool.close()

        precision = 1 - int(math.log(settings_storage.eps, 10))

        # graceful way to interrupt all processes by Ctrl+C
        min_counter = 0
        while True:
            try:
                multiple_results = pool_map.get(60)
#                    60 * settings_storage.time_to_print_summary)
                break
            # catch TimeoutError and get again
            except multiprocessing.TimeoutError as ex:
                print_best_solution_now(
                    start_time, shared_dict, settings_storage, None, 
                    precision, None)
            except Exception as e:
                pool.terminate()
                raise RuntimeError(str(e))
        print_best_solution_now(start_time, shared_dict, settings_storage, None, 
                precision, None)

        sys.stdout = StdAndFileLogger(log_file)

        print('\n--Finish pipeline--\n')
        if args.test:
            print('--Test passed correctly--')
        if settings_storage.theta0 is None:
            print("\nYou didn't specify theta at the beginning. If you want change it and rescale parameters, please see tutorial.\n")
#        if params.resume_dir is not None and (params.initial_structure != params.final_structure).any():
#            print('\nYou have resumed from another launch. Please, check best AIC model, as information about it was lost.\n')

        print('Thank you for using GADMA!')

    except KeyboardInterrupt as e:
        pool.terminate()
        raise KeyboardInterrupt(e)

if __name__ == '__main__':
    main()
