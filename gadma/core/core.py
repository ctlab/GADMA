
from ..models import DemographicModel
from ..cli import arg_parser
from ..utils import parallel_wrap, StdAndFileLogger
from ..engines import get_engine

from functools import partial
import numpy as np
import os
import sys
import copy

from datetime import datetime
import operator
from collections import defaultdict
import multiprocessing
import signal
from multiprocessing import Manager, Pool
import math


def increase_structure(model, X, final_structure,
                       model_generator=DemographicModel.from_structure):
    cur_structure = model.get_structure()
    if (cur_structure == final_structure):
        return
    diff = np.array(final_structure) - np.array(cur_structure)
    struct_index = np.random.choice(np.arange(len(cur_structure))[diff != 0])
    event_index = np.random.choice(np.arange(cur_structure[struct_index]))
    event_index += sum(cur_structure)[:struct_index] - 1 + struct_index


    new_structure = copy.copy(cur_structure)
    new_structure[structure_index] += 1
    new_model = model_generator(new_structure)

    oldvar2newvar = {}
    for i, (old_event, new_event) in enumerate(zip(model.events, new_model.events)):
        if i >= struct_index:
            break
        for old_var, new_var in zip(old_event.variables, new_event.variables):
            oldvar2newvar[old_var] = new_var
    for old_event, new_event in zip(model.events, new_model.events)[event_index + 1:]:
        for old_var, new_var in zip(old_event.variables, new_event.variables):
            oldvar2newvar[old_var] = new_var
    new_X = []
    for x in X:
        var2value = model.var2value(x)
        new_var2value = {}
        for var in var2value:
            new_var2value[oldvar2newvar[var]] = var2value[var]
        event1 = new_model.events[event_index]
        event2 = new_model.events[event_index + 1]  # base event
        # Time / 2
        new_var2value[event2.time_arg] / 2
        new_var2value[event1.time_arg] = new_var2value[event2.time_arg]
        # Sizes
        for i, (size1, size2) in enumerate(zip(event1.size_args, event2.size_args)):
            if event2.dyn_args:
                dyn_value = new_var2value[event2.dyn_args[i]]
            else:
                dyn_value = 'Sud'

            if dyn_value == 'Sud':
                new_var2value[size1] = new_var2value[size2]
            else:
                func = DynamicVariable.get_func_from_value(dyn_value)
                y1 = new_var2value[size1]
                y2 = new_var2value[size2]
                x_diff = 2 * new_var2value[event2.time_arg]  # We have already divided it.
                size_func = func(y1, y2, x_diff)
                new_var2value[size1] = size_func[x_diff / 2]
        # Copy other variables
        for var1, var2 in zip(event1.variables, event2.variables):
            if var1 not in new_var2value:
                new_var2value[var1] = new_var2value[var2]
        new_X.append([new_var2value[var] for var in new_model.variables])
    return new_model, new_X


def callback(index, shared_dict, model, x, y):
    if index == 1:
        print(x, y)
    if index in shared_dict:
        model, x_best, y_dict = shared_dict[index]
        y_best = y_dict['log-likelihood']
    if index not in shared_dict or y < y_best:
        shared_dict[index] = (model, x, {'log-likelihood': y})

def run_go_and_lo(model, data, engine_id,
                 global_optimizer, local_optimizer=None,
                 global_kwargs={}, local_kwargs=(),
                 get_aic=False, get_claic=False):

    if engine_id == 'dadi':
        assert 'pts' in kwargs
    engine = get_engine(engine_id)
    engine.set_data(data)
    engine.set_model(model)

    # Run global search
    f = engine.evaluate
    variables = engine.model.variables
    if len(variables) == 0:
        return None
    global_result = global_optimizer.optimize(f, variables, **global_kwargs)

    print(global_result)
    # If there is no ocal optimizer then we stop here
    if not local_optimizer:
        return global_result

    # Run local search 
    x_best = copy.copy(global_result.x)
    fixed_model = copy.copy(model)
    fixed_model.fix_dynamics(x_best)
    is_fixed = np.array(fixed_model.is_fixed)
    engine.set_model(fixed_model)
    variables = engine.model.variables
    local_kwargs['x0'] = x_best[is_fixed == False]
    local_result = local_optimizer.optimize(f, variables, **local_kwargs)

    # Save result
    result = copy.deepcopy(global_result)
    result.y = local_result.y
    result.x[if_fixed == False] = local_result.x
    result.X_out, result.Y_out = sort_by_other_list(result.X_out,
                                                    result.Y_out)
    if np.all(result.X_out[0] == x_best):
        result.X_out[0] = result.x
    else:
        result.X_out.insert(0, result.x)
    return result

def run_pipeline_with_increase(index, shared_dict, model, data, engine_id,
                               final_structure,
                               global_optimizer, local_optimizer,
                               init_kwargs={}, common_kwargs={},
                               boots_for_claic=None):
    np.random.seed()
    print(f'Run launch number {index}')

    # First run
    common_kwargs['callback'] = partial(callback, index, shared_dict, model)
    if model.get_structure() == [1]:
        model, _ = increase_structure(model, final_structure, [])
    result = run_go_and_lo(model, data, engine_id,
                           global_optimizer, local_optimizer,
                           {**init_kwargs, **common_kwargs}, common_kwargs)

    while final_structure and (model.get_structure() != final_structure):
        model, X_init = increase_structure(model, final_structure, result.X_out)
        Y_init = copy.copy(result.Y_out)
        init_kwargs = {'X_init': X_init, 'Y_init': Y_init}
        common_args['callback'] = partial(callback, index, shared_dict, model)
        result = run_go_and_lo(model, data, engine_id, go, lo,
                               {**init_kwargs, **common_kwargs}, common_kwargs)

        # TODO: CLAIC AIC

    print(f'Finish genetic algorithm number {index}')
    return result
    
def write_several_runs_log(time, dem_model, X_dict, to_stdout=True, out_file=None):
    """
    Function for printing log of several runs.
    :param time: time of writing since launch.
    :param dem_model: demographic model of run.
    :param shared_dict: dict with all information.
    :param out_file: output file othervise output will be std sysout.
    """
    s = (time).total_seconds()
    time_str = f"\n[{s // 3600: 3d}:{s % 3600 // 60:2d}:{s % 60:2d}]" 
    write_to_stdout_and_file(time_str, to_stdout, out_file)

    results = [[ind, *shared_dict[ind]] for ind in shared_dict]
    print(results)
    # TODO: print
    sorted_results = sorted(results)
#    support.print_set_of_models(log_file, all_models, 
#                params, first_col='GA number', heading='All best logLL models:', silence=params.silence)


def worker_init():
    """Graceful way to interrupt all processes by Ctrl+C."""
    # ignore the SIGINT in sub process
    def sig_int(signal_num, frame):
        pass

    signal.signal(signal.SIGINT, sig_int)


#def print_best_solution_now(start_time, shared_dict, params,
#                            log_file, precision, draw_model):
#    """Prints best demographic model by logLL among all processes.
#
#    start_time :    time when equation was started.
#    shared_dict :   dictionary to share information between processes.
#    log_file :      file to write logs.
#    draw_model :    plot model best by logll and best by AIC (if needed).
#    """
#
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
    sys.stdout = StdAndFileLogger(log_file)
    sys.stderr = StdAndFileLogger(log_file)

    print("--Successful arguments' parsing. Start launch--")

    # Data reading
    print("Data reading")
    data = settings_storage.read_data()
    print("--Successful data reading--")

    # Save parameters
    settings_storage.to_files(params_file, extra_params_file)
    if not args.test:
        print(f"Parameters of launch are saved in output directory: "
              f"{params_file}")
        print(f"All aoutput is saved in output directory: {log_file}")

    print("--Run pipeline--")

    # Change output stream both to stdout and log file
    sys.stdout = StdAndFileLogger(log_file, settings_storage.silence)

    # Model
    model = settings_storage.get_model()
    # Optimizations
    global_optimizer = settings_storage.get_global_optimizer()
    local_optimizer = settings_storage.get_local_optimizer()
    # Kwargs
    common_kwargs = settings_storage.get_optimizers_kwargs()

    # For debug
#    run_genetic_algorithm((1, params, log_file, None))

    # Create shared dictionary
    m = Manager()
    shared_dict = m.dict()

    # Start pool of processes
    start_time = datetime.now()

    pool = Pool(processes=settings_storage.number_of_processes,
                initializer=worker_init)
    try:
        pool_map = pool.map_async(
            partial(parallel_wrap, run_pipeline_with_increase),
            [(i + 1, shared_dict, model, data, settings_storage.engine,
              settings_storage.final_structure, global_optimizer,
              local_optimizer, {}, common_kwargs)
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
                    start_time, shared_dict, params, log_file, 
                    precision, draw_model=params.matplotlib_available)
            except Exception as e:
                pool.terminate()
                raise RuntimeError(str(e))
#        print_best_solution_now(start_time, shared_dict, params,log_file, 
#                precision, draw_model=params.matplotlib_available)

        sys.stdout = StdAndFileLogger(log_file)

        print('\n--Finish pipeline--\n')
        if params.test:
            print('--Test passed correctly--')
        if params.theta is None:
            print("\nYou didn't specify theta at the beginning. If you want change it and rescale parameters, please see tutorial.\n")
#        if params.resume_dir is not None and (params.initial_structure != params.final_structure).any():
#            print('\nYou have resumed from another launch. Please, check best AIC model, as information about it was lost.\n')

        print('Thank you for using GADMA!')

    except KeyboardInterrupt:
        pool.terminate()
        support.error('KeyboardInterrupt')

if __name__ == '__main__':
    main()
