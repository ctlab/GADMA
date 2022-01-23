import unittest
import pytest

from .test_data import YRI_CEU_DATA
from gadma import *
from gadma.optimizers import *
from gadma.engines import *
from gadma.models import *
from gadma.cli.arg_parser import test_args
from gadma.core import SharedDictForCoreRun
from gadma.utils import ident_transform
if smac_available:
    from gadma.optimizers import smac_optim

import gadma
import dadi
import scipy
import shutil
import warnings
import os
import sys
import numpy as np
import pickle
import copy

warnings.filterwarnings(action='ignore', category=UserWarning,
                        module='.*\.optimizer', lineno=139)

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

def calc_func(x, y):
    x = np.array(x)
    return np.sum(x ** 4 + 2 * x ** 3 - 12 * x ** 2 - 2 * x + 6)

def get_func(engine, variables):
    def func(x, *args):
        y = - engine.evaluate(list(x), *args)
        return y
    return func

def get_1pop_sim_example_1(engine_id, args=(), data_size=4):
    """
    Classical bottleneck example.
    """
    t = TimeVariable('t')
    nu1 = PopulationSizeVariable('nu1')
    nu2 = PopulationSizeVariable('nu2')
    Nanc = PopulationSizeVariable("Nanc", units="physical")

    Nanc_size = None
    if engine_id not in ["dadi", "moments"]:
        Nanc_size = Nanc

    dm = EpochDemographicModel(Nanc_size=Nanc_size, mutation_rate=1e-8)

    dm.add_epoch(t, [nu1])
    dm.add_epoch(t, [nu2])
    values = {'Nanc': 10000, 'nu1': 0.1, 'nu2': 2, 't': 1.5}

    engine = get_engine(engine_id)
    engine.set_model(dm)
    seq_len = 1e6
    pops = None
    data = engine.simulate(values, [data_size], seq_len, pops, *args)
    engine.set_data(data)

    variables = dm.variables
    f = get_func(engine, variables)
    return f, variables


def get_1pop_sim_example_2(engine_id, args=(), data_size=4):
    """
    Exponential increase.
    """
    t = TimeVariable('t')
    nu1 = PopulationSizeVariable('nu1')
    Dyn = DynamicVariable('Dyn')
    Nanc = PopulationSizeVariable("Nanc", units="physical")

    Nanc_size = None
    if engine_id not in ["dadi", "moments"]:
        Nanc_size = Nanc
    if engine_id in ["momi"]:
        Dyn.domain = ["Sud", "Exp"]

    dm = EpochDemographicModel(Nanc_size=Nanc_size, mutation_rate=1e-8)

    dm.add_epoch(t, [nu1], dyn_args=[Dyn])
    values = {'Nanc': 10000, 'nu1': 5, 't': 1.5, 'Dyn': 'Exp'}

    engine = get_engine(engine_id)
    engine.set_model(dm)
    seq_len = 1e6
    pops = None
    data = engine.simulate(values, [data_size], seq_len, pops, *args)
    engine.set_data(data)

    variables = dm.variables
    f = get_func(engine, variables)
    return f, variables


def get_2pop_sim_example_1(engine_id, args=(), data_size=4):
    """
    Simple division.
    """
    t = TimeVariable('t')
    nu1 = PopulationSizeVariable('nu1')
    nu2 = PopulationSizeVariable('nu2')
    Nanc = PopulationSizeVariable("Nanc", units="physical")

    Nanc_size = None
    if engine_id not in ["dadi", "moments"]:
        Nanc_size = Nanc

    dm = EpochDemographicModel(Nanc_size=Nanc_size, mutation_rate=1e-8)

    dm.add_split(0, [nu1, nu2])
    dm.add_epoch(t, [nu1, nu2])
    values = {'Nanc': 10000, 'nu1': 1, 'nu2': 0.5, 't': 1.5}

    engine = get_engine(engine_id)
    engine.set_model(dm)
    seq_len = 1e6
    pops = None
    data = engine.simulate(values, [data_size, data_size], seq_len, pops, *args)
    engine.set_data(data)

    variables = dm.variables
    f = get_func(engine, variables)
    return f, variables

class TestBaseOptClass(unittest.TestCase):
    def test_not_implemented_error(self):
        opt = Optimizer()
        def f(x):
            return 10
        self.assertRaises(
            NotImplementedError,
            opt.optimize,
            f=f,
            variables=[TimeVariable("t")]
        )

    def check_class(self, cls):
        """
        Common checks for optmizers class
        """
        opt1 = cls()
        opt2 = cls(log_transform=True)
        opt3 = cls(maximize=True)
        opt4 = cls(log_transform=True, maximize=True)

        def f(x):
            return np.sin(x)[0]
        x = [0.5]
        y = f(x)
        variables = [ContinuousVariable("var", domain=[1, 2])]

        msg = f" ({cls.__name__})"
        self.assertEqual(opt1.evaluate(f=f, variables=variables, x=x),
                         y,
                         msg=msg)
        self.assertEqual(opt2.evaluate(f=f, variables=variables, x=np.log(x)),
                         y,
                         msg=msg)
        self.assertEqual(opt3.evaluate(f=f, variables=variables, x=x),
                         -y,
                         msg=msg)
        self.assertEqual(opt4.evaluate(f=f, variables=variables, x=np.log(x)),
                         -y,
                         msg=msg)

        # And test that if 0 is in domain log is not working
        variables = [ContinuousVariable("var", domain=[0, 2])]
        self.assertEqual(opt2.evaluate(f=f, variables=variables, x=x),
                         y,
                         msg=msg)
        self.assertEqual(opt4.evaluate(f=f, variables=variables, x=x),
                         -y,
                         msg=msg)

    def test_initialization(self):
        """
        Check that base classes initialize correctly
        """
        self.check_class(Optimizer)
        self.check_class(ContinuousOptimizer)
        self.check_class(ConstrainedOptimizer)
        self.check_class(UnconstrainedOptimizer)

    def test_evaluate_with_none_and_not_implemented_erros(self):
        def f(x):
            return None
        def g(x):
            return 10
        opt = Optimizer()
        opt.maximize = True
        self.assertEqual(opt.evaluate(f, [], []), np.inf)
        opt.maximize = False
        self.assertEqual(opt.evaluate(f, [], []), np.inf)
        opt.maximize = True
        self.assertEqual(opt.evaluate(g, [], []), -10)
        opt.maximize = False
        self.assertEqual(opt.evaluate(g, [], []), 10)

        self.assertRaises(NotImplementedError, opt.optimize, f,
                          [ContinuousVariable("var", domain=[0, 1])])
        self.assertRaises(NotImplementedError, opt.write_report,
                          variables=[], run_info=opt._create_run_info(),
                          report_file=None)

        opt = ContinuousOptimizer()
        self.assertRaises(AssertionError, opt.check_variables,
                          [DiscreteVariable("var", domain=[1, 2])])
        opt = UnconstrainedOptimizer()
        self.assertRaises(AssertionError, opt.check_variables,
                          [ContinuousVariable("var", domain=[1, 2])])

    def test_no_variables_func(self):
        def fixed_f(x):
            return 10
        for opt in list(all_global_optimizers()) + list(all_local_optimizers()):
            kwargs = {}
            if isinstance(opt, LocalOptimizer):
                kwargs["x0"] = []
            res = opt.optimize(fixed_f, variables=[], **kwargs)
            self.assertEqual(list(res.x), [])
            self.assertEqual(res.y, 10)
            self.assertEqual(res.n_iter, 0)
            self.assertEqual(res.n_eval, 1)


class TestLocalOpt(TestBaseOptClass):
    def test_not_implemented_error(self):
        opt = LocalOptimizer()
        def f(x):
            return 10
        self.assertRaises(
            NotImplementedError,
            opt.optimize,
            f=f,
            variables=[TimeVariable("t")],
            x0=[10]
        )

    def test_valid_restore_file(self):
        for opt in all_local_optimizers():
            output_file = os.path.join(DATA_PATH, "save_file")
            try:
                self.assertEqual(opt.valid_restore_file(output_file), False)

                with open(output_file, 'wb') as f:
                    pickle.dump([1, 2, 3], f)
                self.assertEqual(opt.valid_restore_file(output_file), False)

                opt.save(opt._create_run_info(), output_file)
                self.assertEqual(opt.valid_restore_file(output_file), True)
            finally:
                os.remove(output_file)

    def test_registered_local_optimizers_fails(self):
        self.assertRaises(ValueError, get_local_optimizer, 'some strange_id')
        ex_id = 'BFGS_log'
        opt = get_local_optimizer(ex_id)
        self.assertRaises(ValueError, register_local_optimizer,
                          ex_id, opt.__class__)
        self.assertRaises(ValueError, register_local_optimizer, 'id_ok', list)

    def test_initialization(self):
        """
        Check that local optimizers initialize correctly
        """
        self.check_class(LocalOptimizer)
        self.assertTrue(len(list(all_local_optimizers())) > 0)

        f = np.sin
        x = [0.5]
        y = f(x)
        variables = [ContinuousVariable('var', domain=[1e-5, 1])]

        for optim in all_local_optimizers():
            vars_tr = optim._prepare_variables(variables)
            self.assertTrue(vars_tr[0].correct_value(optim.transform(x)[0]))
            self.assertEqual(optim.evaluate(f=f, variables=vars_tr,
                                            x=optim.transform(x), args=()), y)

        self.assertRaises(ValueError, ScipyOptimizer, "strange_name")
        ScipyOptimizer.scipy_methods = ['method']
        opt = ScipyOptimizer('method')
        self.assertRaises(NotImplementedError, opt.get_addit_scipy_kwargs, '')

    def test_optimization_run(self):
        var1 = ContinuousVariable('var1', domain=[0, 1])
        var2 = ContinuousVariable('var2', domain=[1, 2])
        var3 = ContinuousVariable('var3', domain=[0, 20])
        variables = [var1, var2, var3]
        x0 = [0.5, 1.5, 10]
        eval_file = 'eval_file'
        save_file = 'save_file'
        report_file = 'report_file'
        if os.path.isfile(eval_file):
            os.remove(eval_file)
        if os.path.isfile(save_file):
            os.remove(save_file)
        if os.path.isfile(report_file):
            os.remove(report_file)

        def wrap(*args, **kwargs):
            wrap.counter += 1
            y = calc_func(*args, **kwargs)
            return y
        wrap.counter = 0

        for opt in all_local_optimizers():
            maxit = 30
            maxev = 10
            wrap.counter = 0
            res = opt.optimize(wrap, variables, x0, args=(5,),
                               verbose=1, maxiter=maxit, maxeval=maxev,
                               eval_file=eval_file,
                               save_file=save_file, report_file=report_file)
            self.assertTrue(calc_func(x0, 5) >= calc_func(res.x, 5))
            self.assertTrue(res.n_iter <= maxit)
            method = None
            if isinstance(opt, ScipyOptimizer):
                method = opt.method
                maxeval_kwarg = opt.maxeval_kwarg
            elif isinstance(opt, ManuallyConstrOptimizer):
                method = opt.optimizer.method
                maxeval_kwarg = opt.optimizer.maxeval_kwarg
            known_falues = ['Powell', 'Nelder-Mead']
            if method is not None and method in maxeval_kwarg:
                if method not in known_falues:
                    # sometimes optimizers do not stop at exactly maxeval so
                    # we allow to do 2x evaluations in test
                    self.assertTrue(res.n_eval <= 2 * maxev)
            self.assertEqual(res.n_eval, wrap.counter)
            self.assertTrue(os.path.getsize(eval_file) > 0)
            def nlines(filename):
                n_lines = 0
                with open(filename) as fl:
                    for line in fl:
                        n_lines += 1
                return n_lines
            self.assertEqual(res.n_eval + 1, nlines(eval_file))
            self.assertTrue(os.path.getsize(save_file) > 0)
            self.assertTrue(os.path.getsize(report_file) > 0)

            # maxiter = 0
            res2 = opt.optimize(wrap, variables, res.x, args=(5,),
                               verbose=1, maxiter=0, maxeval=maxev,
                               eval_file=eval_file,
                               save_file=save_file, report_file=report_file)
            self.assertEqual(res.y, res2.y)
            self.assertEqual(calc_func(res.x, 5), calc_func(res2.x, 5))
            self.assertEqual(res2.n_iter, 0)
            self.assertEqual(res2.n_eval, 1)

            os.remove(eval_file)
            os.remove(save_file)
            os.remove(report_file)        

#    def get_yri_ceu_ll(self, x_dict, ns=[20,20]):
#        def func(x_dict, ns, pts):
#            # Define the grid we'll use
#            xx = yy = dadi.Numerics.default_grid(pts)
#
#            # phi for the equilibrium ancestral population
#            phi = dadi.PhiManip.phi_1D(xx)
#            # Now do the population growth event.
#            phi = dadi.Integration.one_pop(phi, xx, x_dict['Tp'],
#                                           nu=x_dict['nu1F'])
#
#            # The divergence
#            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
#            # We need to define a function to describe the non-constant
#            # population 2 size. lambda is a convenient way to do so.
#            nu2_func = lambda t: x_dict['nu2B'] * (x_dict['nu2F'] / x_dict['nu2B'])**(t/x_dict['T'])  # NOQA
#            phi = dadi.Integration.two_pops(phi, xx, x_dict['T'],
#                                            nu1=x_dict['nu1F'], nu2=nu2_func,
#                                            m12=x_dict['m'], m21=x_dict['m'])
#
#            sfs = dadi.Spectrum.from_phi(phi, ns, (xx, yy))
#            return sfs
#
#        pts = [40, 50, 60]
#        data = dadi.Spectrum.from_file(YRI_CEU_DATA)
#        data = data.project(ns)
#        func_ex = dadi.Numerics.make_extrap_log_func(func)
#
#        model = func_ex(x_dict, data.sample_sizes, pts)
#        ll_model = dadi.Inference.ll_multinom(model, data)
#        return ll_model
#
#    def test_yri_ceu_example(self):
#        nu1F = PopulationSizeVariable('nu1F')
#        nu2B = PopulationSizeVariable('nu2B')
#        nu2F = PopulationSizeVariable('nu2F')
#        m = MigrationVariable('m')
#        Tp = TimeVariable('Tp')
#        T = TimeVariable('T')
#        Dyn = DynamicVariable('Dyn')
#
#        dm = EpochDemographicModel()
#        dm.add_epoch(Tp, [nu1F])
#        dm.add_split(0, [nu1F, nu2B])
#        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', 'Exp'])
#        dic = {'nu1F': 2.0, 'nu2B': 0.1, 'nu2F': 2, 'm': 1,
#               'Tp':  0.2, 'T': 0.2}
#
#        proj = (4, 4)
#        data = SFSDataHolder(YRI_CEU_DATA, projections=proj)
#        d = DadiEngine(model=dm, data=data)
#        values = [dic[var.name] for var in dm.variables]
#
#        d = get_engine('dadi')
#        d.set_data(data)
#        d.set_model(dm)
#
#        args = ([40, 50, 60],)
#        def f(x, *args):
#                y = - d.evaluate(list(x), *args)
#                #print(x, y)
#                return y
#
#        for opt in all_global_optimizers():
#            with self.subTest(optimizer=opt.id):
#                if hasattr(opt, "kernel_name"):
#                    opt.kernel_name = "auto"
#                res = opt.optimize(f, dm.variables, num_init=10,
#                                   args=args, maxeval=15, maxiter=3)
#        for opt_name in ["BFGS", "L-BFGS-B"]: #all_local_optimizers():
#            opt = get_local_optimizer(opt_name)
#            with self.subTest(local_optimizer=opt.id):
#                res = opt.optimize(f, dm.variables, x0=values,
#                                   args=args, maxiter=2)
#                self.assertEqual(res.y, f(res.x, *args))
#                self.assertEqual(res.y, -self.get_yri_ceu_ll(
#                    {var.name: val for var, val in zip(dm.variables, res.x)},
#                    proj))
#                self.assertTrue(res.y <= f(values, *args))
#
#    def run_example(self, engine_id, example_func, will_collapse=False):
#        args = ()
#        if engine_id == 'dadi':
#            args = ([40,50,60],)
#        f, variables = example_func(engine_id, args)
#        x0 = [var.resample() for var in variables]
#
#        def callback(x, y):
#            pass
#
##        save_file = 'save_file'
#        report_file = 'report_file'
#        eval_file = 'eval_file'
#
#        for opt in all_local_optimizers():
#            with self.subTest(local_optimizer=opt.id):
##                print(opt.id)
#                msg = f"(optimization {opt.id}, engine {engine_id})"
#                if will_collapse and opt.id != 'None' and opt.id != None:
#                    self.assertRaises(AssertionError, opt.optimize, f,
#                                      variables, x0=x0, args=args,
#                                      maxeval=10, maxiter=2,
#                                      callback=callback,
#                                      report_file=report_file,
##                                      save_file=save_file,
#                                      eval_file=eval_file)
#                else:
#                    res = opt.optimize(f, variables, x0=x0,
#                                       args=args, maxiter=2, maxeval=10,
#                                       callback=callback,
#                                       verbose=1,
#                                       report_file=report_file,
##                                       save_file=save_file,
#                                       eval_file=eval_file)
#                    self.assertEqual(res.y, f(res.x, *args), msg=msg)
#                    self.assertTrue(res.y <= f(x0, *args),
#                                    msg=msg + f" {res.y} > {f(x0, *args)}")
#
#    def test_1pop_example_1(self):
#        for engine in all_engines():
#            self.run_example(engine.id, get_1pop_sim_example_1)
#
#    def test_1pop_example_2(self):
#        for engine in all_engines():
#            self.run_example(engine.id, get_1pop_sim_example_2,
#                             will_collapse=True)
#
    def test_combinations_misses(self):
        ls_opt = get_local_optimizer("BFGS")
        ls_opt.maximize = False
        gs_opt = get_global_optimizer("Genetic_algorithm")
        gs_opt.maximize = True

        self.assertRaises(ValueError, GlobalOptimizerAndLocalOptimizer,
                          gs_opt, ls_opt)

        ls_opt.maximize = True
        # remove ids
        gs_opt = GlobalOptimizer()
        ls_opt = ManuallyConstrOptimizer(ScipyUnconstrOptimizer('BFGS'))
        self.assertRaises(AttributeError, gs_opt.__getattribute__, 'id')
        self.assertRaises(AttributeError, ls_opt.__getattribute__, 'id')

        opt = GlobalOptimizerAndLocalOptimizer(gs_opt, ls_opt)
        self.assertTrue(hasattr(opt.global_optimizer, 'id'))
        self.assertTrue(hasattr(opt.local_optimizer, 'id'))

        def f(x):
            return np.sum(x)

        gs_opt = get_global_optimizer("Genetic_algorithm")
        opt = GlobalOptimizerAndLocalOptimizer(gs_opt, ls_opt)
        variables = [ContinuousVariable(f"var{i}", domain=[0, 1])
                     for i in [1, 2]]

        opt.optimize(f, variables, verbose=0)

#    def test_2pop_example_1(self):
#        for engine in all_engines():
#            self.run_example(engine.id, get_2pop_sim_example_1)

class TestCoreRun(unittest.TestCase):
    @pytest.mark.timeout(0)
    def test_core_run(self):
        settings = test_args()
        settings.input_data = os.path.join(DATA_PATH, "DATA", "sfs",
                                           'small_1pop.fs')
        settings.draw_models_every_n_iteration = 100
        settings.print_models_code_every_n_iteration = 100
        settings.verbose = 10
        shared_dict = gadma.shared_dict.SharedDictForCoreRun(
            multiprocessing=False)
        gadma.core.core.job(0, shared_dict, settings)

        settings.custom_filename = os.path.join(DATA_PATH, "MODELS",
                                                "small_1pop_dem_model_dadi.py")
        settings.directory_with_bootstrap = os.path.join(
            DATA_PATH, "DATA", "sfs", 'small_1_pop_bootstrap')
        settings.read_bootstrap_data()
        settings.linked_snp_s = True
        settings.relative_parameters = True
        settings.pts = [4, 6, 8]
        settings.global_maxiter = 2
        settings.local_maxiter = 1
        shared_dict = gadma.shared_dict.SharedDictForCoreRun(
            multiprocessing=False)
        gadma.core.core.job(0, shared_dict, settings)

        for engine in all_engines():
            print(engine.id)
            if engine.id != "momentsLD":
                try:
                    settings = SettingsStorage()
                    settings.engine = engine.id
                    settings.output_directory = "Some_out_dir"
                    settings.num_init_const = 2
                    settings.global_maxiter = 4
                    settings.local_maxiter = 1
                    settings.model_plot_engine = "demes"
                    vcf_file = os.path.join(DATA_PATH, "DATA", "vcf", 'out_of_africa_chr22_sim.vcf')
                    popmap = os.path.join(DATA_PATH, "DATA", "vcf", 'out_of_africa_chr22_sim.popmap')
                    settings.input_data = f"{vcf_file}, {popmap}"
                    settings.is_valid()
                    gadma.core.core.job(0, shared_dict, settings)
                finally:
                    if check_dir_existence("Some_out_dir"):
                        shutil.rmtree("Some_out_dir")

            # best place to check
            if engine.id == "momi":
                try:
                    # prints warning that units will be years
                    settings.model_plot_engine = "momi"
                    settings.time_for_generation = 1
                    settings.units_of_time_in_drawing = "kya"
                    settings.selection = True
                    settings.is_valid()
                    self.assertEqual(settings.units_of_time_in_drawing, "years")
                    self.assertFalse(settings.selection)
                    settings.engine = "moments"
                    settings.no_migrations = False
                    settings.is_valid()
                finally:
                    if check_dir_existence("Some_out_dir"):
                        shutil.rmtree("Some_out_dir")

            elif engine.id == "momentsLD":
                try:
                    moments.LD.Inference._varcov_inv_cache = {}
                    settings = SettingsStorage()
                    settings.engine = engine.id
                    settings.output_directory = "Some_out_dir"
                    settings.num_init_const = 2
                    settings.global_maxiter = 1
                    settings.local_maxiter = 1
                    settings.only_sudden = True
                    settings.model_plot_engine = "demes"
                    settings.time_for_generation = 1
                    settings.initial_structure = [1, 1]
                    settings.mutation_rate = 1.5e8
                    settings.units_of_time_in_drawing = "kya"
                    vcf_file = os.path.join(
                        DATA_PATH, "DATA",
                        "vcf_ld", "small",
                        "data.vcf")
                    popmap = os.path.join(
                        DATA_PATH, "DATA",
                        "vcf_ld", "small",
                        "pop_map.txt")
                    settings.input_data = f"{vcf_file}, {popmap}"
                    settings.preprocessed_data = os.path.join(
                        DATA_PATH, "DATA",
                        "vcf_ld", "small",
                        "preprocessed_data.bp"
                    )
                    settings.is_valid()
                    gadma.core.core.job(0, shared_dict, settings)
                finally:
                    if check_dir_existence("Some_out_dir"):
                        shutil.rmtree("Some_out_dir")


    def test_core_run_restore(self):
        old_run_out = os.path.join(DATA_PATH, "my_example_run")
        sys.argv = ['gadma', "--resume", old_run_out]
        settings, _ = get_settings()
        settings.generate_x_transform = True
        settings.final_structure = [3, 4]
        shared_dict = SharedDictForCoreRun(False)

        try:
            # create fake files
            new_folder_of_4_run = os.path.join(old_run_out, "4")
            new_save_file = os.path.join(new_folder_of_4_run, "save_file")
            new_save_file_1_1 = os.path.join(new_folder_of_4_run,
                                             "save_file_1_1")
            new_save_file_2_1 = os.path.join(new_folder_of_4_run,
                                             "save_file_2_1")
            if not check_dir_existence(new_folder_of_4_run):
                os.mkdir(new_folder_of_4_run)
            open(new_save_file, 'w').close()
            open(new_save_file_2_1, 'w').close()
            shutil.copyfile(os.path.join(old_run_out, "1", "save_file_1_1"),
                            new_save_file_1_1)

            # check options of core_run
            for index in range(1, 5):
                core_run = CoreRun(index, shared_dict, settings)
                core_run.get_run_options()
        finally:
            if check_dir_existence(new_folder_of_4_run):
                shutil.rmtree(new_folder_of_4_run)
            if check_dir_existence(settings.output_directory):
                shutil.rmtree(settings.output_directory)


class TestLinearConstrains(unittest.TestCase):
    def test_linear_constrain(self):
        f = scipy.optimize.rosen
        variables = list()
        variables = [ContinuousVariable('var1', [-1, 2]),
                     ContinuousVariable('var2', [0, 1.5]),
                     ContinuousVariable('var3', [0.5, 4])]
        A = [[1, 0, 0],
             [2, 1, 0],
             [-1, 0, 1]]
        lb = [None, None, -0.5]
        ub = [1.5, 4, None]
        constrain = LinearConstrain(np.zeros_like(A),
                                    np.zeros_like(lb), np.zeros_like(ub))
        constrain.A = A
        constrain.lb = lb
        constrain.ub = ub
        self.assertTrue(np.allclose(constrain.A, np.array(A)))
        self.assertTrue(np.allclose(constrain.lb,
                                    np.array([-np.inf, -np.inf, -0.5])))
        self.assertTrue(np.allclose(constrain.ub,
                                    np.array([1.5, 4, np.inf])))
        constrain.__str__()
        x0 = [0, 0.5, 2]
        maxiter = 10
        for opt in all_local_optimizers():
            if opt.log_transform:
                continue
            opt.optimize(f, variables, x0, linear_constrain=constrain,
                         maxiter=maxiter)
            
        for opt in all_global_optimizers():
            if opt.log_transform:
                continue
            opt.optimize(f, variables, linear_constrain=constrain,
                         maxiter=maxiter)


class TestGlobalOptimizer(unittest.TestCase):
    def test_global_opt(self):
        def f(x):
            return np.sum(x)

        variables = [ContinuousVariable("v", domain=[0, 1])]
        X = [[variables[0].resample()] for _ in range(10)]
        ga = get_global_optimizer("Genetic_algorithm")
        ga.initial_design(f, variables, 10,
                          X_init=X, Y_init=None)

        opt = GlobalOptimizer()
        self.assertRaises(NotImplementedError,
                          opt.optimize, f, variables, num_init=10)
        self.assertRaises(NotImplementedError,
                          opt.write_report, variables, None, None)

    def test_implementations_in_instances(self):
        for opt in all_global_optimizers():
            run_info = opt._create_run_info()
            run_info_2 = opt._update_run_info(copy.deepcopy(run_info),
                                              x_best=[1, 2, 3], 
                                              y_best=10,
                                              X=[],
                                              Y=[],
                                              n_eval=2)
            self.assertEqual(run_info.result.n_iter + 1,
                             run_info_2.result.n_iter)
            self.assertEqual(run_info.result.n_eval + 2,
                             run_info_2.result.n_eval)
            self.assertNotEqual(run_info.result.x,
                                run_info_2.result.x)
            self.assertNotEqual(run_info.result.y,
                                run_info_2.result.y)

            run_info_3 = opt._apply_transform_to_run_info(
                run_info,
                x_transform=ident_transform,
                y_transform=ident_transform
            )

            self.assertFalse(opt.valid_restore_file(None))

    def test_update_of_initial_design(self):
        opt = get_global_optimizer("Genetic_algorithm")

        def f(x):
            return x * np.sin(x)
        X = [[np.random.uniform()] for _ in range(10)]
        Y = [f(x) for x in X]
        X_init, Y_init = [], []
        div = 5
        X_init = X[:div]
        Y_init = Y[:div]
        X_out = X[div:]
        Y_out = Y[div:]
        self.assertTrue(len(X_init) > 0)
        self.assertTrue(len(X_out) > 0)
        self.assertEqual(len(Y_init) + len(Y_out), len(Y))
        _X, _Y = opt._update_X_init_Y_init(X_init=X_init, Y_init=None, X_out=X_out, Y_out=Y_out)
        self.assertEqual(_Y, Y_out)
        self.assertEqual(_X[:len(X_out)], X_out)
        self.assertEqual(len(_X), len(X))

        _X, _Y = opt._update_X_init_Y_init(X_init=X_init, Y_init=Y_init[:-1], X_out=X_out, Y_out=Y_out)
        self.assertEqual(_Y[:len(Y_init)-1], Y_init[:-1])
        self.assertEqual(_Y[len(Y_init)-1:len(Y_out)-1+len(Y_out)], Y_out)
        self.assertEqual(len(_Y), len(Y)-1)
        self.assertEqual(_X[:len(Y_init)-1], X_init[:-1])
        self.assertEqual(_X[len(Y_init)-1:len(Y_out)-1+len(Y_out)], X_out)
        self.assertEqual(len(_X), len(X))

    def test_optimizer_run(self):
        f = scipy.optimize.rosen
        variables = [
            PopulationSizeVariable("nu", domain=[0, 2]),
            ContinuousVariable("var1", domain=[-1, 1.5]),
            DiscreteVariable("d1", domain=[0, 1]),
        ]

        for opt in all_global_optimizers():
            report_file = "report_file"
            if os.path.isfile(report_file):
                os.remove(report_file)

            kwargs = {
                "f": f,
                "variables": variables,
                "verbose": 1,
                "report_file": report_file,
                "num_init": 2,
            }
            if opt.id == "Genetic_algorithm":
                opt.gen_size = 2

            res1 = opt.optimize(**kwargs, maxiter=None, maxeval=5)
            res2 = opt.optimize(**kwargs, maxiter=5, maxeval=None)
            res3 = opt.optimize(**kwargs, maxiter=5, maxeval=5)

            self.assertTrue(res1.n_eval <= 5)
            self.assertTrue(res2.n_iter <= 5)
            self.assertTrue(res3.n_eval <= 5)
            self.assertTrue(res3.n_iter <= 5)

            os.remove(report_file)


class TestSMACOptimizations(unittest.TestCase):
    def test_smac_optimization(self):
        if not smac_available:  # We are successful when smac is not available
            return

        opt = get_global_optimizer("SMAC_BO_optimization")
        for kernel in ["exponential", "matern32", "matern52", "rbf", "auto"]:
            opt.kernel_name = kernel
        opt._kernel_name = "not_valid"
        self.assertRaises(ValueError, opt._get_kernel_class_and_nu)

        opt = get_global_optimizer("SMAC_BO_optimization")
        for acq in ["LogEI", "EI", "PI", "LCB"]:
            opt.acquisition_type = acq
        opt._acquisition_type = "not_valid"
        self.assertRaises(ValueError, opt.get_acquisition_function_class)

        opt = get_global_optimizer("SMAC_BO_optimization")
        opt.acquisition_type = "LogEI"
        variables = [DiscreteVariable("d", domain=[1, 2])]
        config_space = opt.get_config_space(variables)
        opt.get_kernel(config_space)
        scenario = opt.get_scenario(maxeval=1, config_space=config_space)
        opt.get_runhistory2epm(scenario=scenario)

    def test_smac_optim_file(self):
        if not smac_available:  # We are successful when smac is not available
            return
        import smac
        from smac.tae import StatusType
        opt = get_global_optimizer("SMAC_squirrel_optimization")
        variables = [DiscreteVariable("d1", domain=[1, 2, 3])]
        api_config, config_space = opt.get_configs(variables)
        self.assertRaises(ValueError, optimizers.smac_optim.SMAC4EPMOpimizer,
                          api_config, config_space, parallel_setting="not_valid")
        smac4epm = optimizers.smac_optim.SMAC4EPMOpimizer(api_config, config_space)
        smac4epm.combinations = smac4epm.combinations[:3]
        self.assertEqual(len(smac4epm.suggest()), 0)
        for i in [1, 2, np.inf]:
            smac4epm.runhistory.add(config=config_space.sample_configuration(),
                                    cost=100 * i,
                                    time=0,
                                    status=StatusType.SUCCESS,)
        smac4epm.parallel_setting = "not_valid"
        self.assertRaises(ValueError, smac4epm.suggest, 2)
        smac4epm.next_evaluations = []
        for parallel_setting in ["CL_min", "CL_max", "CL_mean", "LS", "KB"]:
            smac4epm.parallel_setting = parallel_setting
            smac4epm.suggest(2)
            smac4epm.next_evaluations = []
        model, acq, acq_opt, rh2epm = [], 1, 1, 1
        smac4epm.combinations.append((model, acq, acq_opt, rh2epm))
        self.assertRaises(ValueError, smac4epm.suggest, 10)
        smac4epm.combinations = smac4epm.combinations[:-1]
        model, acq, acq_opt, rh2epm = smac4epm.combinations[-1]
        rh2epm = []
        smac4epm.combinations.append((model, acq, acq_opt, rh2epm))
        smac4epm.next_evaluations = []
        self.assertRaises(ValueError, smac4epm.suggest, 10)

        api_config, config_space = opt.get_configs(variables=[])
        self.assertRaises(ValueError, optimizers.smac_optim.SMAC4EPMOpimizer,
                          api_config, config_space)
