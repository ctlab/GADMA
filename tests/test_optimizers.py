import unittest

from .test_data import YRI_CEU_DATA
from gadma import *
import gadma
import dadi
import scipy
import shutil
import warnings

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

    dm = EpochDemographicModel()
    dm.add_epoch(t, [nu1])
    dm.add_epoch(t, [nu2])
    values = {'nu1': 0.1, 'nu2': 2, 't': 1.5}

    engine = get_engine(engine_id)
    engine.set_model(dm)
    data = engine.simulate(values, [data_size], *args)
    engine.set_data(data)

    variables = dm.variables
    f = get_func(engine, variables)
    return f, variables


def get_1pop_sim_example_2(engine_id, args=(), data_size=4):
    """
    Exponential incease.
    """
    t = TimeVariable('t')
    nu1 = PopulationSizeVariable('nu1')
    Dyn = DynamicVariable('Dyn')

    dm = EpochDemographicModel()
    dm.add_epoch(t, [nu1], dyn_args=[Dyn])
    values = {'nu1': 5, 't': 1.5, 'Dyn': 'Exp'}

    engine = get_engine(engine_id)
    engine.set_model(dm)
    data = engine.simulate(values, [data_size], *args)
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

    dm = EpochDemographicModel()
    dm.add_split(0, [nu1, nu2])
    dm.add_epoch(t, [nu1, nu2])
    values = {'nu1': 1, 'nu2': 0.5, 't': 1.5}

    engine = get_engine(engine_id)
    engine.set_model(dm)
    data = engine.simulate(values, [data_size, data_size], *args)
    engine.set_data(data)

    variables = dm.variables
    f = get_func(engine, variables)
    return f, variables

class TestBaseOptClass(unittest.TestCase):
    def check_class(self, cls):
        """
        Common checks for optmizers class
        """
        opt1 = cls()
        opt2 = cls(log_transform=True)
        opt3 = cls(maximize=True)
        opt4 = cls(log_transform=True, maximize=True)

        f = np.sin
        x = 0.5
        y = f(x)

        msg = f" ({cls.__name__})"
        self.assertEqual(opt1.evaluate(f, x), y, msg=msg)
        self.assertEqual(opt2.evaluate(f, np.log(x)), y, msg=msg)
        self.assertEqual(opt3.evaluate(f, x), -y, msg=msg)
        self.assertEqual(opt4.evaluate(f, np.log(x)), -y, msg=msg)

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
        opt = Optimizer()
        opt.maximize = True
        self.assertEqual(opt.evaluate(f, []), -np.inf)

        self.assertRaises(NotImplementedError, opt.valid_restore_file, 'file')
        self.assertRaises(NotImplementedError, opt.optimize, f, [])
        opt.write_report(0, [], [], 10, report_file=None)


class TestLocalOpt(TestBaseOptClass):
    def test_not_implemented_error(self):
        opt = LocalOptimizer()
        self.assertRaises(NotImplementedError, opt.optimize, 'f', 'vars', 'x0')

    def test_valid_restore_file(self):
        opt = get_local_optimizer("None")
        output_file = os.path.join(DATA_PATH, "save_file")
        try:
            self.assertEqual(opt.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump([1, 2, 3], f)
            self.assertEqual(opt.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump((1, 2, 3), f)
            self.assertEqual(opt.valid_restore_file(output_file), False)
        finally:
            os.remove(output_file)

        opt = get_local_optimizer("BFGS")
        try:
            self.assertEqual(opt.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump([1, 2, 3], f)
            self.assertEqual(opt.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump((1, 2, 3), f)
            self.assertEqual(opt.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump((1, 2, 3.5, 4, 5), f)
            self.assertEqual(opt.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump((1, 2, 3, 4, 5), f)
            self.assertEqual(opt.valid_restore_file(output_file), False)

            with open(output_file, 'wb') as f:
                pickle.dump((1, 2, 3, 4, True), f)
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
        x = 0.5
        y = f(x)

        for optim in all_local_optimizers():
            self.assertEqual(optim.evaluate(f, optim.transform(x)), y)

        self.assertRaises(ValueError, ScipyOptimizer, "strange_name")
        ScipyOptimizer.scipy_methods = ['method']
        opt = ScipyOptimizer('method')
        self.assertRaises(NotImplementedError, opt.get_addit_scipy_kwargs, '')

    def test_optimization_run(self):
        var1 = ContinuousVariable('var1', domain=[0, 1])
        var2 = ContinuousVariable('var2', domain=[1, 2])
        var3 = ContinuousVariable('var3', domain=[0, 20])
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
            res = opt.optimize(wrap, [var1, var2, var3], x0, args=(5,),
                               verbose=1, maxiter=maxit, maxeval=maxev,
                               eval_file=eval_file,
                               save_file=save_file, report_file=report_file)
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
            os.remove(eval_file)
            os.remove(save_file)
            os.remove(report_file)

    def get_yri_ceu_ll(self, x_dict, ns=[20,20]):
        def func(x_dict, ns, pts):
            # Define the grid we'll use
            xx = yy = dadi.Numerics.default_grid(pts)

            # phi for the equilibrium ancestral population
            phi = dadi.PhiManip.phi_1D(xx)
            # Now do the population growth event.
            phi = dadi.Integration.one_pop(phi, xx, x_dict['Tp'],
                                           nu=x_dict['nu1F'])

            # The divergence
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            # We need to define a function to describe the non-constant
            # population 2 size. lambda is a convenient way to do so.
            nu2_func = lambda t: x_dict['nu2B'] * (x_dict['nu2F'] / x_dict['nu2B'])**(t/x_dict['T'])  # NOQA
            phi = dadi.Integration.two_pops(phi, xx, x_dict['T'],
                                            nu1=x_dict['nu1F'], nu2=nu2_func,
                                            m12=x_dict['m'], m21=x_dict['m'])

            sfs = dadi.Spectrum.from_phi(phi, ns, (xx, yy))
            return sfs

        pts = [40, 50, 60]
        data = dadi.Spectrum.from_file(YRI_CEU_DATA)
        data = data.project(ns)
        func_ex = dadi.Numerics.make_extrap_log_func(func)

        model = func_ex(x_dict, data.sample_sizes, pts)
        ll_model = dadi.Inference.ll_multinom(model, data)
        return ll_model

    def test_yri_ceu_example(self):
        nu1F = PopulationSizeVariable('nu1F')
        nu2B = PopulationSizeVariable('nu2B')
        nu2F = PopulationSizeVariable('nu2F')
        m = MigrationVariable('m')
        Tp = TimeVariable('Tp')
        T = TimeVariable('T')
        Dyn = DynamicVariable('Dyn')

        dm = EpochDemographicModel()
        dm.add_epoch(Tp, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m], [m, None]], ['Sud', 'Exp'])
        dic = {'nu1F': 2.0, 'nu2B': 0.1, 'nu2F': 2, 'm': 1,
               'Tp':  0.2, 'T': 0.2}

        proj = (10, 10)
        data = SFSDataHolder(YRI_CEU_DATA, projections=proj)
        d = DadiEngine(model=dm, data=data)
        values = [dic[var.name] for var in dm.variables]

        d = get_engine('dadi')
        d.set_data(data)
        d.set_model(dm)

        args = ([40, 50, 60],)
        def f(x, *args):
                y = - d.evaluate(list(x), *args)
                #print(x, y)
                return y

        for opt in all_global_optimizers():
            with self.subTest(optimizer=opt.id):
                res = opt.optimize(f, dm.variables, num_init=10,
                                   args=args, maxeval=25, maxiter=5)
        for opt in all_local_optimizers():
            with self.subTest(local_optimizer=opt.id):
                res = opt.optimize(f, dm.variables, x0=values,
                                   args=args, maxiter=2)
                self.assertEqual(res.y, f(res.x, *args))
                self.assertEqual(res.y, -self.get_yri_ceu_ll(
                    {var.name: val for var, val in zip(dm.variables, res.x)},
                    proj))
                self.assertTrue(res.y <= f(values, *args))

    def run_example(self, engine_id, example_func, will_collapse=False):
        args = ()
        if engine_id == 'dadi':
            args = ([40,50,60],)
        f, variables = example_func(engine_id, args)
        x0 = [var.resample() for var in variables]

        def callback(x, y):
            pass

#        save_file = 'save_file'
        report_file = 'report_file'
        eval_file = 'eval_file'

        for opt in all_local_optimizers():
            with self.subTest(local_optimizer=opt.id):
#                print(opt.id)
                msg = f"(optimization {opt.id}, engine {engine_id})"
                if will_collapse and opt.id != 'None' and opt.id != None:
                    self.assertRaises(AssertionError, opt.optimize, f,
                                      variables, x0=x0, args=args,
                                      maxeval=10, maxiter=2,
                                      callback=callback,
                                      report_file=report_file,
#                                      save_file=save_file,
                                      eval_file=eval_file)
                else:
                    res = opt.optimize(f, variables, x0=x0,
                                       args=args, maxiter=2, maxeval=10,
                                       callback=callback,
                                       report_file=report_file,
#                                       save_file=save_file,
                                       eval_file=eval_file)
                    self.assertEqual(res.y, f(res.x, *args), msg=msg)
                    self.assertTrue(res.y <= f(x0, *args),
                                    msg=msg + f" {res.y} > {f(x0, *args)}")

    def test_1pop_example_1(self):
        for engine in all_engines():
            self.run_example(engine.id, get_1pop_sim_example_1)

    def test_1pop_example_2(self):
        for engine in all_engines():
            self.run_example(engine.id, get_1pop_sim_example_2,
                             will_collapse=True)

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
    def test_core_run(self):
        settings = test_args()
        settings.input_file = os.path.join(DATA_PATH, 'small_1pop.fs')
        settings.draw_models_every_n_iteration = 100
        settings.print_models_code_every_n_iteration = 100
        settings.verbose = 10
        shared_dict = gadma.shared_dict.SharedDictForCoreRun(
            multiprocessing=False)
        gadma.core.job(0, shared_dict, settings)

        settings.custom_filename = os.path.join(DATA_PATH,
                                                "small_1pop_dem_model_dadi.py")
        settings.directory_with_bootstrap = os.path.join(
            DATA_PATH, 'small_1_pop_bootstrap')
        settings.read_bootstrap_data()
        settings.linked_snp_s = True
        settings.relative_parameters = True
        shared_dict = gadma.shared_dict.SharedDictForCoreRun(
            multiprocessing=False)
        gadma.core.job(0, shared_dict, settings)

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
        self.assertRaises(NotImplementedError, opt.optimize, f, variables, 10)
