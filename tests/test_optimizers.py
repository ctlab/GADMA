import unittest

from .test_data import YRI_CEU_DATA
from gadma import *
import dadi


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


class TestLocalOpt(TestBaseOptClass):
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

    def f(self, x, y):
        return np.sum(x ** 4 + 2 * x ** 3 - 12 * x ** 2 - 2 * x + 6)

    def test_optimization_run(self):
        var1 = ContinuousVariable('var1', domain=[0, 1])
        var2 = ContinuousVariable('var2', domain=[1, 2])
        var3 = ContinuousVariable('var3', domain=[0, 20])
        x0 = [0.5, 1.5, 10]
        for opt in all_local_optimizers():
            opt.optimize(self.f, [var1, var2, var3], x0, args=(5,), verbose=0, maxiter=1)

    def get_yri_ceu_ll(self, x_dict):
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
        func_ex = dadi.Numerics.make_extrap_log_func(func)

        model = func_ex(x_dict, data.sample_sizes, pts)
        ll_model = dadi.Inference.ll_multinom(model, data)
        return ll_model

    def _test_yri_ceu_example(self):
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

        data = SFSDataHolder(YRI_CEU_DATA)
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
                                   args=args, maxeval=25)
        for opt in all_local_optimizers():
            with self.subTest(local_optimizer=opt.id):
                res = opt.optimize(f, dm.variables, x0=values,
                                   args=args, maxiter=2)
                self.assertEqual(res.y, f(res.x, *args))
                self.assertEqual(res.y, -self.get_yri_ceu_ll(
                    {var.name: val for var, val in zip(dm.variables, res.x)}))
                self.assertTrue(res.y <= f(values, *args))

    def run_example(self, engine_id, example_func):
        args = ()
        if engine_id == 'dadi':
            args = ([40,50,60],)
        f, variables = example_func(engine_id, args)
        x0 = [var.resample() for var in variables]

        for opt in all_local_optimizers():
            with self.subTest(local_optimizer=opt.id):
                #print(opt.maximize)
                msg = f"(optimization {opt.id}, engine {engine_id})"
                res = opt.optimize(f, variables, x0=x0,
                                   args=args, maxiter=2)
                self.assertEqual(res.y, f(res.x, *args), msg=msg)
                self.assertTrue(res.y <= f(x0, *args), msg=msg + f" {res.y} > {f(x0, *args)}")

    def test_1pop_example_1(self):
        for engine in all_engines():
            self.run_example(engine.id, get_1pop_sim_example_1)

    def _test_1pop_example_2(self):
        for engine in all_engines():
            with self.assertRaises(AssertionError):
                self.run_example(engine.id, get_1pop_sim_example_2)

    def _test_2pop_example_1(self):
        for engine in all_engines():
            self.run_example(engine.id, get_2pop_sim_example_1)

