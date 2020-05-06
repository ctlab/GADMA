import unittest

from .test_data import YRI_CEU_DATA
from gadma import *
import dadi

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

        self.assertEqual(opt1.evaluate(f, x), y)
        self.assertEqual(opt2.evaluate(f, np.log(x)), y)
        self.assertEqual(opt3.evaluate(f, x), -y)
        self.assertEqual(opt4.evaluate(f, np.log(x)), -y)

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
        return np.sum(x**4 + 2*x**3 -12*x**2 - 2*x + 6)

    def test_optimization_run(self):
        var1 = ContinuousVariable('var1', domain=[0,1])
        var2 = ContinuousVariable('var2', domain=[1,2])
        var3 = ContinuousVariable('var3', domain=[0,20])
        x0 = [0.5, 1.5, 10]
        for opt in all_local_optimizers():
            opt.optimize(self.f, [var1, var2, var3], x0, args=(5,), maxiter=1)

    def get_yri_ceu_ll(self, x_dict):
        def func(x_dict, ns, pts):
            # Define the grid we'll use
            xx = yy = dadi.Numerics.default_grid(pts)

            # phi for the equilibrium ancestral population
            phi = dadi.PhiManip.phi_1D(xx)
            # Now do the population growth event.
            phi = dadi.Integration.one_pop(phi, xx, x_dict['Tp'], nu=x_dict['nu1F'])

            # The divergence
            phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
            # We need to define a function to describe the non-constant population 2
            # size. lambda is a convenient way to do so.
            nu2_func = lambda t: x_dict['nu2B']*(x_dict['nu2F']/x_dict['nu2B'])**(t/x_dict['T'])
            phi = dadi.Integration.two_pops(phi, xx, x_dict['T'], nu1=x_dict['nu1F'], nu2=nu2_func, 
                                            m12=x_dict['m'], m21=x_dict['m'])

            sfs = dadi.Spectrum.from_phi(phi, ns, (xx,yy))
            return sfs

        pts = [40,50,60]
        data = dadi.Spectrum.from_file(YRI_CEU_DATA)
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

        dm = DemographicModel()
        dm.add_epoch(Tp, [nu1F])
        dm.add_split(0, [nu1F, nu2B])
        dm.add_epoch(T, [nu1F, nu2F], [[None, m],[m, None]], ['Sud', 'Exp'])
        dic = {'nu1F': 2.0, 'nu2B': 0.1, 'nu2F': 2, 'm': 1,
               'Tp':  0.2, 'T': 0.2}

        data = SFSDataHolder(YRI_CEU_DATA)
        d = DadiEngine(model=dm, data=data)
        values = [dic[var.name] for var in dm.variables]

        d = get_engine('dadi')
        d.set_data(data)
        d.set_model(dm)

        for opt in all_local_optimizers():
            if d.id == 'dadi':
                args = ([40,50,60],)
            else:
                args = ()
            def f(x, *args):
                y = - d.evaluate(list(x), *args)
                print(x, y)
                return y
            res = opt.optimize(f, dm.variables, x0=values, args=args, maxiter=2)
            print(d.id, opt.id,  res, f(values, *args))
            self.assertEqual(res[1], f(res[0], *args))
            self.assertEqual(res[1], -self.get_yri_ceu_ll({var.name: val for var, val in zip(dm.variables, res[0])}))
            self.assertTrue(res[1] < f(values, *args))
