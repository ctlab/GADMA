

class Target(object):
    def __init__(self, dem_model, data, engine):
        self.dem_model = dem_model
        self.data = data
        self.engine = engine
        self.variables = self.dem_model.variables()
        self.domain = [x.get_domain() for x in self.variables()]

    def objective_func(self, x):
        var_value_dict = {var: value for var, value in zip(self.variables, x)} 
        return self.engine.evaluate(self.dem_model, self.data, var_value_dict)


from demographic_model import DemographicModel

dm = DemographicModel()
nu1F = PopulationSizeVariable('nu1F')
nu2B = PopulationSizeVariable('nu2B')
nu2F = PopulationSizeVariable('nu2F')
m = MigrationVariable('m')
Tp = TimeVariable('Tp')
T = TimeVariable('T')
Dyn = DynamicVariable('Dyn')

dm.add_epoch(Tp, [nu1F])
dm.add_split(0, [nu1F, nu2B])
dm.add_epoch(T, [nu1F, nu2F], [[None, 0],[0, None]], ['Sud', Dyn])

print(dm.variables)

dic = {'nu1F': 1.880, 'nu2B': 0.0724, 'nu2F': 1.764, 'm': 0.930, 'Tp':  0.363, 'T': 0.112, 'Dyn': 'Exp'}

data = SFSDataHolder("../../dadi/examples/YRI_CEU/YRI_CEU.fs")

d = DadiEngine(dm, [40,50,60])

data.prepare_for_engine(d)

print(d.evaluate(dic, data))
