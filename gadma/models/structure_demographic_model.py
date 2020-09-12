from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import VariablePool
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable,\
                    FractionVariable
from . import Model, Epoch, Split
from .variables_combinations import Multiplication, Subtraction
from .demographic_model import EpochDemographicModel
from collections import OrderedDict
import copy
import numpy as np


class StructureDemographicModel(EpochDemographicModel):
    """
    Special class for demographic model created by structure.

    :param initial_structure: List of ints with number of intervals
                              in initial structure.
    :type initial_structure: list of ints
    :param final_structure: List of ints with number of intervals
                            in final structure.
    :type final_structure: list of ints
    :param have_migs: If True then model will have migrations.
    :type have_migs: bool
    :param have_sels: If True then model will have selection coefficients.
    :type have_sels: bool
    :param have_dyns: If True then model will create dynamics of size
                      change different to Sudden change.
    :type have_dyns: bool
    :param sym_migs: If True then migrations will be symetric.
    :type sym_migs: bool
    :param frac_split: If True then populations split in some proportion. If
                       False then newly formed population has size as
                       an independent variable.
    :type frac_split: bool
    """
    def __init__(self, initial_structure, final_structure,
                 have_migs, have_sels, have_dyns, sym_migs, frac_split,
                 gen_time=None, theta0=None, mu=None):
        super(StructureDemographicModel, self).__init__(gen_time, theta0, mu)
        if np.any(np.array(initial_structure) > np.array(final_structure)):
            raise ValueError(f"Elements of the initial structure "
                             f"({initial_structure}) "
                             f"could not be greater than elements of the "
                             f"final structure ({final_structure}).")
        self.initial_structure = np.array(initial_structure)
        self.final_structure = np.array(final_structure)
        self.have_migs = have_migs
        self.have_sels = have_sels
        self.have_dyns = have_dyns
        self.sym_migs = sym_migs
        self.frac_split = frac_split
        self.from_structure(self.initial_structure)

    def from_structure(self, structure):
        """
        Creates new model from given structure.
        It is base constructor of the model.

        :param structure: Structure of the model.
        :type structure: list of ints
        """
        super(StructureDemographicModel, self).__init__(self.gen_time,
                                                        self.theta0, self.mu)
        if not (np.all(np.array(structure) >= self.initial_structure)):
            raise ValueError(f"Elements of model structure ({structure}) "
                             f"could not be smaller than elements of the "
                             f"initial structure ({self.initial_structure}).")

        if not (np.all(np.array(structure) <= self.final_structure)):
            raise ValueError(f"Elements of model structure ({structure}) "
                             f"could not be greater than elements of the final "
                             f"structure ({self.final_structure}).")
        if np.any(np.array(structure) <= 0):
            raise ValueError(f"Elements of model structure ({structure}) "
                             "should be positive (> 0).")

        if list(structure) == [1]:
            return self
        i_int = 0
        if self.frac_split:
            size_vars = [1.0]
        else:
            size_vars = [PopulationSizeVariable('nu')]
        for n_pop in range(1, len(structure) + 1):
            n_int = structure[n_pop - 1]
            if n_pop == 1:
                n_int -= 1
            for _i in range(1, n_int + 1):
                i_int += 1
                time_var = TimeVariable('t%d' % (i_int))
                size_vars = list()
                for i_pop in range(n_pop):
                    var = PopulationSizeVariable('nu%d%d' % (i_int, i_pop+1))
                    size_vars.append(var)
                mig_vars = None
                if self.have_migs and n_pop > 1:
                    mig_vars = np.zeros(shape=(n_pop, n_pop), dtype=object)
                    for i in range(n_pop):
                        for j in range(n_pop):
                            if i == j:
                                continue
                            if self.sym_migs and j < i:
                                continue
                            var = MigrationVariable('m%d_%d%d' % 
                                                    (i_int, i+1, j+1))
                            mig_vars[i][j] = var
                            if self.sym_migs:
                                mig_vars[j][i] = var
                sel_vars = None
                if self.have_sels:
                    sel_vars = list()
                    for i in range(n_pop):
                        var = SelectionVariable('g%d%d' % (i_int, i+1))
                        sel_vars.append(var)
                dyn_vars = None
                if self.have_dyns:
                    dyn_vars = list()
                    for i in range(n_pop):
                        var = DynamicVariable('dyn%d%d' % (i_int, i+1))
                        dyn_vars.append(var)
                self.add_epoch(time_var, size_vars,
                             mig_vars, dyn_vars, sel_vars)
            if n_pop < len(structure):
                if self.frac_split:
                    frac_var = FractionVariable(f"s{n_pop}")
                    size_split = copy.copy(size_vars[-1])
                    size_1 = Multiplication(frac_var, size_split)
                    size_2 = Multiplication(Subtraction(1, frac_var),
                                            size_split)
                    size_vars = [size_1, size_2]
                else:
                    name = size_vars[-1].name
                    size_vars = [PopulationSizeVariable(name + '_1'),
                                 PopulationSizeVariable(name + '_2')]
                self.add_split(n_pop - 1, size_vars)
        return self

    def get_structure(self):
        """
        Returns current structure of the model.
        """
        structure = [1]
        for event in self.events:
            if isinstance(event, Split):
                structure.append(0)
            elif isinstance(event, Epoch):
                structure[-1] += 1
            else:
                raise ValueError("Event is not Split or Epoch.")
        return structure

    def increase_structure(self, new_structure=None, X=None):
        """
        Increase structure of the model. Raises ValueError if structure is
        equal or greater than `final_structure`. 

        :param new_structure: New structure for the model. Should be greater
                              by 1 in one element. E.g. structure is (1,2),
                              then new structure could be either (2,2) or
                              (1,3). If None random available element is
                              chosen to increase.
        :param X: list of values to transform as values of new model.

        :note: Function is specific for\
               :class:`gadma.models.StructureDemographicModel`.\
               It probably will not work for any other class.
        """
        cur_structure = self.get_structure()
        diff = np.array(self.final_structure) - np.array(cur_structure)
        if np.any(diff < 0):
            raise ValueError(f"Demographic model has its final structure "
                             f"({list(self.final_structure)}). It is not "
                             f"possible to increase it")
        # TODO check that new structure is not greater than final structure
        if new_structure is None:
            struct_index = np.random.choice(
                np.arange(len(cur_structure))[diff != 0])

            new_structure = copy.copy(cur_structure)
            new_structure[struct_index] += 1
        else:
            diff = np.array(new_structure) - np.array(cur_structure)
            if np.max(diff) < 1:
                raise ValueError(f"New structure ({new_structure}) should be "
                                 f"greater than the current one "
                                 f"({cur_structure})")
            if np.min(diff) > 1 or np.sum(diff) > 1:
                raise ValueError(f"New structure ({new_structure}) should "
                                 f"differ in one value and maximum by 1. "
                                 f"Received structure: {new_structure}")
            struct_index = np.where(diff == 1)[0]
            assert len(struct_index) == 1
            struct_index = struct_index[0]

        event_index = np.random.choice(np.arange(cur_structure[struct_index]))
        event_index += sum(cur_structure[:struct_index]) - 1 + struct_index

        old_model = copy.deepcopy(self)
        self.from_structure(new_structure)
        if X is None:
            return self, None

        # We consider that we have put new event (interval) before the chosen
        # event. Special case is when it is first interval - we put new event
        # after it.
        if event_index == -1:
            new_event = self.events[0]
            new_values = []
            for var in new_event.variables:
                if var in new_event.init_size_args:
                    continue
                # We put as time some random value
                if isinstance(var, TimeVariable):
                    new_values.append(new_event.time_arg.resample())
                # We put size of population as 1.0
                elif isinstance(var, PopulationSizeVariable):
                    new_values.append(1.0)
                # Dynamic as Sud
                elif isinstance(var, DynamicVariable):
                    new_values.append("Sud")
                elif isinstance(var, SelectionVariable):
                    new_values.append(0)
                else:
                    raise ValueError(f"Unknown type of variable: "
                                     f"{var.__class__}")
            new_X = []
            for x in X:
                new_X.append(copy.copy(new_values))
                new_X[-1].extend(x)
            return self, new_X

        # So we build dict with variables correspondence - they will be moved
        # forward by one event starting with chosen event.
        oldvar2newvar = {}
        for i, (old_event, new_event) in enumerate(zip(old_model.events,
                                                       self.events)):
            if i >= event_index:
                break
            for old_var, new_var in zip(old_event.variables,
                                        new_event.variables):
                oldvar2newvar[old_var] = new_var
        for old_event, new_event in zip(old_model.events[event_index:],
                                        self.events[event_index + 1:]):
            # Remove init_sizes variables from list of vriables.
            # Then both lists will have the same length and have special order
            if isinstance(old_event, Split):
                old_vars = [var for var in old_event.variables]
                new_vars = [var for var in new_event.variables]
            else:
                old_vars = old_event.get_vars_not_in_init_args()
                new_vars = new_event.get_vars_not_in_init_args()
            assert len(old_vars) == len(new_vars)
            # Now we cretae correspondence between those variables        
            for old_var, new_var in zip(old_vars, new_vars):
                assert type(old_var) == type(new_var)  #addit. check for types
                oldvar2newvar[old_var] = new_var
    #    print(oldvar2newvar)
        new_X = []
        for x in X:
            # Our initial var2value
            var2value = old_model.var2value(x)
            # Now we get new var2value. We should be carefull as variables
            # in new_model have the same names but they are different from
            # those in model.
            varname2value = {var.name: var2value[var] for var in var2value}
            new_var2value = {var: varname2value[var.name]
                             for var in self.variables
                             if var.name in varname2value}
            for var in var2value:
                new_var2value[oldvar2newvar[var]] = var2value[var]
            event1 = self.events[event_index] # our new event
            event2 = self.events[event_index + 1]  # base event
            # Time / 2
            new_var2value[event2.time_arg] /= 2
            new_var2value[event1.time_arg] = new_var2value[event2.time_arg]
            time_value = new_var2value[event1.time_arg]

            # Sizes
            for i, (size_in, size_out) in enumerate(zip(event1.init_size_args,
                                                   event1.size_args)):
                if event2.dyn_args is not None:
                    dyn_value = new_var2value[event2.dyn_args[i]]
                else:
                    dyn_value = 'Sud'
                if dyn_value != 'Sud':
#                    new_var2value[size_out] = new_var2value[size_out]
#                else:
                    func = DynamicVariable.get_func_from_value(dyn_value)
                    if event_index == 0:
                        y1 = 1.0
                    else:
                        # Init size could be with fraction.
                        if not isinstance(size_in, Variable):
                            vals = [new_var2value[var] 
                                    for var in size_in.variables]
                            y1 = size_in.get_value(vals)
                        else:
                            y1 = new_var2value[size_in]
                    y2 = new_var2value[size_out]
                    # We have already divided it.
                    x_diff = 2 * time_value
                    size_func = func(y1, y2, x_diff)
                    new_var2value[size_out] = size_func(x_diff / 2)
            # Copy other variables
            for var1, var2 in zip(event1.variables, event2.variables):
                if var1 not in new_var2value:
                    new_var2value[var1] = new_var2value[var2]
            new_X.append([new_var2value[var] for var in self.variables])

        return self, new_X
