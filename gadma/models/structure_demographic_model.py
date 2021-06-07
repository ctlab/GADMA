from ..utils import Variable, PopulationSizeVariable, TimeVariable
from ..utils import MigrationVariable, DynamicVariable, SelectionVariable,\
                    FractionVariable, ContinuousVariable
from . import Epoch, Split
from .variables_combinations import Multiplication, Subtraction
from .demographic_model import EpochDemographicModel
import copy
import numpy as np
import warnings


class StructureDemographicModel(EpochDemographicModel):
    r"""
    Special class for demographic model created by structure.

    :param initial_structure: List of ints with number of intervals
                              in initial structure.
    :type initial_structure: list of ints
    :param final_structure: List of ints with number of intervals
                            in final structure.
    :type final_structure: list of ints
    :param has_migs: If True then model will have migrations.
    :type has_migs: bool
    :param has_sels: If True then model will have selection coefficients.
    :type has_sels: bool
    :param has_inbr: If True then model will have inbreeding.
    :type has_inbr: bool
    :param has_dyns: If True then model will create dynamics of size
                      change different to Sudden change.
    :type has_dyns: bool
    :param sym_migs: If True then migrations will be symetric.
    :type sym_migs: bool
    :param frac_split: If True then populations split in some proportion. If
                       False then newly formed population has size as
                       an independent variable.
    :type frac_split: bool
    :param migs_mask: List of matrices of 0 and 1 for each time interval (after
                      first split) that defines what migrations this interval
                      has. E.g. [[[0, 1],[0, 0]], [[0, 0], [0, 0]]] for
                      structure (\*, 2) will allow migration from pop2 to pop1
                      in the interval right after split. Note that structure
                      should be fixed if migs_mask is set
                      (:meth:`increase_structure` raises ValueError).
    :type migs_mask: list
    :param has_anc_size: If True then Nanc_size variable is created.
    :type has_anc_size: bool
    :param gen_time: Time in years of one generation.
    :type gen_time: float
    :param theta0: Mutation flux (4\*mu\*L).
    :type theta0: float
    :param mu: Mutation rate per base per generation.
    :type mu: float
    """
    def __init__(self, initial_structure, final_structure,
                 has_migs, has_sels, has_dyns, sym_migs, frac_split,
                 migs_mask=None, has_anc_size=False,
                 gen_time=None, theta0=None, mu=None, has_inbr=None):
        if has_anc_size:
            Nanc_size = PopulationSizeVariable("Nanc", units="physical")
        else:
            Nanc_size = None
        super(StructureDemographicModel, self).__init__(
            gen_time=gen_time,
            theta0=theta0,
            mu=mu,
            has_anc_size=has_anc_size,
            Nanc_size=Nanc_size,
        )
        if np.any(np.array(initial_structure) > np.array(final_structure)):
            raise ValueError(f"Elements of the initial structure "
                             f"({initial_structure}) "
                             f"could not be greater than elements of the "
                             f"final structure ({final_structure}).")
        self.initial_structure = np.array(initial_structure)
        self.final_structure = np.array(final_structure)
        self.has_migs = has_migs
        self.has_sels = has_sels
        self.has_dyns = has_dyns
        self.sym_migs = sym_migs
        self.frac_split = frac_split
        self.migs_mask = migs_mask
        self.has_inbr = has_inbr
        # check that mask is correct
        if len(self.initial_structure) == 1 and self.migs_mask is not None:
            warnings.warn("Migration mask is used only when more than one "
                          "population is observed")
            self.migs_mask = None
        if not self.has_migs:
            if self.sym_migs:
                warnings.warn("There is no migrations and option about "
                              "symmetric migrations will be ignored")
                self.sym_migs = False
            if self.migs_mask is not None:
                warnings.warn("There is no migrations and option about "
                              "migrations masks will be ignored.")
                self.migs_mask = None
        if self.migs_mask is not None:
            if not np.all(self.initial_structure == self.final_structure):
                raise ValueError("Migration masks should be used for models "
                                 f"with equal initial ({initial_structure}) "
                                 f"and final ({final_structure}) structures.")

            if len(self.migs_mask) != sum(self.initial_structure[1:]):
                raise ValueError(f"Number of masks in migrations mask "
                                 f"({len(self.migs_mask)}) should be equal to "
                                 "the number of time intervals after first "
                                 f"split ({sum(self.initial_structure[1:])}).")
            for i, mask in enumerate(self.migs_mask):
                # TODO work only for 3 populations
                if i < self.initial_structure[1]:
                    npop = 2
                else:
                    npop = 3
                mask = np.array(mask)
                if mask.shape != (npop, npop):
                    raise ValueError(f"Mask number {i} should have size equal "
                                     f"to {(npop, npop)} but it has "
                                     f"{mask.shape}")
                self.migs_mask[i] = np.array(mask)
        if self.sym_migs and self.migs_mask is not None:
            for i, mask in enumerate(self.migs_mask):
                if not np.allclose(mask, mask.T):
                    raise ValueError(f"For symmetric migrations masks should "
                                     f"be symmetric. Mask number {i}:\n{mask}")
        self.from_structure(self.initial_structure)

    def from_structure(self, structure):
        """
        Creates new model from given structure.
        It is base constructor of the model.

        :param structure: Structure of the model.
        :type structure: list of ints
        """
        super(StructureDemographicModel, self).__init__(
            gen_time=self.gen_time,
            theta0=self.theta0,
            mu=self.mu,
            has_anc_size=self.has_anc_size,
            Nanc_size=self.Nanc_size
        )
        if not (np.all(np.array(structure) >= self.initial_structure)):
            raise ValueError(f"Elements of model structure ({structure}) "
                             f"could not be smaller than elements of the "
                             f"initial structure ({self.initial_structure}).")

        if not (np.all(np.array(structure) <= self.final_structure)):
            raise ValueError(f"Elements of model structure ({structure}) "
                             f"could not be greater than elements of the "
                             f"final structure ({self.final_structure}).")
        if np.any(np.array(structure) <= 0):
            raise ValueError(f"Elements of model structure ({structure}) "
                             "should be positive (> 0).")

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
                if self.has_migs and n_pop > 1:
                    mig_vars = np.zeros(shape=(n_pop, n_pop), dtype=object)
                    mask = None
                    if self.migs_mask is not None:
                        mask = self.migs_mask[i_int - structure[0]]
                        assert mask.shape == mig_vars.shape
                    for i in range(n_pop):
                        for j in range(n_pop):
                            if i == j:
                                continue
                            if self.sym_migs and j < i:
                                continue
                            var = MigrationVariable('m%d_%d%d' %
                                                    (i_int, i+1, j+1))
                            if mask is not None and mask[i][j] == 0:
                                var = 0
                            mig_vars[i][j] = var
                            if self.sym_migs:
                                mig_vars[j][i] = var
                sel_vars = None
                if self.has_sels:
                    sel_vars = list()
                    for i in range(n_pop):
                        var = SelectionVariable('g%d%d' % (i_int, i+1))
                        sel_vars.append(var)
                dyn_vars = None
                if self.has_dyns:
                    dyn_vars = list()
                    for i in range(n_pop):
                        var = DynamicVariable('dyn%d%d' % (i_int, i+1))
                        dyn_vars.append(var)
                assert not self.has_inbreeding
                self.add_epoch(time_arg=time_var, size_args=size_vars,
                               mig_args=mig_vars, dyn_args=dyn_vars,
                               sel_args=sel_vars)
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
                assert not self.has_inbreeding
                self.add_split(n_pop - 1, size_vars)

        if self.has_inbr:
            inbr_args = list()
            for n_pop in range(1, len(structure) + 1):
                var = FractionVariable('F%d' % n_pop)
                inbr_args.append(var)
            self.add_inbreeding(inbr_args)

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
        if self.migs_mask is not None:
            raise ValueError("Structure of the model could not be increased: "
                             "there is a mask on migrations.")
        cur_structure = self.get_structure()
        diff = np.array(self.final_structure) - np.array(cur_structure)
        if np.all(diff <= 0):
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
                if var not in new_event.get_vars_not_in_init_args():
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
                new_X.append([])
                # if we have anc size then we should put after first variable
                if self.has_anc_size:
                    new_X[-1].extend([x[0]])
                    x = x[1:]
                new_X[-1].extend(copy.copy(new_values))
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
                assert type(old_var) == type(new_var)  # addit. check for types
                oldvar2newvar[old_var] = new_var
        if self.has_anc_size:
            assert self.has_anc_size
            oldvar2newvar[old_model.Nanc_size] = self.Nanc_size
        if old_model.has_inbreeding:
            assert self.has_inbreeding
            for old_inbr_arg, new_inbr_arg in zip(old_model.inbreeding_args,
                                                  self.inbreeding_args):
                oldvar2newvar[old_inbr_arg] = new_inbr_arg
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
            event1 = self.events[event_index]  # our new event
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

    def transform_values_from_other_model(self, model, x):
        assert isinstance(model, StructureDemographicModel)
        assert np.all(self.get_structure() == model.get_structure())

#        print('other', model.variables)
#        print('self', self.variables)
#        print('x', x)
        other_var2value = model.var2value(x)
        other_varname2value = {var.name: other_var2value[var]
                               for var in other_var2value}
        var2value = {}
        # First fill with common variables
        for var in self.variables:
            if isinstance(var, MigrationVariable):
                if self.has_migs == model.has_migs:
                    if self.sym_migs == model.sym_migs:
                        # when we have some new masks name could be missed
                        if var.name in other_varname2value:
                            var2value[var] = other_varname2value[var.name]
                        else:
                            var2value[var] = 0
            elif isinstance(var, SelectionVariable):
                if self.has_sels == model.has_sels:
                    var2value[var] = other_varname2value[var.name]
            elif isinstance(var, DynamicVariable):
                if self.has_dyns == model.has_dyns:
                    var2value[var] = other_varname2value[var.name]
            elif isinstance(var, FractionVariable):
                if self.frac_split == model.frac_split \
                        and var.name.startswith('s'):
                    var2value[var] = other_varname2value[var.name]
                if self.has_inbr == model.has_inbr \
                        and var.name.startswith('F'):
                    var2value[var] = other_varname2value[var.name]
            elif var.name in other_varname2value:
                var2value[var] = other_varname2value[var.name]

        # Transform other values
        varname2value = {}
        for var in self.variables:
            if var.name in varname2value:
                var2value[var] = varname2value[var.name]
            if var in var2value:
                continue
            if isinstance(var, MigrationVariable):
                if not model.has_migs:
                    var2value[var] = 0
                elif self.sym_migs != model.sym_migs:
                    ij = var.name.split('_')[-1]
                    assert len(ij) == 2
                    sym_mig_name = var.name[:-2] + ij[::-1]
                    mij_value = other_varname2value.get(var.name, 0)
                    mji_value = other_varname2value.get(sym_mig_name, 0)
                    if self.sym_migs and not model.sym_migs:
                        var2value[var] = (mij_value + mji_value) / 2
                    else:
                        var2value[var] = mij_value
                        varname2value[sym_mig_name] = mji_value
            elif isinstance(var, SelectionVariable):
                assert self.has_sels and not model.has_sels
                var2value[var] = 0
            elif isinstance(var, DynamicVariable):
                assert self.has_dyns and not model.has_dyns
                var2value[var] = 'Sud'
            elif isinstance(var, FractionVariable) \
                    and var.name.startswith('s'):
                assert self.frac_split and not model.frac_split
                n_split = int(var.name[1:])
                ind_before_split = sum(self.get_structure()[:n_split]) - 1
                nu1_before_split_name = "nu%d%d" % (ind_before_split, n_split)
                nu1_after_split_name = "nu%d%d_1" % (ind_before_split,
                                                     n_split)
                if n_split == 1 and self.get_structure()[0] == 1:
                    size_before_split_name = 1.0
                    size_after_split_time = other_varname2value["nu_1"]
                else:
                    size_before_split_name = other_varname2value[
                        nu1_after_split_name]
                    size_after_split_time = other_varname2value[
                        nu1_before_split_name]
                fraction = size_after_split_time / size_before_split_name
                fraction = max(fraction, var.domain[0])
                fraction = min(fraction, var.domain[1])
                var2value[var] = fraction
            elif isinstance(var, FractionVariable) \
                    and var.name.startswith('F'):
                assert self.has_inbr and not model.has_inbr
                var2value[var] = 0
            elif isinstance(var, PopulationSizeVariable):
                assert not self.frac_split and model.frac_split
                assert len(var.name.split("_")) > 1

                nu_before_split_name = var.name[:-2]  # remove last _1 or _2

                # if there was ancestral population split there is no variable
                if (nu_before_split_name == "nu" and
                        nu_before_split_name not in varname2value):
                    pop_size = 1.0
                    frac_name = "s1"
                else:
                    pop_size = other_varname2value[nu_before_split_name]
                    frac_name = "s" + nu_before_split_name[-1]
                fraction = other_varname2value[frac_name]

                if var.name[-1] == "1":
                    var2value[var] = fraction * pop_size
                elif var.name[-1] == "2":
                    var2value[var] = (1 - fraction) * pop_size
            else:
                raise ValueError("Some changes in demographic models are not "
                                 "allowed or implemented. Got new variable "
                                 f"({var}) that cannot be processed.")
        x_final = [var2value[var] for var in self.variables]
        for i in range(len(self.variables)):
            var = self.variables[i]
            if isinstance(var, ContinuousVariable):
                x_final[i] = min(max(x_final[i], var.domain[0]), var.domain[1])
        return x_final
