import io
import os
import re
import shutil
import tempfile
from io import StringIO
from math import log
from pathlib import Path
from typing import Final, Union, List, Dict, NoReturn, Tuple, Callable, Optional, ClassVar, NamedTuple, Sequence, \
    Mapping

import numpy as np
from dadi import Spectrum
from numpy import ndarray

from . import Engine, register_engine
from ..code_generator.fsc2_generator import EstimationFile, TemplateFile, DefinitionFile
from ..data import SFSDataHolder
from ..models import CustomDemographicModel, EpochDemographicModel, TreeDemographicModel, Epoch, \
    Split, \
    PopulationSizeChange, Leaf, LineageMovement
from ..models.variables_combinations import Addition, Division
from ..utils.variables import PopulationSizeVariable, TimeVariable, DynamicVariable

LINE = ("\n" + "-" * 80 + "\n")

EpochModelEvent = Union[Epoch, Split]
TreeModelEvent = Union[Leaf, LineageMovement, PopulationSizeChange]
Model = Union[EpochDemographicModel, TreeDemographicModel]
ParameterValues = Union[List, Dict]

FSC2_PATH: Final[Optional[str]] = os.getenv('FSC2_PATH')


def fsc2_available() -> bool:
    if not FSC2_PATH:
        return False
    return True


def is_multi_sfs(data_holder: SFSDataHolder):
    filename = Path(data_holder.filename).name
    if 'DSFS' in filename or 'MSFS' in filename:
        return True
    else:
        return False


def _find_matching_event(tree_event: TreeModelEvent,
                         epoch_events: List[EpochModelEvent]) -> Optional[EpochModelEvent]:
    """
    Find all events in an ``EpochDemographicModel``
    that match to the event from a ``TreeDemographicModel``

    :param tree_event: A ``TreeDemographicModel`` event that is used for comparison
    :param epoch_events: A list of ``EpochDemographicModel`` events that should be
        searched for matching events
    """
    # TODO: Make a fix for this edge case.
    #  TreeDemographicModel doesn't have an event for migration matrix switch.
    #  Therefore, this algorithm will not find a matching Epoch/Split,
    #  if there's an Epoch that has migration matrix switch as its only change
    #  compared to the Epoch after it.
    #  Which means that this change will not be recorded in the estimation file.
    time: TimeVariable = tree_event.t.variables[-1]

    def next_epoch_time(split: Split) -> Union[TimeVariable, int]:
        event_index = epoch_events.index(split)
        try:
            event_after_index = event_index + 1
            if not isinstance(epoch_events[event_after_index], Epoch):
                event_after_index += 1
        except IndexError:
            return 0
        else:
            return epoch_events[event_after_index].time_arg

    def pop_size_change_condition(epoch: Epoch) -> bool:
        return (isinstance(tree_event, PopulationSizeChange)
                and isinstance(epoch, Epoch)
                and epoch.time_arg == time)

    def lineage_movement_condition(split: Split) -> bool:
        return (isinstance(tree_event, LineageMovement)
                and isinstance(split, Split)
                and next_epoch_time(split) == time
                and split.pop_to_div == tree_event.pop_from)

    if isinstance(tree_event, Leaf):
        return None
    for event in epoch_events:
        if (pop_size_change_condition(event)
                or lineage_movement_condition(event)):
            return event


def _map_model_events(model_tree_events: List[TreeModelEvent],
                      model_epoch_events: List[EpochModelEvent]) -> Dict[TreeModelEvent,
                                                                         EpochModelEvent]:
    """
    Map events of an ``TreeDemographicModel`` model
    to its ``EpochDemographicModel`` counterpart

    :param model_tree_events: Its tree-based counterpart events
    :param model_epoch_events: Epoch-based demographic model events

    :return: Dictionary with ``PopulationSizeChange`` and ``LineageMovement`` events
        mapped to ``Epoch`` and ``Split`` events, accordingly.
    """
    event_mapping = {}
    for tree_event in model_tree_events:
        if isinstance(tree_event, Leaf):
            continue
        matching_event = _find_matching_event(tree_event, model_epoch_events)
        event_mapping[tree_event] = matching_event
    return event_mapping


def _get_fsc2_model(model, values: ParameterValues) -> Tuple[Model, ParameterValues]:
    from gadma import DemographicModel
    assert isinstance(model, DemographicModel)
    fsc2_model = model
    if isinstance(model, EpochDemographicModel):
        fsc2_model, values = model.translate_to(TreeDemographicModel, values)
    return fsc2_model, values


def _get_initial_pop_sizes(model: TreeDemographicModel) -> Tuple:
    """
    Get population sizes at current time.
    :returns: Tuple of variables representing initial population sizes.
    """
    assert isinstance(model, TreeDemographicModel)
    leaves = _leaves(model)
    init_pop_sizes = []
    for leaf in leaves:
        init_pop_sizes.append(leaf.size_pop.name)
    return tuple(init_pop_sizes)


def _get_growth_rate_for_population(model, values: ParameterValues,
                                    pop: int,
                                    epoch: Epoch) -> float:
    # TODO: Rewrite with TreeModelEvent in mind
    growth_rates = _get_growth_rate(model, values, epoch)
    return growth_rates[pop]


def _get_growth_rate(model, values: ParameterValues,
                     event: TreeModelEvent) -> float:
    # TODO: Rewrite with TreeModelEvent in mind
    var2value: Callable = model.get_value_from_var2value

    init_pop_sizes_vars: List[PopulationSizeVariable] = event.init_size_args
    final_pop_sizes_vars: List[PopulationSizeVariable] = event.size_args
    time_var: TimeVariable = event.time_arg

    init_pop_sizes: Tuple = tuple(var2value(values, v) for v in init_pop_sizes_vars)
    final_pop_sizes: Tuple = tuple(var2value(values, v) for v in final_pop_sizes_vars)
    time: int = var2value(values, time_var)

    growth_rates = []
    for size_0, size_t in zip(final_pop_sizes, init_pop_sizes):
        # Growth is exponential: N_t=N_0e^{rt}, where `r` is the growth rate.
        # Therefore, r=\frac{\ln \left(\frac{N_t}{N_0}\right)}{t}
        growth_rate = (log(size_t / size_0)) / time
        growth_rates.append(growth_rate)
    return growth_rates


def _get_zero_matrix(model: TreeDemographicModel) -> ndarray:
    leaf_count = len([e for e in model.events if isinstance(e, Leaf)])
    zero_matrix = np.zeros([leaf_count, leaf_count], dtype=int)
    return zero_matrix


def _get_migration_matrices(model, values: ParameterValues) -> Tuple[ndarray, ...]:
    """
    Gathers all migration matrices in the model in a single tuple.
    Also adds a zero matrix at the end of the tuple.
    :param values: Model parameter values
    :return: Migration matrices.
    """
    # TODO: Write code for resizing all matrices to one size according to the population count
    matrices = []
    for event in model.events:
        if isinstance(event, Epoch):
            matrix: ndarray = event.mig_args.copy()
            if matrix is None:
                continue
            matrix = model.get_value_from_var2value(values, matrix)
            matrices.append(matrix)
    matrices += _get_zero_matrix(model)
    return tuple(matrices)


def _first_event(model: TreeDemographicModel):
    non_leaf_events = [e for e in model.events if not isinstance(e, Leaf)]
    first_event = non_leaf_events[0]
    return first_event


def _leaves(model: TreeDemographicModel) -> List[Leaf]:
    leaves = []
    for event in model.events:
        if isinstance(event, Leaf):
            leaves.append(event)
    leaves.sort(key=lambda leaf: leaf.pop)
    return leaves


def _find_max_obs_lhood(workdir, prefix) -> float:
    best_lhood_fpath = Path(workdir, prefix, f'{prefix}.bestlhoods')
    best_lhood_data = np.loadtxt(best_lhood_fpath, delimiter='\t', skiprows=1)
    max_obs_lhood = best_lhood_data[-1]
    assert isinstance(max_obs_lhood, float)
    return max_obs_lhood


def _new_sfs_name(sfs_file_name: str, base_name: str) -> str:
    """
    Replaces the base name (or adds one if absent) of an fsc2 SFS file
    with another base name. Mainly used to conform all input files for
    fastsimcoal2 to a single format, so that it actually recognises them.

    Accepted input file name formats:

    * ``*DAFpop0.obs``, ``*_DAFpop0.obs``
    * ``*MAFpop0.obs``, ``*_MAFpop0.obs``
    * ``*jointDAFpop1_0.obs``, ``*_jointDAFpop1_0.obs``
    * ``*jointMAFpop1_0.obs``, ``*_jointMAFpop1_0.obs``
    * ``*DSFS.obs``, ``*_DSFS.obs``
    * ``*MSFS.obs``, ``*_MSFS.obs``

    :param sfs_file_name: Name of the file to be renamed. For expected formats see the list.
    :param base_name: New base name of the file (usually something like "gadma_fsc2")

    :return: New file name.

    :raise ValueError: if *sfs_file_name* doesn't conform to expected file name formats.
    """
    pattern_ = re.compile(r'(.*)(_(?:DAFpop0|MAFpop0|jointDAFpop1_0|jointMAFpop1_0|DSFS|MSFS)\.obs)')
    pattern = re.compile(r'(.*)((?:DAFpop0|MAFpop0|jointDAFpop1_0|jointMAFpop1_0|DSFS|MSFS)\.obs)')
    match_ = re.search(pattern_, sfs_file_name)
    match = re.search(pattern, sfs_file_name)
    if match is not None:
        # '*jointDAFpop1_0.obs' -> '{base_name}_jointDAFpop1_0.obs
        suffix = match.group(2)
        new_name = f'{base_name}_{suffix}'
    elif match_ is not None:
        # '*_jointDAFpop1_0.obs' -> '{base_name}_jointDAFpop1_0.obs
        suffix = match_.group(2)
        new_name = f'{base_name}{suffix}'
    else:
        raise ValueError(f"File name {sfs_file_name} does not conform to expected format")
    return new_name


def _infer_sample_sizes(sfs_data: Spectrum) -> Tuple[int, ...]:
    """
    Infer sample sizes from the observed SFS file.
    :return: Sample sizes.
    """
    return tuple(sfs_data.sample_sizes.tolist())


class FastSimCoal2InputFiles(NamedTuple):
    template_file: str
    estimation_file: str
    definition_file: str


class FastSimCoal2Engine(Engine):
    """
    Class representing the fastsimcoal2 engine

    **Note:** when passing SFS data to the engine,
    files with more than two observed samples should be multidimensional SFS
    (refer to `fastsimcoal2 manual <http://cmpg.unibe.ch/software/fastsimcoal2/man/fastsimcoal27.pdf>`_ for details).
    """
    id: ClassVar = 'fsc2'
    supported_models: Final = [CustomDemographicModel,
                               EpochDemographicModel,
                               TreeDemographicModel]
    # TODO: Edit the failed tests to skip the VCFDataHolder check
    supported_data: Final = [SFSDataHolder]
    inner_data_type: Final = str

    can_evaluate: Final = True

    data: inner_data_type
    data_holder: SFSDataHolder

    PREFIX: Final[str] = 'gadma_fsc2'

    def __init__(self, data=None, model=None):
        super().__init__(data, model)
        self._fsc2_complex_parameters: Dict[str, str] = {}

    def evaluate(self, values: ParameterValues, **options):
        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")

        var2value = self._process_values(values)

        if isinstance(self.model, CustomDemographicModel):
            assert (isinstance(self.model.function, FastSimCoal2InputFiles)
                    and all(isinstance(elem, (str, os.PathLike)) for elem in self.model.function))
            tpl_file, est_file, def_file = self.model.function
        else:
            fsc2_model, values = _get_fsc2_model(self.model, values)
            self.model = fsc2_model

        with tempfile.TemporaryDirectory(prefix=f'{self.PREFIX}_') as workdir:
            sfs_name = Path(self.data).name
            new_sfs_name = _new_sfs_name(sfs_name, self.PREFIX)
            print(f'{sfs_name} -> {new_sfs_name}', end=LINE)
            new_sfs_path = Path(workdir, new_sfs_name)
            shutil.copy(self.data, new_sfs_path)

            if isinstance(self.model, TreeDemographicModel):
                tpl_file, est_file, def_file = self.generate_code(var2value)
                tpl_file.save_to_file(Path(workdir, f'{self.PREFIX}.tpl'))
                est_file.save_to_file(Path(workdir, f'{self.PREFIX}.est'))
                def_file.save_to_file(Path(workdir, f'{self.PREFIX}.def'))

            n_simulations: int = options['n_sims'] if 'n_sims' in options else 1000
            n_loops: int = options['n_loops'] if 'n_loops' in options else 20

            # See fastsimcoal2 manual (http://cmpg.unibe.ch/software/fastsimcoal2/man/fastsimcoal27.pdf)
            # for full command line options descriptions.
            args = ['-t', Path(tpl_file.path).name,  # template parameter file
                    '-e', Path(est_file.path).name,  # parameter estimation file
                    '-F', Path(def_file.path).name,  # definition file (for the fixed variables)
                    '-M',  # estimation by composite max likelihood
                    '-n', n_simulations,  # number of simulations
                    '-L', n_loops]  # number of ECM cycles

            if self.data_holder.outgroup is True:
                # If 'outgroup' is True, then the spectrum is not folded (i.e. *not* minor allele)
                args.append('-d')
            else:
                args.append('-m')

            if is_multi_sfs(self.data_holder):
                args.append('--multiSFS')

            print([f.name for f in Path(workdir).iterdir()])
            self.run_fsc2(FSC2_PATH, args, workdir=workdir)
            print('-' * 80)

            likelihood = _find_max_obs_lhood(workdir, self.PREFIX)
        return likelihood

    def _process_values(self, values: ParameterValues) -> Dict:
        """
        Process demographic model parameter values for fsc2 usage.

        :param values: Parameter values.
        :return: Processed values.
        """
        var2value: dict = self.model.var2value(values)

        n_anc_var = self.model.Nanc_size
        n_anc = self.model.get_value_from_var2value(var2value, n_anc_var)

        values = self.model.translate_values(
            units="physical",
            values=values,
            Nanc=n_anc
        )

        return self.model.var2value(values)

    def generate_code(self,
                      values: ParameterValues,
                      filename: Optional[Union[str, os.PathLike]] = None,
                      nanc=None,
                      gen_time=None,
                      gen_time_units="years") -> Optional[Tuple[TemplateFile,
                                                                EstimationFile,
                                                                DefinitionFile]]:
        """
        Generate three input files for fastsimcoal2.

        :param values: Model parameter values
        :param filename: If this parameter is a string or a path-like object,
            save the files in the directory specified by the parameter.
            In this case, the function won't return anything.

            TODO: @Ekaterina Noskova: 'path' might be a better name for this parameter
        :param nanc: Unused.
        :param gen_time: Unused.
        :param gen_time_units: Unused.
        :return: A tuple with three :py:class:`fsc2_generator.FSC2InputFile`: objects
        """
        tpl_file = self._generate_template_file()
        est_file = self._generate_estimation_file(values)
        def_file = self._generate_definitions_file(values)
        if isinstance(filename, (str, os.PathLike)):
            assert Path(filename).exists()
            tpl_file.save_to_file(filename + f"{self.PREFIX}.tpl")
            est_file.save_to_file(filename + f"{self.PREFIX}.est")
            def_file.save_to_file(filename + f"{self.PREFIX}.def")
            return None
        return tpl_file, est_file, def_file

    def _generate_template_file(self):
        file = TemplateFile()

        current_n_of_pops: int = self.sfs_data.Npop
        file.write_number_of_population_samples(current_n_of_pops)

        current_pop_sizes: Tuple = _get_initial_pop_sizes(self.model)
        file.write_initial_effective_population_sizes(current_pop_sizes)

        sample_sizes: Tuple[int, ...] = _infer_sample_sizes(self.sfs_data)
        file.write_sample_sizes(sample_sizes)

        initial_growth_rates: Tuple[Union[int, str]] = self._get_initial_growth_rates()
        file.write_initial_growth_rates(initial_growth_rates)

        matrices: Tuple[ndarray, ...] = tuple([_get_zero_matrix(self.model)])
        file.write_migration_matrices(matrices)

        events: Tuple[dict, ...] = self._get_events()
        file.write_multiple_events(events)

        file.write_chromosomes(number_of_chromosomes=1)
        file.write_chromosome_blocks(blocks=1)
        file.write_block_properties(data_type='FREQ', markers=1,
                                    recombination_rate=0, mutation_rate=2.5e-8)
        return file

    @property
    def sfs_data(self) -> Spectrum:
        from gadma.engines.dadi_moments_common import read_sfs_data
        import dadi
        data_holder = self.data_holder if self.data_holder is not None else SFSDataHolder(self.inner_data)
        sfs = read_sfs_data(dadi, data_holder)
        assert isinstance(sfs, dadi.Spectrum_mod.Spectrum)
        return sfs

    def _get_initial_growth_rates(self) -> Tuple[Union[int, str], ...]:
        leaves = _leaves(self.model)
        growth_rates = []
        for leaf in leaves:
            if leaf.g == 0:
                growth_rate = leaf.g
            else:
                growth_rate = self._g_to_complex_param(leaf.g, leaf.dyn)
            growth_rates.append(growth_rate)
        return tuple(growth_rates)

    def _get_events(self) -> Tuple[Dict[str, Union[str, int]], ...]:
        model_events: list = self.model.events

        fsc2_events = []
        for event in model_events:
            fsc2_event = {}
            if isinstance(event, Leaf):
                continue

            if isinstance(event.t, Addition):
                time: str = self._time_to_complex_param(event.t)
            else:
                time = event.t.name + '$'

            if isinstance(event, PopulationSizeChange):
                source: int = event.pop
                sink: int = event.pop
                migrants: int = 0
                new_sink_size: str = event.size_pop.name + '$'
                if event.g == 0:
                    new_sink_growth_rate: int = 0
                else:
                    new_sink_growth_rate: str = self._g_to_complex_param(event.g, event.dyn)
                new_migration_matrix = 0
            elif isinstance(event, LineageMovement):
                source = event.pop_from
                sink = event.pop
                migrants = 1
                new_sink_size = event.size_pop.name + '$'
                new_sink_growth_rate: str = 'keep'
                new_migration_matrix = 'keep'
            else:
                raise RuntimeError(f'Wrong event type, expected types: '
                                   f'{PopulationSizeChange, LineageMovement}')
            fsc2_event.update({"time": time,
                               "source": source,
                               "sink": sink,
                               "migrants": migrants,
                               "new_sink_size": new_sink_size,
                               "new_sink_growth_rate": new_sink_growth_rate,
                               "new_migration_matrix": new_migration_matrix})
            fsc2_events.append(fsc2_event)
        return tuple(fsc2_events)

    @staticmethod
    def run_fsc2(fsc2_path: str,
                 args: Sequence[str],
                 workdir: Union[str, os.PathLike]) -> NoReturn:
        """
        Run fastsimcoal2 with specified arguments.

        :param fsc2_path: Path to fastsimcoal2 binary file.
        :param args: Arguments to pass to fastsimcoal2.
        :param workdir: Working directory for calculations.
        """
        assert Path(fsc2_path).exists(), "Can't find fsc2 binary"
        args: str = " ".join([str(e) for e in args])
        command = f"cd {workdir}; {fsc2_path} {args}"
        print(command)
        os.system(command)
        # TODO: Come up with a way to use the `subprocess` library
        # stdout = io.StringIO()
        # with redirect_stdout(stdout):
        #     retcode = subprocess.call([fsc2_path, args], shell=True, cwd=workdir)

    def update_data_holder_with_inner_data(self):
        self.data_holder.projections = _infer_sample_sizes(self.sfs_data)
        self.data_holder.population_labels = None
        self.data_holder.outgroup = None

    @classmethod
    def _read_data(cls, data_holder: SFSDataHolder) -> inner_data_type:
        sfs_path = Path(data_holder.filename)
        if data_holder.outgroup is None:
            raise RuntimeError("Parameter 'outgroup' of the data holder is not set")
        is_folded = not data_holder.outgroup
        if is_folded != ('MSFS' in sfs_path.name or 'MAF' in sfs_path.name):
            raise RuntimeError(f'Site frequency spectrum {sfs_path}'
                               f'is expected to be {"not" if not is_folded else ""} folded, '
                               f'but the file name does not conform to that; '
                               'check if the file name is correct.')

        return data_holder.filename

    def _g_to_complex_param(self, g: Division, dyn: DynamicVariable) -> str:
        """
        Convert growth rate variable to fsc2-style complex parameter.

        :param g: VariablesCombination that determines the growth rate formula.
        :param dyn: Population dynamic variable.
        :return: fsc2-style complex parameter name.
        """
        n_t = g.variables[0].name
        n_0 = g.variables[1].name
        n_div_name = f'{n_t}_{n_0}div$'
        n_div_parameter = f'{n_t}$/{n_0}$'

        complex_parameters = self._fsc2_complex_parameters
        if n_div_name not in complex_parameters:
            complex_parameters.update({n_div_name: n_div_parameter})
        self._fsc2_complex_parameters = complex_parameters

        time = g.variables[2]
        if isinstance(time, Addition):
            time_name = self._time_to_complex_param(time)
        else:
            time_name = time.name + '$'

        dyn_name = dyn.name + '$'
        dyn_parameter = f'log({n_div_name})/{time_name}'

        if dyn_name not in complex_parameters:
            complex_parameters.update({dyn_name: dyn_parameter})
        self._fsc2_complex_parameters = complex_parameters
        return dyn_name

    def _time_to_complex_param(self, time_var: Addition) -> str:
        """
        Converts time variable to fsc2-style complex parameter.

        :param time_var: Time variable.
        :return: fsc2-style complex parameter name.
        """
        var_names = [v.name for v in time_var.variables]
        name = '_'.join(var_names) + 'sum$'
        parameter = ' + '.join([f'{n}$' for n in var_names])
        complex_parameters = self._fsc2_complex_parameters
        if name not in complex_parameters:
            complex_parameters.update({name: parameter})
        self._fsc2_complex_parameters = complex_parameters
        return name

    def _generate_estimation_file(self, values: ParameterValues) -> EstimationFile:
        file = EstimationFile()

        params: List[Dict] = self._variables_to_fsc2_est_params(values)
        file.write_parameters(params)

        compl_params = self._complex_parameters()
        file.write_complex_parameters(compl_params)
        return file

    def _variables_to_fsc2_est_params(self, values: ParameterValues) -> List[Dict]:
        variables = self.model.variables
        est_params = []
        for variable in variables:
            if isinstance(variable, DynamicVariable):
                continue
            param = {'is_int': 1,
                     'name': variable.name + '$',
                     'dist': 'unif',
                     'minimum': variable.domain[0],
                     'maximum': variable.domain[1],
                     'output': True,
                     'bounded': True}
            var_value = self.get_value_from_var2value(values, variable)
            if isinstance(var_value, float):
                param['is_int'] = 0
            est_params.append(param)
        return est_params

    def _complex_parameters(self) -> Sequence[Mapping[str, Union[int, str]]]:
        complex_params = []
        fsc2_complex_params = self._fsc2_complex_parameters
        for name, value in fsc2_complex_params.items():
            param = {'is_int': 1,
                     'name': name,
                     'definition': value,
                     'output': False}
            if 'dyn' in name or '/' in value:
                param['is_int'] = 0
            if not param['name'].endswith('$'):
                param['name'] += '$'
            complex_params.append(param)
        return complex_params

    def _generate_definitions_file(self, values: ParameterValues) -> DefinitionFile:
        names = tuple(var.name for var in self.model.variables
                      if not isinstance(var, DynamicVariable))
        vals = [values[var] for var in self.model.variables
                if not isinstance(var, DynamicVariable)]
        file = DefinitionFile(names, vals)
        return file


if Path(FSC2_PATH).exists():
    register_engine(FastSimCoal2Engine)
