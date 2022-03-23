import io
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import NoReturn, Union, Any, Optional, Tuple, List, Callable, Final, Literal

import numpy as np
from numpy import ndarray

DataType = Literal['dna', 'microsat', 'standard', 'freq',
                   'DNA', 'MICROSAT', 'STANDARD', 'FREQ']


def _ndarray_to_multiline_str(array: ndarray, formatter: Optional[dict] = None) -> str:
    """
    Turn a NumPy array into a string.

    Example --  this array::

        array([[1, 2],
               [3, 4]])

    ...is turned into::

        1 2
        3 4

    :param array: Array to convert
    :param formatter: Formatter passed to ``numpy.array2string``
    :return: NumPy array as a string
    """
    matrix: str = np.array2string(array, separator=' ', formatter=formatter)
    matrix = matrix.replace('[', '')
    matrix = matrix.replace(']', '')
    matrix = matrix.replace('\n ', '\n')
    return matrix


def _format_as_fsc2_keyword(value: Union[int, float, str]) -> Union[int, float, str]:
    """Formats input as a fastsimcoal2 keyword, if the input is a string"""
    if isinstance(value, str) and not value.endswith('$'):
        return value + '$'
    return value


class FSC2InputFile(object):
    """Base class for fastsimcoal2 input files."""
    def __init__(self):
        self._path: Optional[Path] = None
        self.file: StringIO = io.StringIO()

    def write(self, line: str) -> NoReturn:
        self.file.write(f'{line}\n')

    @property
    def path(self) -> str:
        return str(self._path)

    def write_comment(self, comment: str) -> NoReturn:
        """Write a comment to the file."""
        self.write(f'// {comment}')

    def save_to_file(self, path: Union[str, PathLike]) -> NoReturn:
        """Save in-memory representation of the input file in filesystem"""
        self._path = Path(path)
        self._path.write_text(self.file.getvalue())


class TemplateFile(FSC2InputFile):
    """Class for fastsimcoal2 template files."""
    def write_number_of_population_samples(self, samples: int) -> NoReturn:
        """Write the number of population samples to the file.

        :param samples: Number of population samples"""
        self.write_comment('Number of population samples (demes)')
        self.write(f'{samples}')

    def write_initial_effective_population_sizes(self, sizes: tuple) -> NoReturn:
        """Write effective population sizes to the file.

        :param sizes: Effective population sizes"""
        self.write_comment('Population effective sizes (number of genes)')
        self._write_multiline_data(sizes)

    def write_sample_sizes(self, sizes: tuple) -> NoReturn:
        self.write_comment('Sample sizes')
        self._write_multiline_data(sizes)

    def write_initial_growth_rates(self, rates: Tuple[Union[str, int, float], ...]) -> NoReturn:
        self.write_comment('Growth rates (negative growth implies population expansion)')
        self._write_multiline_data(rates)

    def _write_multiline_data(self, data: Tuple[Union[str, int, float], ...]) -> NoReturn:
        """Write data that spans multiple lines.

        :param data: data to write"""
        for point in data:
            if isinstance(point, (float, int)):
                self.write(f'{str(point)}')
            else:
                self.write(f'{_format_as_fsc2_keyword(str(point))}')

    def write_migration_matrices(self, matrices: Tuple[ndarray, ...]) -> NoReturn:
        self.write_comment('Number of migration matrices. 0 implies no migration between demes')
        self.write(f'{len(matrices)}')
        for array, n in zip(matrices, range(0, len(matrices))):
            if array.ndim > 2:
                raise ValueError(f'arrays should be 2-dimensional. Offending array\'s index: {n}')
            self.write_comment(f'migration matrix {n}')
            formatter: dict = {'str_kind': lambda x: _format_as_fsc2_keyword(x)}
            matrix = _ndarray_to_multiline_str(array, formatter)
            self.write(f'{matrix}')

    def write_multiple_events(self, events: Tuple[dict, ...]) -> NoReturn:
        """
        Write historical events to the template file.

        :param events: A tuple containing dictionaries that describe the events.

        Each dictionary should contain following keys:

        * ``time``
        * ``source``
        * ``sink``
        * ``migrants``
        * ``new_sink_size``

        Optional keys:

        * ``new_sink_growth_rate`` (default value: ``'keep'``)
        * ``new_migration_matrix`` (default value: ``'keep'``)
        * ``no_migrations`` (default value: ``False``)
        """
        self.write_comment('hist. event: time, source, sink, migrants, '
                           'new size, new growth rate, migr. matrix')
        self.write(f'{len(events)} historical events')
        for event in events:
            self._write_event(**event)

    def _write_event(self,
                     time: Union[float, str],
                     source: int,
                     sink: int,
                     migrants: Union[float, str],
                     new_sink_size: Union[float, str],
                     new_sink_growth_rate: Union[float, str] = 'keep',
                     new_migration_matrix: Union[int, str] = 'keep',
                     no_migrations: bool = False,
                     absolute_resize: bool = True) -> NoReturn:
        """
        Write a line containing event details.

        :param time: Number of generations *t* before
            present at which the historical event happened.
        :param source: Source deme (the first listed deme has index 0).
        :param sink: Destination (sink) deme.
        :param migrants: Expected proportion of migrants to move
            from source to sink. Note that this proportion is not
            fixed, and that it also represents the probability
            for each lineage in the source deme to
            migrate to the sink deme.
        :param new_sink_size: New size for the **sink** deme,
            relative to its size at generation *t*.
            If ``absolute_resize`` is ``True``, this is treated
            as an absolute value instead of a relative one.
        :param new_sink_growth_rate: New growth rate for the **sink** deme.
        :param new_migration_matrix: New migration matrix
            to be used further back in time.
        :param no_migrations: If ``True``, migrations
            between demes are suppressed until the end
            of the current coalescent simulation. If
            next in line historical events were to specify
            the use of some new migration matrix, this
            would be ignored by fastsimcoal.
        :param absolute_resize: If ``True``, ``new_sink_size`` is treated
            as an absolute value. ``True`` by default.
        """
        time = _format_as_fsc2_keyword(time)
        migrants = _format_as_fsc2_keyword(migrants)
        new_sink_size = _format_as_fsc2_keyword(new_sink_size)
        if new_sink_growth_rate != 'keep':
            new_sink_growth_rate = _format_as_fsc2_keyword(new_sink_growth_rate)
        if new_migration_matrix != 'keep':
            new_migration_matrix = _format_as_fsc2_keyword(new_migration_matrix)

        event: str = (f'{time} {source} {sink} {migrants} '
                      f'{new_sink_size} {new_sink_growth_rate} {new_migration_matrix}')
        if no_migrations:
            event += ' nomig'
        if absolute_resize:
            event += ' absoluteResize'
        self.write(f'{event}')

    def write_chromosomes(self,
                          number_of_chromosomes: int,
                          different_structure: bool = False) -> NoReturn:
        """
        Write chromome data to the file.

        :param number_of_chromosomes: Number of independent chromosomes.
        :param different_structure: Specifies whether chromosomes have a different structure.
        """
        if different_structure:
            different_structure = 1
        else:
            different_structure = 0
        self.write_comment('Number of independent loci/chromosome')
        self.write(f'{number_of_chromosomes} {different_structure}')

    def write_chromosome_blocks(self, blocks: int) -> NoReturn:
        """
        Write number of chromosome blocks to the template file.

        :param blocks: Number of chromosome blocks
        """
        self.write_comment('Number of linkage blocks per chromosome')
        self.write(f'{blocks}\n')

    def write_block_properties(self,
                               data_type: DataType,
                               markers: Union[int, str],
                               recombination_rate: Union[float, str],
                               **kwargs) -> NoReturn:
        """
        Write properties of a single chromosome block.

        :param data_type: Type of data.
        Possible values are: ``dna``, ``microsat``, ``standard``, ``freq``.
        :param markers: Number of markers with this data type
        to be simulated. For DNA, this is the sequence length.
        :param recombination_rate: Recombination rate
        between adjacent markers (between adjacent nucleotides for DNA).

        ------

        Optional arguments:

        ``mutation_rate``
            (*float* or *string*) Mutation rate per basepair (for 'dna' data type),
            per locus (for 'microsat' data type), per marker (for 'standard' data type).

        ``transition_rate``
            (*float* or *string*) Transition rate
            (fraction of substitutions that are transitions).
            A value of 0.33 implies no transition bias.

            Use with ``data_type = 'dna'``.

        ``geometric_parameter``
            (*float* or *string*) Value of the geometric parameter
            for a Generalized Stepwise Mutation (GSM) model.
            This value represents the proportion of mutations
            that will change the allele size by more than one step.

            Values between 0 and 1 are required.

            A value of 0 is for a strict Stepwise Mutation Model (SMM).

            Use with ``data_type = 'microsat'``.
        """
        data_type = data_type.lower()
        markers = _format_as_fsc2_keyword(markers)
        recombination_rate = _format_as_fsc2_keyword(recombination_rate)

        if data_type == 'dna':
            mutation_rate: Union[float, str] = _format_as_fsc2_keyword(kwargs['mutation_rate'])
            transition_rate: Union[float, str] = _format_as_fsc2_keyword(kwargs['transition_rate'])
        elif data_type == 'microsat':
            mutation_rate: Union[float, str] = _format_as_fsc2_keyword(kwargs['mutation_rate'])
            geometric_parameter: Union[float, str] = _format_as_fsc2_keyword(kwargs['geometric_parameter'])
            if not 0 <= geometric_parameter <= 1:
                raise ValueError('geometric_parameter out of bounds [0, 1]')
        elif data_type == 'standard':
            mutation_rate: Union[float, str] = _format_as_fsc2_keyword(kwargs['mutation_rate'])
        elif data_type == 'freq':
            mutation_rate: float = np.format_float_positional(kwargs['mutation_rate'])

        output: str = f'{data_type.upper()} {markers} {recombination_rate} {mutation_rate}'

        if data_type == 'dna':
            output += f' {transition_rate}'
        elif data_type == 'microsat':
            output += f' {geometric_parameter}'
        self.write(output)


class EstimationFile(FSC2InputFile):
    """Class for fastsimcoal2 estimation files."""
    def __init__(self):
        super().__init__()
        self.write_comment('Priors and rules file')
        self.write_comment('*********************')

    # TODO: Implement the `padding`
    def write_parameters(self, parameters: List[dict], padded: bool = False) -> NoReturn:
        """
        Write parameters.

        :param parameters: List of parameters to write. Each parameter is defined by a dictionary
            (see FSC2EstimationFile._write_single_parameter method for keys and value types).
        :param padded: [Unused] Make the output neatly aligned using whitespace padding.
        """
        self.write('[PARAMETERS]')
        self.write_comment('#isInt? #name    #dist.  #min #max')
        for p in parameters:
            self._write_single_parameter(**p)
        self.write('\n')

    def _write_single_parameter(self,
                                is_int: bool,
                                name: str,
                                dist: str,
                                minimum: Union[float, int],
                                maximum: Union[float, int],
                                output: bool,
                                bounded: bool) -> NoReturn:
        """
        Write a single parameter to the file.

        :param is_int: Flag to indicate whether the parameter is an integer.
        :param name: Name of the parameter.
        :param dist: Distribution of the parameter.
        :param minimum: Minimum value.
        :param maximum: Maximum value.
        :param output: Flag to indicate whether to output the optimized parameter value or not.
        :param bounded:
        """
        is_int, name, output, bounded = self._process_arguments(is_int, name, output, bounded)
        self.write(f'{is_int} {name} {dist} {minimum} {maximum} {output} {bounded}')

    @staticmethod
    def _process_arguments(is_int: bool,
                           name: str,
                           output: bool,
                           bounded: bool = None) -> Tuple[int, str, str, str]:
        if is_int:
            is_int = 1
        else:
            is_int = 0
        name = _format_as_fsc2_keyword(name)
        if output:
            output = 'output'
        else:
            output = 'hide'
        if bounded:
            bounded = 'bounded'
        else:
            bounded = ''
        return is_int, name, output, bounded

    def write_complex_parameters(self, parameters: List[dict]) -> NoReturn:
        """
        Write complex parameters.

        :param parameters: List of parameters. Each parameter should be a dictionary
            (see :py:meth:`_write_single_complex_parameter` for keys).
        """
        self.write('[COMPLEX PARAMETERS]')
        for p in parameters:
            self._write_single_complex_parameter(**p)

    def _write_single_complex_parameter(self,
                                        is_int: bool,
                                        name: str,
                                        definition: str,
                                        output: bool) -> NoReturn:
        is_int, name, output, _ = self._process_arguments(is_int, name, output)
        self.write(f'{is_int} {name} = {definition} {output}')


class DefinitionFile(FSC2InputFile):
    """Class for fastsimcoal2 definition files."""
    def __init__(self, parameters_names: Tuple[str, ...], parameters_values: Union[ndarray, list]):
        super().__init__()
        self.parameters_names: tuple = tuple(_format_as_fsc2_keyword(n)
                                             for n in parameters_names)
        self.parameters_values: Union[ndarray, list] = parameters_values

        delimiter = ' '
        self.write(delimiter.join(self.parameters_names))
        self.write_parameters_values()

    def write_parameters_values(self):
        if isinstance(self.parameters_values, ndarray):
            values = _ndarray_to_multiline_str(self.parameters_values)
        else:
            values = ' '.join(str(v) for v in self.parameters_values)
        self.write(values)
