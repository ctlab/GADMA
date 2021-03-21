from ..utils import check_file_existence, ensure_file_existence
from ..utils import read_popinfo, get_list_of_names_from_vcf
import warnings


class DataHolder(object):
    """
    Class for data holding.

    : param filename: name of file with data
    : param outgroup: information if there is outgroup in data
    : type outgroup: bool
    : params pop_labels: labels of populations in data
    : param seq_len: length of sequence that was used to build data
    """
    def __init__(self, filename, projections,
                 outgroup, population_labels, sequence_length):
        self.data = None
        self.filename = filename
        self.projections = projections
        self.outgroup = outgroup
        self.population_labels = population_labels
        self.sequence_length = sequence_length

        if self.filename is not None and check_file_existence(self.filename):
            self.filename = ensure_file_existence(self.filename)


class SFSDataHolder(DataHolder):
    """
    Class for SFS data holding.
    if any parameter is None then it will be taken from the file
    """
    def __init__(self, sfs_file, projections=None, outgroup=None,
                 population_labels=None, sequence_length=None):
        super(SFSDataHolder, self).__init__(sfs_file, projections,
                                            outgroup, population_labels,
                                            sequence_length)


class VCFDataHolder(DataHolder):
    """
    Class for VCF data holding.
    It saves some information while it is created.
    """
    def __init__(self, vcf_file, popmap_file, sample_sizes=None, outgroup=None,
                 ploidy=2, population_labels=None, seq_len=None, bed_file=None,
                 reference_file=None):
        sample2pop = read_popinfo(popmap_file)
        samples = get_list_of_names_from_vcf(vcf_file)
        wrong_samples = list()
        for sample in sample2pop:
            if sample not in samples:
                wrong_samples.append(sample)
        if len(wrong_samples) > 0:
            warnings.warn(f"Some samples from popmap file are not presented "
                          f"in VCF file: {wrong_samples}. They were ignored.")
            for sample in wrong_samples:
                del sample2pop[sample]
        # Check pop labels
        pop_labels = list()
        for sample in samples:
            if sample in sample2pop:
                if sample2pop[sample] not in pop_labels:
                    pop_labels.append(sample2pop[sample])
        if population_labels is None:
            population_labels = pop_labels
        if set(population_labels) != set(pop_labels):
            raise AssertionError("Population labels are different in popmap "
                                 f"file: {population_labels} != {pop_labels}")

        # Check sample sizes
        sizes = [sum([x == pop for x in sample2pop.values()])
                 for pop in population_labels]
        sizes = [x * ploidy for x in sizes]
        if sample_sizes is None:
            sample_sizes = sizes
        sample_sizes = list(sample_sizes)
        assert sample_sizes == sizes, ("Sizes are not equal: "
                                       f"{sample_sizes} != {sizes}")
        super(VCFDataHolder, self).__init__(vcf_file, sample_sizes, outgroup,
                                            population_labels, seq_len)
        self.popmap_file = popmap_file
        self.bed_file = bed_file
        self.ploidy = ploidy
        self.reference_file = reference_file
