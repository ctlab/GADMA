import functools

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
    """
    def __init__(self, vcf_file, popmap_file, sample_sizes, outgroup,
                 pop_labels=None, seq_len=None,  bed_file=None):
        if pop_labels is None:
            pop_labels = set()
            with open(popmap_file) as f:
                for line in f:
                    pop_labels.add(line.split()[-1])
        super(VCFDataHolder, self).__init__(vcf_file, sample_sizes, outgroup,
                                            pop_labels, seq_len)
        self.popmap_file = popmap_file
        self.bed_file=bed_file
