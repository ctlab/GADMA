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
    def __init__(self, filename, sample_sizes,
                 outgroup, pop_labels, seq_len):
        self.data = None
        self.filename = filename
        self.sample_sizes = sample_sizes
        self.outgroup = outgroup
        self.pop_labels = pop_labels
        self.seq_len = seq_len
        self.ready_for_engine = []

    def prepare_for_engine(self, engine):
        """
        Reads data from file in format of corresponding engine.

        : param engine: class or instance of engine
        : type engine: gadma.Engine
        """
        self.data = engine.read_data(self)
        self.ready_for_engine.append(engine.base_module.__name__)
        self.pop_labels = engine.get_pop_labels(self.data)
        self.outgroup = engine.get_outgroup(self.data)
        self.seq_len = engine.get_seq_len(self.data)
        self.sample_sizes = engine.get_sample_sizes(self.data)


class SFSDataHolder(DataHolder):
    """
    Class for SFS data holding.
    if any parameter is None then it will be taken from the file
    """
    def __init__(self, sfs_file, sample_sizes=None, outgroup=None,
                 pop_labels=None, seq_len=None):
        super(SFSDataHolder, self).__init__(sfs_file, sample_sizes,
                                            outgroup, pop_labels, seq_len)


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
