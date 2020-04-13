import functools

class DataHolder(object):
    """
    Class for data holding.

    :param filename: name of file with the data.
    :type filename: str
    :param outgroup: information if there is an outgroup in the data.
    :type outgroup: bool
    :params pop_labels: labels of populations in the data
    :type pop_labels: list of str
    :param seq_len: length of the sequence that was used to build the data
    :type seq_len: int

    :note: if any parameter in :func:`__init__` is None then if it is\
        possible it will be taken from the file after reading.
   """
    def __init__(self, filename, outgroup, pop_labels, seq_len=None):
        self.data = None
        self.filename = filename
        self.seq_len = seq_len
        self.outgroup = outgroup
        self.pop_labels = pop_labels
        self.ready_for_engine = None

    def prepare_for_engine(self, engine):
        """
        Reads data from file in format of the corresponding engine.

        :param engine: class or instance of engine.
        :type engine: :class:`gadma.Engine`
        """
        self.data = engine.read_data(self)
        self.ready_for_engine = engine.id
        self.pop_labels = engine.get_pop_labels(self.data)
        self.outgroup = engine.get_outgroup(self.data)
        self.seq_len = self.seq_len or engine.get_seq_len(self.data)


class SFSDataHolder(DataHolder):
    """
    Class for SFS data holding.
    
    :param sfs_file: name of file with the SFS data
    :type sfs_file: str
    :param outgroup: information if there is an outgroup in the data
    :type outgroup: bool
    :params pop_labels: labels of populations in the SFS data
    :type pop_labels: list of str
    :param seq_len: length of the sequence that was used to build the data
    :type seq_len: int
    :param sample_sizes: population sample sizes
    :type sample_sizes: list of int

     :note: if any parameter in :func:`__init__` is None then if it is\
        possible it will be taken from the file after reading.
   """
    def __init__(self, sfs_file, outgroup=None, pop_labels=None, seq_len=None, sample_sizes=None):
        super(SFSDataHolder, self).__init__(sfs_file, outgroup, pop_labels, seq_len)
        self.sample_sizes = sample_sizes

    def prepare_for_engine(self, engine):
        super(SFSDataHolder, self).prepare_for_engine(engine)
        self.sample_sizes = engine.get_sample_sizes(self.data)


class VCFDataHolder(DataHolder):
    """
    Class for VCF data holding.

    :param vcf_file: name of file with the SFS data
    :type vcf_file: str
    :param popmap_file: name of file with the popmap data
    :type popmap_file: str
    :param outgroup: information if there is an outgroup in the data
    :type outgroup: bool
    :params pop_labels: labels of populations in the SFS data
    :type pop_labels: list of str
    :param seq_len: length of the sequence that was used to build the data, 
        defaults to None
    :type seq_len: int
    :param bed_file: name of bed file to filter SNP's in VCF file,
        defaults to None
    :type bed_file: str

    :note: if any parameter in :func:`__init__` is None then if it is\
        possible it will be taken from the file after reading.

    """
    def __init__(self, vcf_file, popmap_file, outgroup, seq_len=None,  bed_file=None):
        pop_labels = set()
        with open(popmap_file) as f:
            for line in f:
                pop_labels.add(line.split()[-1])
        super(VCFDataHolder, self).__init__(vcf_file, outgroup, pop_labels, seq_len)
        self.popmap_file = popmap_file
        self.bed_file=bed_file
