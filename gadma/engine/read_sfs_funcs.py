import os
import numpy as np

# Those functions are common for dadi and moments engines.
def _check_missing_pop_labels(sfs, default_pop_labels=None):
    """
    Check that SFS has population labels. If not then make them default.

    : param sfs: site frequency spectrum to check
    : type sfs: dadi.Spectrum (or analogue)
    : param default_pop_labels: if pop. labels are missing use this values.
    """
    if sfs.pop_ids is None:
        if default_pop_labels is not None:
            Warning("Spectrum file %s is in an old format - without population labels, so they will be taken from corresponding parameter: %s." % (filename, ', '.join(pop_labels)))
            sfs.pop_ids = pop_labels
        else:
            sfs.pop_ids = ['Pop %d' % (i+1) for i in range(sfs.ndim)]
    return sfs


def _new_pop_labels(sfs, new_labels):
    """
    Assign new order of population labels of SFS.

    : param sfs: site frequency spectrum to change its labels
    : type sfs: dadi.Spectrum (or analogue)
    : param new_labels: new population labels
    """
    if new_labels is None:
        return sfs
    if sfs.pop_ids != new_labels:
        # Create a permutation of axis
        d = {x: i for i, x in enumerate(sfs.pop_ids)}
        try:
            d = [d[x] for x in new_labels]
        except:
            raise ValueError("Wrong Population labels parameter, population labels are: " + ', '.join(sfs.pop_ids))
        # Rotate axis
        sfs = np.transpose(sfs, d)
        sfs.pop_ids = new_labels
    return sfs


def _project(sfs, new_size):
    """
    Project SFS to new sample size.

    : param sfs: site frequency spectrum to change its size
    : type sfs: dadi.Spectrum (or analogue)
    : param new_size: new sample size
    : type new_size: np.ndarray
    """
    if new_size is None:
        return sfs
    if not list(new_size) == list(sfs.sample_sizes):
        try:
            sfs = sfs.project(new_size)
        except Exception as e:
            raise ValueError("Wrong projections of SFS: " + str(e))
    return sfs



def _get_default_from_snp_format(filename):
    """
    Returns population labels, the possibility of outgroup and approximation
    of size from file of dadi's SNP format.
    """
    with open(filename) as f:
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        # Read the header of file
        info = line.split()
        if (len(info) - 6) % 2 != 0:
            raise ValueError("Cannot calculate number of populations in dadi's SNP input file. Maybe it's wrong?")
        n_pop = (len(info) - 6) / 2
        pop_ids = info[3 : 3 + n_pop]
        # Find approximate size and check existence of the outgroup
        has_outgroup = True
        appr_size = np.zeros(n_pop)
        for line in f:
            info = line.split()
            if info[1][1].lower() not in ['a', 't', 'c', 'g']:
                has_outgroup = False
            for num in range(n_pop):
                cur_size = int(info[3 + num]) + int(info[4 + n_pop + num])
                if cur_size > appr_size[num]:
                    appr_size[num] = cur_size
    return pop_ids, has_outgroup, appr_size

def _change_outgroup(sfs, new_outgroup):
    """
    Change polarization of the data. If data does not have outgroup then error.
    """
    if new_outgroup is not None:
        if new_outgroup and sfs.folded:
            raise ValueError("Data does not have outgroup.")
        if not new_outgroup:
            sfs.fold()
    return sfs

def _read_data_sfs_type(module, data_holder):
    """
    Read filename of dadi's sfs format. Check dadi's manual for further information.

    : param module: dadi or moments module (or analogue)
    : param data_holder: object holding the data.
    : type  data_holder: gadma.data.DataHolder
    """
    sfs = module.Spectrum.from_file(data_holder.filename)
    ns = np.array(sfs.shape) - 1
    
    sfs = _check_missing_pop_labels(sfs, data_holder.pop_labels)
    sfs = _new_pop_labels(sfs, data_holder.pop_labels)
    sfs = _project(sfs, data_holder.sample_sizes)
    sfs = _change_outgroup(sfs, data_holder.outgroup)
    return sfs


def _read_data_snp_type(module, data_holder):
    """
    Read filename of dadi's SNP format. Check dadi's manual for further information.

    : param module: dadi or moments module (or analogue)
    : param data_holder: object holding the data.
    : type  data_holder: gadma.data.DataHolder
    """
    try:
        dd = module.Misc.make_data_dict(filename)
    except Exception as e:
        raise SyntaxError("Construction of data_dict failed: " + str(e))
    pop_labels, has_outgroup, size = _get_default_from_snp_format(data_holder.filename)
    if data_holder.sample_sizes is not None:
        size = data_holder.sample_sizes
    if data_holder.pop_labels is not None:
        pop_labels = data_holder.pop_labels
    sfs = module.Spectrum.from_data_dict(dd, pop_labels, size, has_outgroup)
    sfs = _change_outgroup(sfs, data_holder.outgroup)
    return sfs


def read_dadi_data(module, data_holder):
    """
    Read file in one of dadi's formats.

    :param module: dadi or moments module (or analogue)
    :param data_holder: object holding the data.
    :type  data_holder: gadma.data.DataHolder

    :returns: ('sfs'/'snp', data)
    :rtype: (str, :class:`dadi.Spectrum`)
    """
    _, ext = os.path.splitext(data_holder.filename)
    if ext == '.fs' or ext == '.sfs':
        return 'sfs', _read_data_sfs_type(module, data_holder)
    elif ext == '.txt':
        return 'snp', _read_data_snp_type(module, data_holder)
    else:
        # Try to guess
        try:
            return 'sfs', _read_data_sfs_type(module, data_holder)
        except:
            try:
                return 'snp', _read_data_snp_type(module, data_holder)
            except:
                raise SyntaxError("Data filename extension is neither .fs (.sfs) or .txt. Attempts to guess the file type failed.\nTo get the error message, please, change the extension.")
