import re
import warnings
import os
import numpy as np
from .data import VCFDataHolder


# function for working with VCF files
def ploidy_from_vcf(path):
    """
    Calculate ploidy from .vcf file

    This function was written by Artyom Ershov as part of internship test task.
    Was taken from repository: https://github.com/iam28th/intership_test_task
    """
    # chrom name whose count of copies differs from ploidity
    bad_chroms = {'Y', 'X', 'M', 'chrX', 'chrY', 'chrM'}

    with open(path, 'r') as f:
        for line in f:
            # skip comments and heading
            if line.startswith('#'):
                continue

            # skip bad chroms
            fields = line.split('\t')
            chrom = fields[0]
            if chrom in bad_chroms:
                continue

            # .vcf contains 8 fixed columns + FORMAT + individual columns
            # i.e. first of individual columns is the 10th

            # find genotype field index
            index = fields[8].split(':').index('GT')
            # we want to divide by either | or /
            ploidy = len(re.split(r"\||/", fields[9].split(':')[index]))
            return ploidy


def read_popinfo(popinfo_file):
    """
    Returns dictionary {sample_name: pop_name} from popmap file and list of
    presented populations.
    """
    populations = []
    sample2pop = {}
    with open(popinfo_file) as fl:
        for line in fl:
            sample, pop = line.strip().split()
            if sample in sample2pop and pop != sample2pop[sample]:
                raise ValueError(f"Sample {sample} is presented in popmap "
                                 f"{popinfo_file} at least twice "
                                 "corresponding to different populations.")
            sample2pop[sample] = pop
            if pop not in populations:
                populations.append(pop)
    return sample2pop, populations


def get_list_of_names_from_vcf(vcf_file):
    """
    Returns list of sample names from vcf file.
    """
    header_line = ""
    with open(vcf_file) as fl:
        for line in fl:
            # Skip metainformation
            if line.startswith("#"):
                if line.startswith("##"):
                    continue
                header_line = line
                continue

            # Read header
            assert len(header_line) != 0, ("There is no header information in"
                                           f" VCF file {vcf_file}")
            # samples starts from 9 element
            header_info = header_line.split()
            assert len(header_info) > 9, ("There is no samples in VCF file "
                                          f" {vcf_file}: {header_line}")
            return header_info[9:]


def get_defaults_from_vcf_format(vcf_file, popmap_file, verbose=False):
    """
    Returns population labels and projections from files.
    if verbose is True then warnings are printed.
    """
    # Read popmap and check samples from vcf
    # check samples in vcf
    vcf_samples = get_list_of_names_from_vcf(vcf_file=vcf_file)

    # check popmap
    sample2pop, all_populations = read_popinfo(popinfo_file=popmap_file)
    pop_is_pres = {pop: False for pop in all_populations}
    for sample in sample2pop:
        pop_is_pres[sample2pop[sample]] = True
    populations = [pop for pop in all_populations if pop_is_pres[pop]]

    # check our lists
    # samples that are in popmap but not in vcf
    missed_samples = [smpl for smpl in sample2pop if smpl not in vcf_samples]
    if len(missed_samples) > 0 and verbose:
        warnings.warn("The following samples are presented in popmap file but "
                      f"not in VCF file: {missed_samples}")
    missed_samples = [smpl for smpl in vcf_samples if smpl not in sample2pop]
    if len(missed_samples) > 0 and verbose:
        warnings.warn("The following samples are presented in VCF file but "
                      f"not in popmap file: {missed_samples}")
    # evaluate maximum projections for our data
    pop2num = {}
    for pop in populations:
        pop2num[pop] = len([_sample for _sample, _pop in sample2pop.items()
                            if _sample in vcf_samples and _pop == pop])
    full_projections = [2 * pop2num[pop] for pop in populations]
    return populations, full_projections


def check_population_labels_vcf(pop_labels, full_pop_labels):
    """
    Checks that current labels are subset of full labels.
    """
    corr_labels = [lab in full_pop_labels for lab in pop_labels]
    assert all(corr_labels), f"Some given labels are not presented in VCF "\
                             f"file.\nGot labels: {pop_labels}\n"\
                             f"Labels in VCF file: {full_pop_labels}"


def check_projections_vcf(projections, full_projections, pop_labels):
    """
    Checks that projections are less than full projections.
    """
    corr_proj = [proj1 <= proj2 for proj1, proj2 in zip(projections,
                                                        full_projections)]
    assert all(corr_proj), "Something wrong with given projections. They are "\
                           "greater than number of samples presented in VCF "\
                           "file.\nGiven pop. labels and projections: "\
                           f"{pop_labels}, {projections}\nPop.labels "\
                           "and projections from VCF file: "\
                           f"{pop_labels}, {full_projections}."


def check_and_return_projections_and_labels(data_holder):
    """
    Takes VCF data holder and checks that its info is okay for vcf and popmap.
    Returns valid projections and population labels (could be used for the
    same data holder).
    """
    assert isinstance(data_holder, VCFDataHolder)
    # get info from popmap and vcf
    full_populations, full_projections = get_defaults_from_vcf_format(
        vcf_file=data_holder.filename,
        popmap_file=data_holder.popmap_file,
        verbose=True
    )
    # get holder info
    holder_pop_labels = data_holder.population_labels
    holder_proj = data_holder.projections
    if holder_proj is not None:
        holder_proj = list(holder_proj)
    populations = full_populations
    # check for population labels
    if holder_pop_labels is not None:
        check_population_labels_vcf(
            pop_labels=holder_pop_labels,
            full_pop_labels=full_populations
        )
        populations = holder_pop_labels
    # check for projections
    pop2proj = dict(zip(full_populations, full_projections))
    projections = [pop2proj[pop] for pop in populations]
    if holder_proj is not None:
        assert projections == holder_proj, "Data cannot be downsized for "\
                                           "diCal2 engine. Sample size of"\
                                           f" VCF data for {populations} "\
                                           "populations are "\
                                           f"{projections} and got "\
                                           f"projections {holder_proj}."
    return projections, populations


def update_data_holder_with_inner_data(data_holder, inner_data):
    """
    Updates given data_holder with given data that was read from it.
    """
    if hasattr(inner_data, "sample_sizes"):
        data_holder.projections = inner_data.sample_sizes
    if hasattr(inner_data, "pop_ids"):
        data_holder.population_labels = inner_data.pop_ids
    if hasattr(inner_data, "folded"):
        data_holder.outgroup = not inner_data.folded
    if isinstance(data_holder, VCFDataHolder):
        projections, pop_labels = check_and_return_projections_and_labels(
            data_holder
        )
        data_holder.projections = projections
        data_holder.population_labels = pop_labels
        # TODO: valid only for dical2
        # data_holder.outgroup = data_holder.reference_file is not None
    return data_holder


def _read_fsc_data(module, data_holder):
    """
    Reads file in .obs fastsimcoal2 format and returns dadi's Spectrum object.

    :param module: dadi or moments module (or analogue)
    :param data_holder: object holding the data.
    :type  data_holder: gadma.data.SFSDataHolder

    :returns: data
    :rtype: (str, :class:`dadi.Spectrum`)
    """

    mask = None

    with open(data_holder.filename, 'r') as f:
        n_observations = int(next(f).split()[0])
        if n_observations > 1:
            warnings.warn("Multiple observations are found in "
                          f"{data_holder.filename}. Calculating mean SFS")

        if 'DSFS' in data_holder.filename or 'MSFS' in data_holder.filename:
            # determine dimensionality
            # s = next(f).strip()
            # print(s)
            ndim = [int(i) + 1 for i in next(f).strip().split()[1:]]
            # ndim = [int(i) + 1 for i in s.split()]
            # print(ndim)
            total = np.zeros(ndim)
            for line in f:
                if not line.isspace():
                    # in files generated by easySFS numeric values are
                    # separeted with spaces, while in fsc examples files
                    # values are separeted with tabs... but Python's split()
                    # works with both
                    total += np.array([float(i)
                                      for i in line.split()]).reshape(ndim)
            if not data_holder.outgroup:
                # in files generated by easySFS and in example files from
                # fastsimcoal manual enries in folded SFS are masked with zeros
                mask = np.where(total == 0, True, False)

        elif 'joint' in data_holder.filename:
            # skip header & determine dimensionality for pop1
            dim1 = len(next(f).split('\t')) - 1

            # determine dimensionality for pop2
            lines = f.readlines()
            if n_observations == 1:
                dim2 = len(lines)
            else:
                for i, line in enumerate(lines[1:]):
                    if line.split()[0].split('_')[1] == '0':
                        dim2 = int(lines[i].split()[0].split('_')[1]) + 1
                        break

            total = np.zeros((dim1, dim2))
            for k in range(n_observations):
                observation = lines[k * dim2:(k+1) * dim2]
                observation = [line.strip().split('\t')[1].split() for line
                               in observation]
                for i in range(dim2):
                    for j in range(dim1):
                        observation[i][j] = float(observation[i][j])
                total += np.array(observation).T

            # construct triangular mask manually if reading unfolded SFS
            if not data_holder.outgroup:
                mask = np.arange(dim1 * dim2).reshape((dim1, dim2))
                # elements mask[i, j] where i + j >= dim1 + 2 are masked
                mask = np.where(mask // dim2 + mask % dim2 >= dim1 + 2,
                                True, False)
        else:
            # skip header & determine dimensionality
            # ndim = len(next(f).split('\t'))
            ndim = len(next(f).split())
            total = np.zeros(ndim)
            for line in f:
                if not line.isspace():
                    total += np.array([float(i) for i in line.strip().split()])

            if not data_holder.outgroup:
                mask = np.arange(ndim)
                mask = np.where(mask > ndim / 2, True, False)

    data = module.Spectrum(total / n_observations,
                           pop_ids=data_holder.population_labels,
                           data_folded=not data_holder.outgroup,
                           mask=mask)
    if data_holder.projections:
        data = data.project(data_holder.projections)

    assert data.S() > 0, "Result SFS built from FSC file is zero matrix."
    return data


def _read_data_sfs_type(module, data_holder):
    """
    Read filename of dadi's sfs format. Check dadi's manual for further
    information.

    : param module: dadi or moments module (or analogue)
    : param data_holder: object holding the data.
    : type  data_holder: gadma.data.DataHolder
    """
    sfs = module.Spectrum.from_file(data_holder.filename)

    sfs = _check_missing_population_labels(sfs, data_holder.population_labels,
                                           data_holder.filename)
    sfs = _new_population_labels(sfs, data_holder.population_labels)
    sfs = _project(sfs, data_holder.projections)
    sfs = _change_outgroup(sfs, data_holder.outgroup)
    return sfs


def _read_data_sfs_type(module, data_holder):
    """
    Read filename of dadi's sfs format. Check dadi's manual for further
    information.

    : param module: dadi or moments module (or analogue)
    : param data_holder: object holding the data.
    : type  data_holder: gadma.data.DataHolder
    """
    sfs = module.Spectrum.from_file(data_holder.filename)

    sfs = _check_missing_population_labels(sfs, data_holder.population_labels,
                                           data_holder.filename)
    sfs = _new_population_labels(sfs, data_holder.population_labels)
    sfs = _project(sfs, data_holder.projections)
    sfs = _change_outgroup(sfs, data_holder.outgroup)
    return sfs


def read_sfs_data(module, data_holder):
    """
    Reads file in one of dadi's or fastsimcoal2 formats.

    :param module: dadi or moments module (or analogue)
    :param data_holder: object holding the data.
    :type  data_holder: gadma.data.DataHolder

    :returns: data
    :rtype: (str, :class:`dadi.Spectrum`)
    """
    _, ext = os.path.splitext(data_holder.filename)
    if ext == '.fs' or ext == '.sfs':
        return _read_data_sfs_type(module, data_holder)
    elif ext == '.txt':
        return _read_data_snp_type(module, data_holder)
    elif ext == '.obs':
        # fastsimcoal2 data
        return _read_fsc_data(module, data_holder)
    else:
        # Try to guess
        try:
            return _read_data_sfs_type(module, data_holder)
        except:  # NOQA
            try:
                return _read_data_snp_type(module, data_holder)
            except:  # NOQA
                raise SyntaxError("Data filename extension is neither .fs"
                                  " (.sfs) or .txt. Attempts to guess the"
                                  " file type failed.\nTo get the error "
                                  "message, please, change the extension.")
