import re
import warnings

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


def get_defaults_from_vcf_format(vcf_file, popmap_file, verbose=False):
    """
    Returns population labels and projections from files.
    if verbose is True then warnings are printed.
    """
    # Read popmap and check samples from vcf
    # check samples in vcf
    vcf_samples = []
    with open(vcf_file) as f:
        for line in f:
            if not line.startswith("##") and line.startswith("#"):
                vcf_samples = line.strip().split()[9:]
                break
    assert len(vcf_samples) > 0, f"VCF file {vcf_file} has bad "\
                                 "header."
    # check popmap
    populations = []
    sample2pop = {}
    with open(popmap_file) as f:
        for line in f:
            sample, pop = line.strip().split()
            if sample in sample2pop and pop != sample2pop[sample]:
                raise ValueError(f"Sample {sample} is presented in popmap "
                                 f"{popmap_file} at least twice "
                                 "corresponding to different populations.")
            sample2pop[sample] = pop
            if pop not in populations:
                if sample in vcf_samples:
                    populations.append(pop)
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
