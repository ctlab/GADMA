from . import Engine
from ..models import DemographicModel, StructureDemographicModel,\
                     CustomDemographicModel
from ..utils import DiscreteVariable, cache_func, bcolors
from .. import SFSDataHolder, VCFDataHolder
from .. import dadi_available, moments_available
from ..code_generator import id2printfunc
from ..data.data_utils import get_defaults_from_vcf_format, ploidy_from_vcf
from ..data.data_utils import check_population_labels_vcf,\
                              check_projections_vcf

import warnings
import os
import numpy as np
from functools import wraps
from collections import defaultdict


class DadiOrMomentsEngine(Engine):
    """
    Base engine class for dadi and moments engines. It has all common methods
    of both engines.
    """

    id = 'dadi_or_moments'  #:
    base_module = None  #:
    supported_models = [DemographicModel, StructureDemographicModel,
                        CustomDemographicModel]  #:
    supported_data = [VCFDataHolder, SFSDataHolder]  #:
    inner_data_type = None  # base_module.Spectrum  #:

    can_evaluate = True
    can_draw_model = False  # dadi cannot
    can_draw_comp = True  # draw comparison between simulated AFS and real one
    can_simulate = True

    @classmethod
    def _read_data(cls, data_holder):
        """
        Reads SFS data from `data_holder`.

        Could read three types of data:

            * :py:mod:`dadi` SFS data type
            * :py:mod:`dadi` SNP data type
            * :py:mod:`fastimcoal2` data (.obs)

        Check :py:mod:`dadi` and :py:mod:`fastsimcoal2` manual
        for additional information.

        :param data_holder: holder of the data.
        :type data_holder: :class:`SFSDataHolder`
        """
        if isinstance(data_holder, SFSDataHolder):
            data = read_sfs_data(cls.base_module, data_holder)
        else:
            assert isinstance(data_holder, VCFDataHolder)
            data = read_vcf_data(cls.base_module, data_holder)
        return data

    def update_data_holder_with_inner_data(self):
        self.data_holder.projections = self.inner_data.sample_sizes
        self.data_holder.population_labels = self.inner_data.pop_ids
        self.data_holder.outgroup = not self.inner_data.folded

    @property
    def multinom(self):
        return not self.model.has_anc_size

    def _get_key(self, x, grid_sizes):
        var2value = self.model.var2value(x)
        key = tuple(var2value[var] for var in self.model.variables)
        if isinstance(grid_sizes, float):
            return (key, grid_sizes)
        return (key, tuple(grid_sizes))

    def _get_theta_from_sfs(self, values, model_sfs):
        optimal_theta = self.base_module.Inference.optimal_sfs_scaling(
            model_sfs, self.data)
        Nanc = self.get_N_ancestral_from_theta(1.0) or 1.0
        theta0 = 1 / Nanc
        if self.model.linear_constrain is None:
            return optimal_theta
        # If we have constrains we deal with them:
        upper_lb = None
        lower_ub = None
        Ax = self.model.linear_constrain._get_value(values)
        theta_lb = theta0 * self.model.linear_constrain.lb / Ax
        theta_ub = theta0 * self.model.linear_constrain.ub / Ax
        if np.any(theta_lb > theta_ub):
            inds = np.where(theta_lb > theta_ub)
            raise ValueError(f"Lower bounds for {inds} constrains in model"
                             f" are greater than upper bounds."
                             f"Please check linear constrain:\n"
                             f"{self.model.linear_constrain}\n"
                             f"and values:\n{values}.")
        theta_upper_lb = max(theta_lb)
        theta_lower_ub = min(theta_ub)
        if theta_upper_lb > theta_lower_ub:
            raise ValueError(f"Upper lower bound ({upper_lb}) in greater than"
                             f" lower upper bound ({lower_ub}). Please check "
                             f" linear constrain:\n"
                             f"{self.model.linear_constrain}\n"
                             f"and values:\n{values}.")

        optimal_theta = max(optimal_theta, theta_upper_lb)
        optimal_theta = min(optimal_theta, theta_lower_ub)
        return optimal_theta
#        for i, (lb, ub, val) in enumerate(zip(self.model.linear_constrain.lb,
#                                              self.model.linear_constrain.ub,
#                                              Ax):
#            if lb > ub:
#                raise ValueError(f"Lower bound for {i} constrain in model for"
#                                 f" values {values} is greater than upper "
#                                 f"bound: {val}, [{low_bound}, {upp_bound}]."
#                                 f"Please check linear constrain:\n"
#                                 f"{self.model.linear_constrain}\n"
#                                 f"and values:\n{values}.")
#            if lb is not None:
#                if upper_lb is None:
#                    upper_lb = lb / val
#                upper_lb = max(upper_lb, lb / val)
#            if ub is not None:
#                if lower_ub is None:
#                    lower_ub = ub / val
#                lower_ub = max(lower_ub, ub / val)
#        if upper_lb > lower_ub:
#            raise ValueError(f"Upper lower bound ({upper_lb}) in greater than"
#                             f" lower upper bound ({lower_ub}). Please check "
#                             f" linear constrain:\n"
#                             f"{self.model.linear_constrain}\n"
#                             f"and values:\n{values}.")
#        optimal_theta = max(optimal_theta, lower_ub)

    def get_theta(self, values, grid_sizes):
        key = self._get_key(values, grid_sizes)
        if key not in self.saved_add_info:
            warnings.warn("Additional evaluation for theta. Nothing to worry "
                          "if this warning is seldom.")
            self.evaluate(values, grid_sizes)
        theta = self.saved_add_info[key]
        return theta

    def get_N_ancestral_from_theta(self, theta):
        if self.model.theta0 is not None:
            theta0 = self.model.theta0
        elif (self.model.mutation_rate is not None and
                self.data_holder is not None and
                self.data_holder.sequence_length is not None):
            theta0 = 4 * self.model.mutation_rate *\
                     self.data_holder.sequence_length
        else:
            return None
        return theta / theta0

    def get_N_ancestral(self, values, grid_sizes):
        if self.model.has_anc_size:
            return super(DadiOrMomentsEngine, self).get_N_ancestral(values)
        theta = self.get_theta(values, grid_sizes)
        return self.get_N_ancestral_from_theta(theta)

    def draw_data_comp_plot(self, values, grid_sizes, save_file=None,
                            vmin=None):
        """
        Draws plots of SFS data for observed data and simulated by model data.

        :param values: Values of the model parameters, it could be list of
                       values or dictionary {variable name: value}.
        :type values: list or dict
        :param grid_sizes: special parameter for numerical solutions. It is
                           `pts` for :class:`DadiEngine` and
                           `dt_fac` for :class:`MomentsEngine`.
        :param save_file: File to save picture. If None then picture will be
                          displayed to the screen.
        :type save_file: str
        :param vmin: Values in sfs below vmin are masked in plot. Should be
                     positive.
        :type vmin: float
        """
        from matplotlib import pyplot as plt
        plt.clf()
        plt.close()
        if vmin is not None:
            assert vmin > 0
        show = save_file is None
        n_pop = len(self.data.sample_sizes)
        # Get simulated data
        model = self.simulate(
            values,
            self.data.sample_sizes,
            None,
            None,
            grid_sizes
        )
        # check for the multinom
        if self.multinom:
            model = self.base_module.Inference.optimally_scaled_sfs(
                model=model,
                data=self.data
            )
        # Draw
        if n_pop == 1:
            self.base_module.Plotting.plot_1d_comp_Poisson(model, self.data)
            if show:
                plt.show()
        else:
            func_name = f"plot_{n_pop}d_comp_Poisson"
            plotting_func = getattr(self.base_module.Plotting, func_name)
            plotting_func(model, self.data, vmin=vmin, show=show)
        if not show:
            plt.savefig(save_file)
            plt.close('all')

    def evaluate(self, values, grid_sizes):
        """
        Simulates SFS from values and evaluate log likelihood between
        simulated SFS and observed SFS.
        """
        if self.data is None or self.model is None:
            raise ValueError("Please set data and model for the engine or"
                             " use set_and_evaluate function instead.")
        values_gen = self.model.translate_values(units="genetic",
                                                 values=values,
                                                 time_in_generations=False)
        model_sfs = self.simulate(values_gen,
                                  self.data.sample_sizes,
                                  None,
                                  None,
                                  grid_sizes)
        # TODO: process it
        if not self.multinom and self.model.linear_constrain is not None:
            raise ValueError(f"{self.id} engine could not process constrains "
                             "on demographic model parameters (bounds of time "
                             "splits) in not-multinom mode.")
        if not self.multinom:
            theta0_inv = self.get_N_ancestral_from_theta(1)
            if theta0_inv is None:
                warnings.warn("Theta0 is not set and translation of Nanc "
                              "variable with theta0=1 could be wrong.")
                theta0_inv = 1.0
            theta0 = 1 / theta0_inv
            # just got value of Nanc from values
            theta = theta0 * self.get_N_ancestral(values, grid_sizes)
        else:
            # The next two lines usually works like ll_multinom, but when we
            # have some constrains it could turn out to be ll with some other
            # theta.
            theta = self._get_theta_from_sfs(values_gen, model_sfs)
        ll_model = self.base_module.Inference.ll(theta * model_sfs, self.data)
        # Save simulated data
        key = self._get_key(values, grid_sizes)
        self.saved_add_info[key] = theta
        return ll_model

    def get_claic_component(self, x0, all_boots, grid_sizes, eps):
        if dadi_available:
            from dadi import Godambe
        elif moments_available:
            from moments import Godambe
        else:
            ImportError("For CLAIC evaluation either dadi or moments is"
                        " required.")
        # Cache evaluations of the frequency spectrum inside our hessian/J
        # evaluation function
        var2val = self.model.var2value(x0)
        is_not_discrete = np.array([not isinstance(var, DiscreteVariable)
                                    for var in var2val])
        if len(x0) > 0 and len(var2val) > 0:
            x0 = np.array(list(var2val.values()), dtype=object)
            p0 = x0[is_not_discrete].astype(float)
        else:
            p0 = x0.astype(float)

        @wraps(self.simulate)
        def simul_func(x):
            p = np.array(x0)
            if len(p) > 0 and len(var2val) > 0:
                p[is_not_discrete] = x
            else:
                p = x
            return self.simulate(
                p,
                self.data.sample_sizes,
                None,
                None,
                grid_sizes
            )

        cached_simul = cache_func(simul_func)

        def func(x, data):
            model = cached_simul(x)
            return self.base_module.Inference.ll_multinom(model, self.data)
        H = - Godambe.get_hess(func, p0, eps, args=[self.data])
        H_inv = np.linalg.inv(H)

        J = np.zeros((len(p0), len(p0)))
        for ii, boot in enumerate(all_boots):
            boot = self.base_module.Spectrum(boot)
            grad_temp = Godambe.get_grad(func, p0, eps, args=[boot])
            J_temp = np.outer(grad_temp, grad_temp)
            J += J_temp

        J = J/len(all_boots)

        # G = J*H^-1
        G = np.dot(J, H_inv)

        return np.trace(G)

    def generate_code(self, values, filename, grid_sizes, nanc=None,
                      gen_time=None, gen_time_units="years"):
        """
        Prints nice formated code in the format of engine to file or returns
        it as string if no file is set.

        :param values: values for the engine's model.
        :param filename: file to print code. If None then the string will
                         be returned.
        """
        if self.data_holder is None:
            raise AttributeError("Engine was initialized with inner "
                                 "data. Need gadma.DataHolder for "
                                 "generation of code.")
        if filename is not None and not filename.endswith("py"):
            filename = filename + ".py"
        return id2printfunc[self.id](self, values,
                                     grid_sizes, filename, nanc, gen_time,
                                     gen_time_units)


# Those functions are common for dadi and moments engines.
def _check_missing_population_labels(sfs, default_pop_labels=None,
                                     filename=None):
    """
    Check that SFS has population labels. If not then make them default.

    : param sfs: site frequency spectrum to check
    : type sfs: dadi.Spectrum (or analogue)
    : param default_population_labels: if pop. labels are missing use
                                       this values.
    """
    if sfs.pop_ids is None:
        if default_pop_labels is not None:
            sfs.pop_ids = default_pop_labels
        else:
            sfs.pop_ids = ['Pop %d' % (i+1) for i in range(sfs.ndim)]
        warnings.warn("Spectrum file %s is in an old format - without"
                      " population labels, so they will be taken from the"
                      " corresponding parameter: %s."
                      % (filename, ', '.join(sfs.pop_ids)))
    return sfs


def _new_population_labels(sfs, new_labels):
    """
    Assign new order of population labels of SFS.

    : param sfs: site frequency spectrum to change its labels
    : type sfs: dadi.Spectrum (or analogue)
    : param new_labels: new population labels
    """
    if new_labels is None:
        return sfs
    if len(sfs.pop_ids) > len(new_labels):
        marginalize_over = []
        for i, label in enumerate(sfs.pop_ids):
            if label not in new_labels:
                marginalize_over.append(i)
        sfs = sfs.marginalize(marginalize_over)
    if sfs.pop_ids != new_labels:
        # Create a permutation of axis
        d = {x: i for i, x in enumerate(sfs.pop_ids)}
        try:
            d = [d[x] for x in new_labels]
        except:  # NOQA
            raise ValueError("Wrong Population labels parameter, population"
                             " labels are: " + ', '.join(sfs.pop_ids))
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
            sfs = sfs.project(np.array(new_size))
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
            raise ValueError("Cannot calculate number of populations in"
                             " dadi's SNP input file. Maybe it's wrong?")
        n_pop = int((len(info) - 6) / 2)
        pop_ids = info[3: 3 + n_pop]
        # Find approximate size and check existence of the outgroup
        has_outgroup = True
        appr_size = np.zeros(n_pop, dtype=int)
        for line in f:
            info = line.split()
            nucleotides = ['a', 't', 'c', 'g']
            assert len(info[0]) % 2 == 1
            assert len(info[1]) % 2 == 1
            info_0_char = info[0][len(info[0]) // 2].lower()
            info_1_char = info[1][len(info[1]) // 2].lower()
            if (info_0_char not in nucleotides or
                    info_1_char not in nucleotides):
                has_outgroup = False
            for num in range(n_pop):
                cur_size = int(info[3 + num]) + int(info[4 + n_pop + num])
                if cur_size > appr_size[num]:
                    appr_size[num] = cur_size
    return pop_ids, has_outgroup, appr_size


def _change_outgroup(sfs, new_outgroup):
    """
    Change polarization of the data. If data does not have outgroup
    then error.
    """
    if new_outgroup is not None:
        if new_outgroup and sfs.folded:
            raise ValueError("Data does not have outgroup.")
        if not new_outgroup and not sfs.folded:
            sfs = sfs.fold()
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


def _read_data_snp_type(module, data_holder):
    """
    Read filename of dadi's SNP format. Check dadi's manual for further
    information.

    : param module: dadi or moments module (or analogue)
    : param data_holder: object holding the data.
    : type  data_holder: gadma.data.DataHolder
    """
    try:
        dd = module.Misc.make_data_dict(data_holder.filename)
    except Exception as e:
        raise SyntaxError("Construction of data_dict failed: " + str(e))
    population_labels, has_outgroup, size = _get_default_from_snp_format(
        data_holder.filename)
    if data_holder.population_labels is not None:
        if len(data_holder.population_labels) < len(population_labels):
            for label in data_holder.population_labels:
                assert label in population_labels
        pop2size = {pop: siz
                    for pop, siz in zip(population_labels, size)}
        size = [pop2size[x] for x in data_holder.population_labels]
        population_labels = data_holder.population_labels
    if data_holder.projections is not None:
        size = data_holder.projections
    sfs = module.Spectrum.from_data_dict(dd, population_labels,
                                         projections=size,
                                         polarized=has_outgroup)
    if has_outgroup != data_holder.outgroup:
        sfs = _change_outgroup(sfs, data_holder.outgroup)
    return sfs


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
            ndim = [int(i) + 1 for i in next(f).strip().split('\t')[1].split()]
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
            ndim = len(next(f).split('\t'))
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


def read_vcf_data(module, data_holder):
    """
    Reads file in vcf format and returns dadi's Spectrum object.

    :param module: dadi or moments module (or analogue)
    :param data_holder: object holding the data.
    :type  data_holder: gadma.data.VCFDataHolder

    :returns: data
    :rtype: (str, :class:`dadi.Spectrum`)
    """
    # Check that we have diploid VCF file
    ploidy = ploidy_from_vcf(data_holder.filename)
    assert ploidy == 2, "Only diploid VCF files could be read. "\
                        f"Ploidy of given file is: {ploidy}"
    # get our info about maximal everything
    populations, full_projections = get_defaults_from_vcf_format(
        vcf_file=data_holder.filename,
        popmap_file=data_holder.popmap_file,
        verbose=True
    )
    # check our lines
    filtered_out_lines = []
    lines_with_no_nucl = []
    repeated_lines = defaultdict(list)
    total_snp = 0
    positions2line_num = {}
    has_anc_allele = True
    with open(data_holder.filename) as f:
        for line_number, line in enumerate(f):
            if line.startswith("#"):
                continue
            total_snp += 1
            splitline = line.split()
            # check filter
            if splitline[6].upper() not in [".", "PASS"]:
                filtered_out_lines.append(line_number)
                continue
            # check nucleotides
            nucleotides = ["A", "T", "G", "C"]
            if (splitline[3].upper() not in nucleotides or
                    splitline[4].upper() not in nucleotides):
                lines_with_no_nucl.append(line_number)
            pos = (splitline[0], splitline[1])
            if pos in positions2line_num:
                repeated_lines[positions2line_num[pos]].append(line_number)
            else:
                positions2line_num[pos] = line_number
            # check outgroup info in INFO column
            got_AA = False
            for info in splitline[7].split(";"):
                if info.strip().startswith("AA="):
                    got_AA = True
                    break
            if not got_AA:
                has_anc_allele = False
    if len(filtered_out_lines) > 0:
        if len(filtered_out_lines) == total_snp:
            raise ValueError("All lines in VCF file will be filtered out "
                             "during reading. Please set FILTER column to "
                             "`.` or `PASS` value.")
        lines = ""
        if len(filtered_out_lines) < 10:
            lines = f" Numbers of lines: {filtered_out_lines}"
        warnings.warn(f"Several lines ({len(filtered_out_lines)} / {total_snp}"
                      ") of VCF file will be ignored as FILTER "
                      f"column is not equal to `.` or `PASS`.{lines}")
    if len(lines_with_no_nucl) > 0:
        if len(lines_with_no_nucl) + len(filtered_out_lines) == total_snp:
            raise ValueError("Reference and alternative alleles in VCF file "
                             "are not nucleotides.")
        lines = ""
        if len(lines_with_no_nucl) < 10:
            lines = f" Numbers of lines: {lines_with_no_nucl}"
        warnings.warn(f"Several lines ({len(lines_with_no_nucl)} / {total_snp}"
                      ") has not-nucleotide reference or alternative allele. "
                      f"Those variants will be ignored during reading.{lines}")
    missed_repeats = 0
    if len(repeated_lines) > 0:
        for line_number in repeated_lines:
            missed_repeats += len(repeated_lines[line_number])
        warnings.warn(f"{bcolors.FAIL}BE CAREFUL several positions are "
                      "repeated in VCF file.For each repeat the last "
                      "information from file will be taken. Number of lost "
                      f"lines: {missed_repeats}{bcolors.ENDC}")
    # number of snp's that will be read
    snp_num = total_snp
    snp_num -= len(filtered_out_lines)
    snp_num -= len(lines_with_no_nucl)
    snp_num -= missed_repeats

    # Check outgroup
    outgroup = data_holder.outgroup
    if data_holder.outgroup is None:
        outgroup = has_anc_allele
    if data_holder.outgroup is True and not has_anc_allele:
        raise ValueError("Information about ancestral allele was missed "
                         "somewhere in VCF file. Check that INFO column "
                         "contains AA field. It is not possible to create "
                         "SFS with outgroup without this information.")
    # Build sfs
    dd = module.Misc.make_data_dict_vcf(
        vcf_filename=data_holder.filename,
        popinfo_filename=data_holder.popmap_file,
        filter=True,
        flanking_info=[None, None],
    )
    if len(dd) == 0:
        raise ValueError("None SNP's were read from the VCF file "
                         f"{data_holder.filename}. Please check the VCF "
                         "format of the file. E.g. variants should be from "
                         "list of nucleotides (A, T, C, G) or lines should be "
                         "marked as PASS or as '.' in FILTER column.")
    assert len(dd) == snp_num, "Something went wrong and lower number of "\
                               f"SNP's were read ({len(dd)} / {snp_num})."
    # populations labels
    population_labels = data_holder.population_labels
    if data_holder.population_labels is None:
        population_labels = list(populations)
    # Check that we have correct pop labels
    check_population_labels_vcf(
        pop_labels=population_labels,
        full_pop_labels=populations
    )
    # and get our projections
    projections = data_holder.projections
    pop2proj = dict(zip(populations, full_projections))
    projected_full_proj = [pop2proj[pop] for pop in population_labels]
    if data_holder.projections is None:
        projections = projected_full_proj
    # check projections are less than in vcf file
    check_projections_vcf(
        projections=projections,
        full_projections=projected_full_proj,
        pop_labels=population_labels
    )
    data = module.Spectrum.from_data_dict(
        data_dict=dd,
        pop_ids=population_labels,
        projections=projections,
        polarized=outgroup,
    )
    assert data.S() > 0, "Result SFS built from VCF file is zero matrix."
    return data
