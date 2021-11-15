from math import log
import os
import shutil
import uuid
import subprocess

import dadi
from . import register_engine, Engine
# TODO: implement check for fastsimcoal2 availability
from .. import SFSDataHolder, VCFDataHolder, dadi_available
from ..utils.variables import TimeVariable, PopulationSizeVariable, \
        DynamicVariable, MigrationVariable
from ..models import Epoch, Split
from ..models import EpochDemographicModel
from ..data.data_utils import read_sfs_data  # , read_vcf_data


class Fsc2Engine(Engine):
    """
    Engine for using :py:mod:`fastsimcoal2` for demographic inference.

    Citation of :py:mod:`fastsimcoal2`:

    Excoffier, L., Dupanloup, I., Huerta-Sánchez, E., Sousa, V.C.,
    and M. Foll (2013) Robust demographic inference from genomic
    and SNP data. PLOS Genetics, 9(10):e1003905.
    """

    id = 'fsc2'
    # if fsc2_available:
    #     do_something ?
    supported_models = [EpochDemographicModel]
    supported_data = [VCFDataHolder, SFSDataHolder]
    if dadi_available:
        import dadi as base_module
        innter_data_type = base_module.Spectrum

    def __init__(self, data=None, model=None):
        super().__init__(data=data, model=model)
        self.temp_dir_path = os.path.join(os.getcwd(), 'gadma_fsc_temp_files')
        if not os.path.exists(self.temp_dir_path):
            os.mkdir(self.temp_dir_path)

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

    @staticmethod
    def generate_tpl(model, ns, var2value=None, values=None, datatype='DNA'):
        """
        Writes .tpl and .est files for fastsimcoal2 launch

        :param model: generate for this model
        """
        # fastsimcoal2 doesn't support linear growth
        for var in model.variables:
            if isinstance(var, DynamicVariable):
                if values:
                    assert values[var.name] != 'Lin', "fastsimcoal2 doesn't \
                            support linear population growth"

        if model.has_anc_size:
            Nanc = model.get_Nanc_size()
        else:
            Nanc = 1000

        epochs = [ev for ev in model.events if isinstance(ev, Epoch)]
        splits = [ev for ev in model.events if isinstance(ev, Split)]

        # rescale variable to physical (time in generations,
        # population size in number of haploid individuals
        phys_values = model.translate_values(
                units='physical',
                values=values,
                Nanc=Nanc,
                time_in_generations=True,
                rescale_back=False,
        )
        var2value_phys = model.var2value(phys_values)
        for k, v in var2value_phys.items():
            if isinstance(k, TimeVariable) or \
                    isinstance(k, PopulationSizeVariable):
                var2value_phys[k] = round(v)

        # construct .tpl file line by line
        lines = []

        # number of populations
        lines.append('//Parameters for the coalescence simulation program')
        lines.append(f'{model.number_of_populations()} samples to simulate :')

        # effective population sizes
        lines.append('//Population effective sizes (number of genes)')
        lines.extend([str(var2value_phys[n]) for n \
            in model.events[-1].size_args])

        # sample sizes
        lines.append('//Sample sizes')
        lines.extend([str(n) for n in ns])

        # growth rates (initial)
        lines.append(('//Growth rates: negative growth implies '
                      'population expansion'))
        ev = model.events[-1]
        if ev.dyn_args:
            # need to calculate growth rate exponentiation coefficient r
            # such that Nt = N0 * exp(rt) for each of exponential dyn_args
            for dyn, init, fin in zip(ev.dyn_args,
                    ev.init_size_args, ev.size_args):
                print(dyn)
        else:
            lines.extend(['0'] * model.number_of_populations())

        # number of migration matrix
        lines.append(("//Number of migration matrices : 0 implies no "
                      "migration between demes "))

        if model.number_of_populations() == 1:
            lines.append("0")
        else:
            has_migration = False
            for i, ep in enumerate(epochs):
                if ep.mig_args is not None:
                    has_migration = True
                    break

            if has_migration:
                # write a matrix for each epoch;
                # if no migration occured, write zero matrix
                lines.append(len(epochs))
                dim = model.number_of_populations()

                # write matrices from present day, backwards in time
                for ep in epochs[::-1]:
                    lines.append('//migration matrix')
                    if ep.mig_args is None:
                        row = ' '.join(['0'] * dim)
                        matrix = '\n'.join([row] * dim)
                        lines.append(matrix)
                    else:
                        elements = ep.mig_args.reshape(-1).tolist()
                        for i in range(dim * dim):
                            if isinstance(elements[i], MigrationVariable):
                                elements[i] = var2value_phys[elements[i]]

                        matrix = ''
                        for i in range(dim):
                            matrix += ' '.join(map(str,
                                elements[i*dim:(i+1)*dim]))
                            if i != dim - 1:
                                matrix += '\n'
                        lines.append(matrix)
            else:
                lines.append(0)  # 0 implies no migration

        # historical events - from present day backwards in time

        lines.append(('//historical event: time, source, sink, migrants, '
                      'new size, growth rate, migr. matrix'))
        he_lines = []

        def add_epoch_event(pop_id, ev):
            # if Dynamic is exponential, fastsimcoal2 historical event
            # occurs and the end of Epoch and is represented
            # by growth rate change.
            #
            # if Dynamic is sudden, event occurs at the start of the epoch
            # and is represented by 'new size' field
            init = ev.init_size_args[pop_id]
            fin = ev.size_args[pop_id]
            epoch_length = ev.time_arg
            dyn = ev.dyn_args[pop_id] if ev.dyn_args is not None else None

            if isinstance(dyn, DynamicVariable):
                dyn = var2value_phys[dyn]
            if isinstance(fin, PopulationSizeVariable):
                fin = var2value_phys[fin]
            else:
                fin *= Nanc
            if isinstance(init, PopulationSizeVariable):
                init = var2value_phys[init]
            else:
                init *= Nanc
            if isinstance(epoch_length, TimeVariable):
                epoch_length = var2value_phys[epoch_length]
            else:
                pass  # might need to add conversion from genetic units
                      # for constants?

            if dyn == 'Exp':
                r = log(init / fin) / epoch_length
                he_lines.append((f'{time} {pop_id} {pop_id} {0} {1} {r} '
                                 f'{migration_matrix}'))
            elif (dyn == 'Sud' or dyn is None) and init != fin:
                he_lines.append((f'{time + epoch_length} {pop_id} {pop_id} '
                                 f'{0} {init / fin} {0} {migration_matrix}'))

        # for first (closest to the present) event
        # starting values are used
        time = var2value_phys[model.events[-1].time_arg]
        migration_matrix = 0

        for event_index, ev in enumerate(model.events[::-1][1:]):
            if isinstance(ev, Epoch):
                migration_matrix += 1
                one_function = True
                for i in range(len(ev.size_args)):
                    add_epoch_event(i, ev)
                time += var2value_phys[ev.time_arg]
            elif isinstance(ev, Split):
                sink = ev.pop_to_div
                # new population is always last at the time of its creation
                source = ev.n_pop
                # 100% of "new" population is merged back
                migrants = 1
                # size of the merged population remains the same ?
                new_deme_size = 1
                # growth rate of the sink - might need to recalculate
                # by looking on the next event
                growth_rate = 0

                # use migration matrix from the next epoch
                line = (f"{time} {source} {sink} {migrants} {new_deme_size} "
                        f"{0} {migration_matrix + 1}")
                he_lines.append(line)

        lines.append(f'{len(he_lines)} historical events')
        lines.extend(he_lines)

        # generic settings
        lines.append('//Number of independent loci [chromosome]')
        lines.append('1 0')

        lines.append(('//Per chromosome: Number of contiguous linkage Block: '
                      'a block is a set of contiguous loci'))
        lines.append(1)

        lines.append(('//per Block:data type, number of loci, per gen recomb '
                      'and mut rates'))
        rec_rate = model.recombination_rate
        if rec_rate is None:
            rec_rate = 0
        mut_rate = model.mutation_rate
        if mut_rate is None:
            mut_rate = 3e-8
        nloci = 10000
        if datatype == 'DNA':
            lines.append(f'DNA {nloci} {rec_rate} {mut_rate} 0.33')
        elif datatype == 'FREQ':
            lines.append(f'FREQ {nloci} {rec_rate} {mut_rate}')

        lines = [str(line) for line in lines]
        return '\n'.join(lines)

    @staticmethod
    def generate_est(variables):
        """
        Returns a string with fastsimcoal2 .est file
        """
        lines = []
        lines.append('// Priors and rules file')
        lines.append('// *********************\n')

        lines.append('[PARAMETERS]')
        lines.append('\t'.join(['//#isInt?', '#name',
                                '#dist.', '#min', '#max']))
        lines.append('//all N are in number of haploid individuals')

        if len(variables) == 0:
            lines.append('1 NCHAIRS unif 12 12 output')

        return '\n'.join(lines) + '\n'


    def simulate(self, event, ns, var2value=None, values=None, n=100):

        # parameter definition file is empty at the moment,
        # because all the values were directly put into template file...
        # however fsc2702 doesn't run with empty definition file
        tpl = self.generate_tpl(event, ns, var2value, values)

        tpl_filename = 'temp_tpl.tpl'
        # tpl_filename = 'temp_tpl_' + str(uuid.uuid4()) + '.tpl'
        with open(tpl_filename, 'w') as tplfile:
            tplfile.write(tpl)

        def_filename = 'temp_def.def'  # simpler filename for debugging...
        # def_filename = 'temp_def_' + str(uuid.uuid4()) + '.def'
        with open(def_filename, 'w') as dfile:
            dfile.write('NCHAIRS\n12')

        results_folder = tpl_filename[:-4]

        def clean_up():
            os.remove(tpl_filename)
            os.remove(def_filename)
            if os.path.exists(results_folder):
                shutil.rmtree(results_folder)

        derived = True
        try:
            command = (f"fsc2702 --tplfile {tpl_filename} -f {def_filename} "
                       f"-n {n} -s 2000 -{'d' if derived else 'm'} -u")
            proc = subprocess.run(command, shell=True, capture_output=True)
            proc.check_returncode()
        except Exception as e:
            clean_up()
            raise e

        extension = f"_{'D' if derived else 'M'}SFS.obs"
        sfs_file = os.path.join(results_folder, results_folder + extension)

        dh = SFSDataHolder(sfs_file,
                outgroup = derived
                )
        sfs = read_sfs_data(dadi, dh)
        clean_up()
        return sfs

    def evaluate(self, values, ns, obs_path, n=1000, L=20,):
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
        tpl = self.generate_tpl(self.model, ns, values=values, datatype='FREQ')
        tpl_filename = 'temp_tpl.tpl'
        with open(tpl_filename, 'w') as tpl_file:
            tpl_file.write(tpl)

        est = self.generate_est([])
        est_filename = 'temp_est.est'
        # with open(est_filename, 'w', newline='') as est_file:
        with open(est_filename, 'w') as est_file:
            est_file.write(est)

        # fastsimcoal2 needs .obs file to have the same name as
        # the prefix of .tpl file (+extension)
        obs_filename = tpl_filename.split('.')[0] + \
            obs_path[obs_path.rfind('_'):]
        shutil.copy(obs_path, obs_filename)

        derived = True  # TODO: add as a function parameter/get from data
        command = (f"fsc2702 -t {tpl_filename} -n {n} "
                   f"-{'d' if derived else 'm'} -e {est_filename} -M -L {1} "
                   f"-q -u")
        # print(command)
        # print('*' * 10)
        # print(est)
        # print('*' * 10)
        # with open('manual.est', 'r') as f:
        #     manual_lines = f.readlines()
        # generated_lines = est
        # for m, g in zip(manual_lines, generated_lines):
        #     if m != g:
        #         print('lengths:', len(m), len(g))
        #         print(m)
        #         print(g)
        #         for chm, chg in zip(m, g):
        #             flag = 0 if chm == chg else 1
        #             print(chm, chg, flag)
        try:
            proc = subprocess.run(command, shell=True, capture_output=True)
            proc.check_returncode()

        except Exception as e:
            print(proc.stdout.decode(encoding='utf8'))
            print(proc.stderr.decode(encoding='utf8'))
            raise e

        results_folder = tpl_filename[:tpl_filename.rfind('.')]
        results_path = os.path.join(results_folder, results_folder + \
                '.bestlhoods')
        print(results_path)
        with open(results_path, 'r') as r:
            lhood = r.readlines()[1].split()[-1]

        print(lhood)
        return float(lhood)



# TODO: add condition to register only if available
register_engine(Fsc2Engine)
