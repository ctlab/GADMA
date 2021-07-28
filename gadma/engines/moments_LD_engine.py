# Need some imports
import numpy as np

from . import register_engine
from . import Engine
from ..models import DemographicModel, StructureDemographicModel
from ..models import CustomDemographicModel, Epoch, Split
from .. import VCFDataHolder, moments_LD_available
from ..utils import DynamicVariable, get_correct_dtype
import moments.LD


class MomentsLD(Engine):

    id = "moments_LD"

    if moments_LD_available:
        import moments.LD as base_module
        inner_data_type = base_module.LDstats

    can_draw = True
    can_evaluate = True
    can_simulate = True

    supported_models = [DemographicModel, StructureDemographicModel,
                        CustomDemographicModel]
    supported_data = [VCFDataHolder]

    # Genotype array?
    def _read_data(cls, data_holder):
        assert isinstance(data_holder, VCFDataHolder)
        bed_file = data_holder.bed_file
        vcf_path = data_holder.filename
        recombibation_map_path = data_holder.recombination_map
        r_bins = np.logspace(-6, -3, 7)
        kwargs = data_holder.kwargs_for_computing_ld_stats
        data = moments.LD.Parsing.compute_ld_statistics(
            vcf_path, r_bins=r_bins, rec_map_file=recombibation_map_path,
            use_genotypes=True, **kwargs)

        return data

    @staticmethod
    def _get_kwargs(event, var2value):
        # Этот статический метод должен подбирать переменные, которые
        # будут в дальнейшем переданы в LDstats.Integrate()

        # Эта функция потом будет использоваться в simulate (_inner_func)
        # Для получения значений переменных

        # ИМЕНА ПЕРЕМЕННЫХ (КЛЮЧЕЙ) ВРЕМЕННЫЕ

        # RHO $ THETA
        # Значение RHO и THETA должны быть переданы в построенную модель
        # Однако они не являются обычными переменными и высчитываются иначе

        ret_dict = {'T': event.time_arg}

        # Первым делом определяется есть ли динамика в размере популяции
        if event.dyn_args is not None:
            ret_dict['Npop'] = list()
            for i in range(event.n_pop):
                dyn_arg = event.dyn_args[i]
                dyn = MomentsLD.get_value_from_var2value(var2value,
                                                         dyn_arg)
                # Так как dyn это объект класса Variable, то у него должен быть
                # атрибут name (наверно), а ещё у него скорее всего переопределён
                # оператор сравнения, поэтому легально сравнивать так

                if dyn == 'Sud':
                    ret_dict['Npop'].append(event.size_args[i])
                else:
                    ret_dict['Npop'].append(f'nu{i+1}_func')
        else:
            ret_dict['Npop'] = event.size_args

        if event.mig_args is not None:
            ret_dict['m'] = np.zeros(shape=(event.n_pop, event.n_pop),
                                     dtype=object)
            for i in range(event.n_pop):
                for j in range(event.n_pop):
                    if i == j:
                        continue
                    ret_dict['m'][i, j] = event.mig_args[i][j]

        # Тут могли бы быть gamma и h, но они не используются в moments.LD
        # Думаю, что можно также запустить падение или Warning на случай,
        # если популяции были заданы параметры sel_args, dom_args
        # Просто, чтобы не забыть
        # if event.sel_args is not None:
        #     # NEED TO CHECK THIS ARGUMENT
        #     pass
        # if event.dom_args is not None:
        #     # NEED TO CHECK THIS ARGUMENT
        #     pass

        return ret_dict

    def simulate(self, values):

        """
        Simulate expected LDstats for proposed values of variables

        :params_shamarms
        Generate function using API of moments.LD
        """

        # Первым делом оба метода _inner_func/simulate у dadi и moments
        # получают значения переменных с использованием model.var2value

        var2value = self.model.var2value(values)
        # Я пока что просто создал эту переменную, чтобы оставить всё как было в других
        # движках. Если надо будет, то переделаю

        if isinstance(self.model, CustomDemographicModel):
            # Дополнить метод потом
            return 'Here is CustomDemographicModel'

        # Определяем moments_LD как базовый модуль, но этот момент мне не до конца ясен
        # Уточнить у Кати

        moments_LD = self.base_module
        rho_m1 = 0.0  # temporary
        theta_m1 = 0.001  # temporary
        # Откуда их взять, как впихнуть

        ld = moments.LD.Numerics.steady_state(rho=rho_m1, theta=theta_m1)
        ld_stats = moments.LD.LDstats(ld, num_pops=1)
        # Нужно ли реализовать pop_ids?
        # Самый первый этап успешно пройден!
        # Y - ld_stats
        # nu - pop_size (list)
        # T - time
        # dt = 0.001 - уточнить значение этой переменной
        # theta =
        # rho =
        # m =
        # num_pops =
        # selfing =
        # frozen =

        addit_values = {}
        for ind, event in enumerate(self.model.events):
            if isinstance(event, Epoch):
                if event.dyn_args is not None:
                    for i in range(event.n_pop):
                        dyn_arg = event.dyn_args[i]
                        value = self.get_value_from_var2value(var2value,
                                                              dyn_arg)
                        if value != 'Sud':
                            func = DynamicVariable.get_func_from_value(value)
                            y1 = self.get_value_from_var2value(
                                var2value, event.init_size_args[i])
                            y2 = self.get_value_from_var2value(
                                var2value, event.size_args[i])
                            x_diff = self.get_value_from_var2value(
                                var2value, event.time_arg)
                            addit_values[f'nu{i+1}_func'] = func(
                                y1=y1,
                                y2=y2,
                                x_diff=x_diff)
                            # Тут мы создали функцию, которая описывает изменение численности
                            # популяции от начала и до конца эпохи
                kwargs_with_vars = self._get_kwargs(event, var2value)
                # dict from _get_kwargs method
                kwargs = {} # new dict for variables
                # В отличие от dadi миграции в moments передаются в виде
                # матрицы, в которой указаны значения миграции из одной популяции
                # to the other
                for x, y in kwargs_with_vars.items():
                    if x == 'm' and isinstance(y, np.ndarray):
                        # Если до этого в kwargs все переменные всё ещё были
                        # Variable, то тут они уже заменяются на непосредственно значения
                        # типа 0.6
                        # Об этом очень очевидно свидетельствует get_value_from_var2value
                        l_s = np.array(y, dtype=get_correct_dtype(y))
                        for i in range(y.shape[0]):
                            for j in range(y.shape[1]):
                                yel = y[i][j]
                                l_s[i][j] = self.get_value_from_var2value(
                                    var2value, yel)
                                l_s[i][j] = addit_values.get(l_s[i][j],
                                                             l_s[i][j])
                        kwargs[x] = l_s
                    elif x == 'Npop':
                        n_pop_list = list()
                        all_sudden = True
                        for y1 in y: # Проходим по значениям листа популяций
                            n_pop_list.append(self.get_value_from_var2value(
                                var2value, y1))
                            # Тут делается попытка перебора значений для параметра популяции
                            # Если не все популяции имеют одну и ту же численность в начале
                            # и конце эпохи, то идёт переход на лямбда функцию
                            if n_pop_list[-1] in addit_values:
                                all_sudden = False
                        if not all_sudden:
                            kwargs[x] = lambda t: [addit_values[el](t)
                                                   if el in addit_values
                                                   else el
                                                   for el in list(n_pop_list)]
                    # Тут могли бы быть gamma и h, но они не используются в moments.LD
                    # Думаю, что можно также запустить падение или Warning на случай,
                    # если популяции были заданы параметры sel_args, dom_args

                ld_stats.integrate(ld_stats, **kwargs, rho=rho_m1, theta=theta_m1)

            elif isinstance(event, Split):
                ld_stats = ld_stats.split(event.n_pop - 1)
                # Тут намного проще со сплитом, потому что метод другой
                # В обычном moments теперь тоже есть метод класса Spectrum, который
                # делает простой сплит
        return ld_stats

    # Very draft vesrsion of method, but now I know what to do!

    def evaluate(self, values, **options):
        # Так мы не наследуемся от DadiMomentsCommon, то этот метод должен быть
        # тут описан целиком и полностью
        model = self.simulate(values)
        vcf_file = 'file'
        region_stats = moments.LD.Parsing.compute_ld_statistics(vcf_file)
        mv = moments.LD.Parsing.bootstrap_data(region_stats)
        sigm = [] # cov matrix
        ll_model = self.base_module.Inference.ll_over_bins(mv, model, sigm)
        return ll_model

    def draw_ld_curves(self):
        # later
        pass

    def generate_code(self, values, filename=None, nanc=None,
                      gen_time=None, gen_time_units="years"):
        # last priority
        pass


if moments_LD_available:
    register_engine(MomentsLD)
