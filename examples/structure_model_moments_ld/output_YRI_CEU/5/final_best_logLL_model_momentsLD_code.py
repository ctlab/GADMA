import moments.LD
import numpy as np
import pickle
import copy


def model_func(params, rho=None, theta=0.001):
    t1, nu11, s1, t2, nu21, nu22, m2_12 = params
    Nanc = 1.0 #This value can be used in splits with fraction variable
    Y = moments.LD.Numerics.steady_state(rho=rho, theta=theta)
    Y = moments.LD.LDstats(Y, num_pops=1, pop_ids=['YRI', 'CEU'])
    nu1_func = lambda t: Nanc * (nu11 / Nanc) ** (t / t1)
    Y.integrate(tf=t1, nu=lambda t: [nu1_func(t)], rho=rho, theta=theta)
    Y = Y.split(0)
    nu2_func = lambda t: ((1 - s1) * nu11) + (nu22 - ((1 - s1) * nu11)) * (t / t2)
    migs = np.array([[0, m2_12], [m2_12, 0]])
    Y.integrate(tf=t2, nu=lambda t: [nu21, nu2_func(t)], m=migs, rho=rho, theta=theta)
    return Y


bed_files = "/home/stas/git/gadma_moments/data_for_documentation/data/output_YRI_CEU_resume/bed_files/"
reg_num = 0
region_stats = {}
chromosomes = {'2': 1, '34': 1, '90': 1, '37': 1, '23': 1, '88': 1, '65': 1, '79': 1, '84': 1, '8': 1, '31': 1, '46': 1, '67': 1, '47': 1, '74': 1, '60': 1, '100': 1, '73': 1, '43': 1, '97': 1, '18': 1, '44': 1, '98': 1, '33': 1, '62': 1, '66': 1, '95': 1, '72': 1, '99': 1, '80': 1, '6': 1, '45': 1, '19': 1, '51': 1, '36': 1, '78': 1, '28': 1, '61': 1, '53': 1, '27': 1, '32': 1, '94': 1, '83': 1, '21': 1, '58': 1, '20': 1, '54': 1, '13': 1, '59': 1, '49': 1, '89': 1, '92': 1, '39': 1, '10': 1, '86': 1, '87': 1, '81': 1, '7': 1, '4': 1, '17': 1, '76': 1, '68': 1, '93': 1, '42': 1, '48': 1, '11': 1, '22': 1, '75': 1, '85': 1, '26': 1, '91': 1, '50': 1, '96': 1, '77': 1, '35': 1, '69': 1, '14': 1, '57': 1, '25': 1, '29': 1, '1': 1, '24': 1, '82': 1, '63': 1, '40': 1, '16': 1, '30': 1, '38': 1, '71': 1, '41': 1, '3': 1, '70': 1, '12': 1, '64': 1, '5': 1, '55': 1, '9': 1, '56': 1, '15': 1, '52': 1}
extension = "txt"
rec_map_name = "flat_map
kwargs = {
    "r_bins": [0.0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001],
    "report": False,
    "bp_bins": [0, 1655050, 3310100, 4965150, 6620200],
    "use_genotypes": True,
    "cM": True,
}
vcf_file = "/home/stas/git/gadma_moments/data_for_documentation/data/YRI_CEU_sim_data.vcf"
pop_map = "/home/stas/git/gadma_moments/data_for_documentation/data/samples.txt"
rec_maps = "/home/stas/git/gadma_moments/data_for_documentation/data/rec_maps"
preprocessed_data = "/home/stas/git/gadma_moments/data_for_documentation/data/preprocessed_data.bp"
if preprocessed_data is not None:
    with open(preprocessed_data, "rb") as fin:
        region_stats = pickle.load(fin)
else:
    for chrom in chromosomes:
        for num in range(1, chromosomes[chrom]):
            region_stats.update({
                f"{reg_num}":
                    moments.LD.Parsing.compute_ld_statistics(
                        vcf_file=vcf_file,
                        rec_map_file=f"/home/stas/git/gadma_moments/data_for_documentation/data/rec_maps/
                         {rec_map_name}_{chrom}.{extension}",
                        pop_file=pop_map,
                        bed_file=f"{bed_files}/bed_file_{chrom}_{num}.bed",
                        pops=['YRI', 'CEU'],
                        **kwargs
                    )
            })
            reg_num += 1
data = moments.LD.Parsing.bootstrap_data(region_stats)

p0 = [0.18154684643488384, 0.3552140446773699, 0.7544814289189229, 0.14035487862181226, 1.9894358260758085, 0.715539980127665, 1.0355376622898436]
Nanc = 7729.565462773236
r_bins = [0.0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001]
mu = 2.35e-08
rhos = 4 * Nanc * np.array(r_bins)
theta = 4 * Nanc * mu

model = model_func(p0, rho=rhos, theta=theta)
model = moments.LD.LDstats(
    [(y_l + y_r) / 2 for y_l, y_r in zip(
        model[:-2], model[1:-1])]
    + [model[-1]],
    num_pops=model.num_pops,
    pop_ids=model.pop_ids,
)
model = moments.LD.Inference.sigmaD2(model)
model_for_plot = copy.deepcopy(model)
print(f'Size of ancestral population: {Nanc}')
model = moments.LD.Inference.remove_normalized_lds(model)
means, varcovs = moments.LD.Inference.remove_normalized_data(
    data['means'],
    data['varcovs'],
    num_pops=model.num_pops,
    normalization=0)
ll_model = moments.LD.Inference.ll_over_bins(means, model, varcovs)

print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

stats_to_plot = [
    [name] for name in model.names()[:-1][0] if name != 'pi2_0_0_0_0'
]
moments.LD.Plotting.plot_ld_curves_comp(
    model_for_plot,
    data['means'][:-1],
    data['varcovs'][:-1],
    rs=np.array([0.0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001]),
    stats_to_plot=stats_to_plot,
    fig_size=(len(stats_to_plot), 7),
    cols=round(len(stats_to_plot) / 3),
    plot_means=True,
    plot_vcs=True,
)

