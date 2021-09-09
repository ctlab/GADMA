import moments.LD
import numpy as np

def model_func(params):
	s1, t1, nu11, nu12, m1_12 = params
	Y = Numerics.steady_state(rho=rho, theta=theta)
	Y = LDstats(Y, num_pops=1, pop_ids=pop_ids))
	Y = Y.split(0)
	migs = np.array([[0, m1_12], [m1_12, 0]])
	Y.integrate(tf=t1, nu=[nu11,nu12], m=migs
	return Y

bed_files = /home/stas/git/gadma_moments/gadma_test_launch/8_sim_data_more_computing/output/bed_files/
reg_num = 0
region_stats = {}
chromosomes = {'1': 16}
extension = txt
rec_map = rec_map
kwargs = {
	r_bins: [0.e+00 1.e-06 2.e-06 5.e-06 1.e-05 2.e-05 5.e-05 1.e-04 2.e-04 5.e-04
 1.e-03],
	report: True,
	bp_bins: [      0 1655050 3310100 4965150 6620200],
	use_genotypes: True,
	cM: True,
}
vcf_file = /home/stas/git/gadma_moments/gadma_test_launch/8_sim_data_more_computing/data_sim.vcf
pop_map = /home/stas/git/gadma_moments/gadma_test_launch/8_sim_data_more_computing/pop_map.txt
rec_maps = /home/stas/git/gadma_moments/gadma_test_launch/8_sim_data_more_computing/rec_maps
for chrom in chromosomes:
    for num in range(1, chromosomes[chrom]):
        region_stats.update({
            f"{reg_num}":
            moments.LD.Parsing.compute_ld_statistics(
                vcf_file,
                rec_map_file=f"rec_maps                         /{rec_map}_{chrom}.{extension}"
                pop_file=pop_map
                bed_file=f"{bed_files}/bed_file_{chrom}_{num}.bed"
                pops=['deme0', 'deme1']
                **kwargs
            )
        })
        reg_num += 1
data = moments.LD.Parsing.bootstrap_data(region_stats)

p0 = [1.0, 0.002719121151619303, 0.2320458397708681, 7.8550975183573994, 3.484955385514677, 0.0]
model = model_func(p0)
Nanc = 4832.461404255972
theta = 4 * Nanc * mu

print('Size of ancestral population: {Nanc}'.format(Nanc))
model = moments.LD.Inference.remove_normalized_lds(model)
means, varcovs = moments.LD.Inference.remove_normalized_data(
    data['means'],
    data['varcovs'],
    num_pops=model.num_pops,
    normalization=0)
ll_model = 'moments.LD.Inference.ll_over_bins(means, model, varcovs)

print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

stats_to_plot = [
    [name] for name in model.names()[:-1][0] if name != 'pi2_0_0_0_0'
]
moments.LD..Plotting.plot_ld_curves_comp(
    model,
    data['means'][:-1],
    data['varcovs'][:-1],
    rs=[0.e+00 1.e-06 2.e-06 5.e-06 1.e-05 2.e-05 5.e-05 1.e-04 2.e-04 5.e-04
 1.e-03],
    stats_to_plot=stats_to_plot,
    fig_size=(len(stats_to_plot), 9),
    cols=round(len(stats_to_plot) / 3),
    plot_means=True,
    plot_vcs=True,
)

