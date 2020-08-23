import moments
from demographic_model import model_func

n_pop = 2
pop_labels = ["YRI", "CEU"]

par_labels = ['nu1F', 'nu2B', 'nu2F', 'm', 'Tp', 'T']
popt = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]

lower_bound = [1e-2, 1e-2, 1e-2, 0, 0, 0]
upper_bound = [100, 100, 100, 10, 3, 3]

mu = 2.35e-8  # mutation rate
L = 4.04e6  # effective length of sequence

ns = [20, 20]

# Get maximum log-likelihood
model = model_func(popt, ns)
data = moments.Spectrum.from_file("fs_data.fs")
max_ll = moments.Inference.ll_multinom(model, data)

# Get ancestral population size
theta =  moments.Inference.optimal_sfs_scaling(model, data) # mutation flux
Nanc = int(theta / (4 * mu * L))

if __name__ == "__main__":
    print('Maximum log composite likelihood: {0}'.format(max_ll))

    print('Optimal value of theta: {0}'.format(theta))

    print('Size of the ancestral population: {0}'.format(Nanc))

    # Draw model
    model = moments.ModelPlot.generate_model(model_func, popt,
                                             [4 for _ in range(n_pop)])
    moments.ModelPlot.plot_model(model,
        save_file='model_plot.png',
        fig_title=f'Demographic model for 2_YRI_CEU_6_Gut, Nanc: {Nanc}',
        pop_labels=pop_labels,
        nref=Nanc,
        draw_scale=True,
        draw_ancestors=True,
        gen_time=1.0,
        gen_time_units='Generations',
        reverse_timeline=True)
    print('Model plot is saved to model_plot.png')
