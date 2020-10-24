import moments
from demographic_model import model_func

n_pop = 3
pop_labels = ["YRI", "CEU", "CHB"]

par_labels = ['nuAf', 'nuB', 'nuEu0', 'nuEu', 'nuAs0', 'nuAs',
              'mAfB', 'mAfEu', 'mAfAs', 'mEuAs', 'TAf', 'TB', 'TEuAs']
popt = [1.68, 0.287, 0.129, 3.74, 0.070, 7.29,
        3.65, 0.44, 0.28, 1.40, 0.211, 0.338, 0.058]

lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2,
               0, 0, 0, 0, 1e-15, 1e-15, 1e-15]
upper_bound = [100, 100, 100, 100, 100, 100, 10, 10, 10, 10, 5, 5, 5]

mu = 2.35e-8  # mutation rate
L = 4.04e6  # effective length of sequence

ns = [20, 20, 20]

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
        fig_title=f'Demographic model for 3_YRI_CEU_CHB_13_Gut',
        pop_labels=pop_labels,
        nref=Nanc,
        draw_scale=True,
        draw_ancestors=True,
        gen_time=1.0,
        gen_time_units='Generations',
        reverse_timeline=True)
    print('Model plot is saved to model_plot.png')
