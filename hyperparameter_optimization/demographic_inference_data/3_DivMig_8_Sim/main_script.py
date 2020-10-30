import moments
from demographic_model import model_func
from matplotlib import pyplot as plt

n_pop = 3
pop_labels = ['1 Pop', '2 Pop', '3 Pop']

par_labels = ['nu1', 'nu2', 'nu3', 'm12', 'm13', 'm23', 'T1', 'T2']
popt = [1.5, 0.5, 1.0, 0.5, 1.0, 3.0, 0.1, 0.05]

lower_bound = [1e-2, 1e-2, 1e-2, 0, 0, 0, 0, 1e-15, 1e-15]
upper_bound = [100, 100, 100, 10, 10, 10, 10, 5, 5]

mu = 2.5e-8  # mutation rate
L = 20000000  # effective length of sequence
Nanc = 10000
theta = 4 * mu * L * Nanc  # mutation flux

ns_per_pop = 20
ns = [ns_per_pop for _ in range(n_pop)]

# Get maximum log-likelihood
model = model_func(popt, ns)
data = model * theta
max_ll = moments.Inference.ll_multinom(model, data)

if __name__ == "__main__":
    data.to_file('fs_data.fs')
    print('Simulated data saved to fs_data.fs')

    print('Maximum log composite likelihood: {0}'.format(max_ll))

    theta = moments.Inference.optimal_sfs_scaling(model, data)
    print('Optimal value of theta: {0}'.format(theta))

    theta0 = 4 * mu * L
    Nanc = int(theta / theta0)
    print('Size of the ancestral population: {0}'.format(Nanc))

    # Draw model
    model = moments.ModelPlot.generate_model(model_func, popt,
                                             [4 for _ in range(n_pop)])
    moments.ModelPlot.plot_model(model,
        save_file='model_plot.png',
        fig_title=f'Demographic model for 3_DivMig_8_Sim, Nanc: {Nanc}',
        pop_labels=pop_labels,
        nref=Nanc,
        draw_scale=False,
        draw_ancestors=True,
        gen_time=1.0,
        gen_time_units='Generations',
        reverse_timeline=True)
    print('Model plot is saved to model_plot.png')
