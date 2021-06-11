Extra parameter file
=====================

GADMA take an extra parameter file as input. **However one probably do not need them**. Nevertheless, if one is interested, ``extra_params_template`` with all options and their descriptions can be found in ``gadma/cli/`` folder.

.. admonition:: Related options

    * **Options of parameter bounds**:

      * ``Min N`` - minimum value (in genetic units) of population size (`0.01`).
      * ``Max N`` - maximum value (in genetic units) of population size (`100`).
      * ``Min T`` - minimum value (in genetic units) of epoch time (`0`).
      * ``Max T`` - maximum value (in genetic units) of epoch time (`5`).
      * ``Min M`` - minimum value (in genetic units) of migration rate (`0`).
      * ``Max M`` - maximum value (in genetic units) of migration rate (`10`).

    * **Options of genetic algorithm**:

      * ``Size of generation`` - number of demographic models on one iteration (generation) of the genetic algorithm (`10`).
      * ``Fractions`` - fractions of best, mutated and crossed models that are taken to the new generation (`None`).
      * ``N elitism`` - number of best models that are taken to the new generation (`2`).
      * ``P mutation`` - fraction of models that are mutants in new generation (`0.56`).
      * ``P crossover`` - fraction of models that are children in new generation (`0.19`).
      * ``P random`` - fraction of models that were randomly generated for new generation (`0.13`).
      * ``Mean mutation strength`` - initial value of mean number of parameters that are mutated in the chosen model during mutation (`0.63`).
      * ``Const for mutation strength`` - constant for 'one-fifth' rule to change mutation strength (`1.02`).
      * ``Mean mutation rate`` - initial value of mean rate of parameter change during mutation (`0.45`).
      * ``Const for mutation rate`` - constant for 'one-fifth' rule to change mutation rate (`1.07`).
      * ``Stuck generation number`` - genetic algorithm stops when there is no improvement during this number of iterations.
      * ``Eps`` - change of log-likelihood that is considered significant.
      * ``Random N_A`` - enables random generation of ancestral size value and scales other parameters to this value during random generation of model parameters (`True`).

    * **Options of bayesian optimization**:

      * ``Kernel`` - name of kernel for Gaussian process (`Matern52`).
      * ``Acquisition function`` - name of acquisition function (`EI`).

    * **Options of global optimizations**:

      * ``X_init`` - points for initial design.
      * ``Y_init`` - value of log-likelihood on this points.
