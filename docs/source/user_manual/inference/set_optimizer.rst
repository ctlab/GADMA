How to choose optimizer
=====================================

The default choice of optimization methods in ``params_file`` is the following:

* `Genetic_algorithm` as ``Global optimizer``,
* `BFGS_log` as ``Local optimizer``.

We recommend to use `BFGS_log` as local search algorithm. `Genetic_algorithm` is also the most efficient method for demographic inference with GADMA.

However, there are special cases when it is better to use Bayesian optimization as a global search algorithm. Usually it is case for demographic inference with more than three populations. More about Bayesian optimization could be found :ref:`here<Dem inference for more than three pops>`.

Below there is a table with recomendations about global search algorithm choice. GA stands for the genetic algorithm, BO for Bayesian optimization. Symbol ? means that we have not checked the efficiency for Bayesian optimization in that settings.

.. list-table::
   :header-rows: 1

   * - Pop.num.\Engine
     - ``dadi``
     - ``moments``
     - ``momi2``
     - ``momentsLD``

   * - 1
     - `GA`
     - `GA`
     - `GA`
     - `GA`

   * - 2
     - `GA`
     - `GA`
     - `GA`
     - `GA`

   * - 3
     - `GA` or `BO` *
     - `GA`
     - `GA`
     - ?
   * - 4
     - `BO`
     - `BO`
     - `GA`
     - ?
   * - 5
     - `BO`
     - `BO`
     - `GA`
     - ?
   * - >5
     - ✖ **
     - ✖ **
     - `GA`
     - ✖ **

\* depends on the time of log-likelihood evaluation. If grid size (``Pts``) or sample size of data are big then BO is better choice, otherwise GA should be used.

\*\* Not supported