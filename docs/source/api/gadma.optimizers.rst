Optimizers (optimizers package)
=====================================

To perform demographic inference optimization should be launched. There are different optimizers in GADMA: local search and global search algorithms.

Base Classes
---------------------------------

Module :mod:`gadma.optimizers.optimizer` contains several base classes of optmizers.

.. automodule:: gadma.optimizers.optimizer
   :members: Optimizer, ContinuousOptimizer, ConstrainedOptimizer, UnconstrainedOptimizer
   :undoc-members:
   :show-inheritance:

Global optimizers
-----------------------------------------

Module :mod:`gadma.optimizers.global_optimizer` contains base class for global optimizers.

Additional global optimizer could be implemented by creating new subclass of class :class:`gadma.optimizers.GlobalOptimizer` and register it with function :func:`gadma.optimizers.register_global_optimizer`.

Registered global optimizers
****************************

The following optimizers are registered:

+--------------------------+--------------------------------+--------------------------------------------------+
| ID                       | Description                    | Instance of                                      |
+==========================+================================+==================================================+
| "Genetic_algorithm"      | Genetic algorithm optimization | :class:`gadma.optimizers.GeneticAlgorithm`       |
+--------------------------+--------------------------------+--------------------------------------------------+
| "Bayesian_optimization"  | Bayesian optimization          | :class:`gadma.optimizers.BayesianOptimization`   |
+--------------------------+--------------------------------+--------------------------------------------------+

.. automodule:: gadma.optimizers.global_optimizer
   :members: register_global_optimizer, get_global_optimizer, all_global_optimizers, GlobalOptimizer
   :undoc-members:
   :show-inheritance:

Genetic algorithm
**************************

.. automodule:: gadma.optimizers.genetic_algorithm
   :members:
   :undoc-members:
   :show-inheritance:

Bayesian optimization
*****************************

.. automodule:: gadma.optimizers.bayesian_optimization
   :members:
   :undoc-members:
   :show-inheritance:


Local optimizers
----------------------------------------

Module :mod:`gadma.optimizers.local_optimizer` contains classes for local serach optimizers.

Additional local optimizer could be implemented by creating new subclass of class :class:`gadma.optimizers.local_optimizer.LocalOptimizer` and register it with function :func:`gadma.optimizers.local_optimizer.register_local_optimizer`.


Registered local optimizers
***************************

The following optimizers are registered:

+-----------------+------------------------------+--------------------------------------------------+
| ID              | Description                  | Instance of                                      |
+=================+==============================+==================================================+
| None or "None"  | None optimization is run     | :class:`gadma.optimizers.NoneOptimizer`          |
+-----------------+------------------------------+--------------------------------------------------+
| "L-BFGS-B"      | L-BFGS-B from scipy          | :class:`gadma.optimizers.ScipyConstrOptimizer`   |
+-----------------+------------------------------+--------------------------------------------------+
| "L-BFGS-B_log"  | L-BFGS-B from scipy with log | :class:`gadma.optimizers.ScipyConstrOptimizer`   |
|                 | transform of values          |                                                  |
+-----------------+------------------------------+--------------------------------------------------+
| "BFGS"          | Constrained BFGS from scipy  | :class:`gadma.optimizers.ManuallyConstrOptimizer`|
+-----------------+------------------------------+--------------------------------------------------+
| "BFGS_log"      | Constrained BFGS from scipy  | :class:`gadma.optimizers.ManuallyConstrOptimizer`|
|                 | with log transform of values |                                                  |
+-----------------+------------------------------+--------------------------------------------------+
| "Powell"        | Constrained Powell's method  | :class:`gadma.optimizers.ManuallyConstrOptimizer`|
|                 | from scipy                   |                                                  |
+-----------------+------------------------------+--------------------------------------------------+
| "Powell_log"    | Constrained Powell's method  | :class:`gadma.optimizers.ManuallyConstrOptimizer`|
|                 | from scipy                   |                                                  |
|                 | with log transform of values |                                                  |
+-----------------+------------------------------+--------------------------------------------------+
| "Nelder-Mead"   | Constrained Nelder-Mead      | :class:`gadma.optimizers.ManuallyConstrOptimizer`|
|                 | method from scipy            |                                                  |
+-----------------+------------------------------+--------------------------------------------------+
|"Nelder-Mead_log"| Constrained Nelder-Mead      | :class:`gadma.optimizers.ManuallyConstrOptimizer`|
|                 | method from scipy            |                                                  |
|                 | with log transform of values |                                                  |
+-----------------+------------------------------+--------------------------------------------------+

.. automodule:: gadma.optimizers.local_optimizer
   :members: register_local_optimizer, get_local_optimizer, all_local_optimizers, LocalOptimizer, ScipyOptimizer, ScipyConstrOptimizer, ScipyUnconstrOptimizer, ManuallyConstrOptimizer, NoneOptimizer
   :undoc-members:
   :show-inheritance:

Combinations of optimizers
------------------------------------

Module :mod:`gadma.optimizers.combinations` contains classes of optimizers that are combinations of other optimizers.

.. automodule:: gadma.optimizers.combinations
   :members:
   :undoc-members:
   :show-inheritance:

Optimizer result
-----------------------------------------

.. automodule:: gadma.optimizers.optimizer_result
   :members:
   :undoc-members:
   :show-inheritance:

Linear constrain
-----------------------------------------

.. automodule:: gadma.optimizers.linear_constrain
   :members:
   :undoc-members:
   :show-inheritance:
