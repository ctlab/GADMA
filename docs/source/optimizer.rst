.. _sec_api_optimizer:

==================================
Optimizers
==================================

To perform demographic inference optimization should be launched. There are different optimizers in GADMA: local search and global search algorithms.

********************
Base classes
********************

----------------------
Class Optimizer
----------------------
.. autoclass:: gadma.optimizers.Optimizer
    :members:

-------------------------
Class ContinuousOptimizer
-------------------------
.. autoclass:: gadma.optimizers.ContinuousOptimizer
    :show-inheritance:
    :members: check_variables

----------------------------
Class UnconstrainedOptimizer
----------------------------
.. autoclass:: gadma.optimizers.UnconstrainedOptimizer
    :show-inheritance:
    :members: check_variables

----------------------------
Class ConstrainedOptimizer
----------------------------
.. autoclass:: gadma.optimizers.ConstrainedOptimizer
    :show-inheritance:
    :members: check_variables

*******************
Local Optimizers
*******************

Additional local optimizer could be implemented by creating new subclass of class
:class:`gadma.optimizers.LocalOptimizer` and register it with function :func:`gadma.optimizers.register_local_optimizer`.

.. autofunction:: gadma.optimizers.register_local_optimizer
.. autofunction:: gadma.optimizers.get_local_optimizer

.. autofunction:: gadma.optimizers.all_local_optimizers


---------------------------
Registered local optimizers
---------------------------

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

-------------------------
Base class LocalOptimizer
-------------------------

.. autoclass:: gadma.optimizers.LocalOptimizer
    :show-inheritance:
    :members: optimize

--------------------
Class ScipyOptimizer
--------------------

.. autoclass:: gadma.optimizers.ScipyOptimizer
    :show-inheritance:
    :members: save, load, optimize

----------------------------
Class ScipyConstrOptimizer
----------------------------

.. autoclass:: gadma.optimizers.ScipyConstrOptimizer
    :show-inheritance:
    :members: scipy_methods, maxeval_kwarg
    
----------------------------
Class ScipyUnconstrOptimizer
----------------------------

.. autoclass:: gadma.optimizers.ScipyUnconstrOptimizer
    :show-inheritance:
    :members: scipy_methods, maxeval_kwarg



-----------------------------
Class ManuallyConstrOptimizer
-----------------------------

.. autoclass:: gadma.optimizers.ManuallyConstrOptimizer
    :show-inheritance:
    :members: optimize


*******************
Global Optimizers
*******************

Additional dglobal optimizer could be implemented by creating new subclass of class
:class:`gadma.optimizers.GlobalOptimizer` and register it with function :func:`gadma.optimizers.register_global_optimizer`.

.. autofunction:: gadma.optimizers.register_global_optimizer
.. autofunction:: gadma.optimizers.get_global_optimizer

.. autofunction:: gadma.optimizers.all_global_optimizers


----------------------------
Registered global optimizers
----------------------------

The following optimizers are registered:

+----------------------+--------------------------------+--------------------------------------------------+
| ID                   | Description                    | Instance of                                      |
+======================+================================+==================================================+
| "Genetic_algorithm"  | Genetic algorithm optimization | :class:`gadma.optimizers.GeneticAlgorithm`       |
+----------------------+--------------------------------+--------------------------------------------------+

--------------------------
Base class GlobalOptimizer
--------------------------

.. autoclass:: gadma.optimizers.GlobalOptimizer
    :show-inheritance:
    :members: randomize, initial_design, optimize

----------------------------
Сlass GeneticAlgorithm
----------------------------

.. autoclass:: gadma.optimizers.GeneticAlgorithm
    :show-inheritance:
    :members: mutation, crossover, selection, is_stopped, write_report, save, load, optimize