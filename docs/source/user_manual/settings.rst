Useful settings
=================

Time for generation
----------------------

Option ``Time for generation`` in the parameter file corresponds to the time per one generation in the Wright-Fisher model. It is responsible basically for one thing: time on the model's plots. If it is specified, then time on the pictures will be converted from genetic units by scaling with this value. Otherwise, if it is not set, time will be shown in genetic units. 

.. note::
    If ``Time for generation`` is specified, it should be consistent with another option: ``Theta0``, which is described the :ref:`corresponding section<theta>`.


Relative parameters
-------------------------------

Sometimes it is more important to see parameters scaled to ``Nref = N_A``. To tell GADMA to show models with scaled parameters, option ``Relative parameters`` should be set to ``True``. By default, it is ``False``. It is convenient when ``Theta0`` is unknown.

No migrations and symmetric migrations
-----------------------------------------

GADMA can exclude migrations rates from optimization and consider them equal to zero. In that case all migrations are disabled. One should set option ``No migrations`` to ``True`` in the parameter file.

To estimate symmetric migrations one should set ``Symmetric migrations`` to True.

.. code-block:: none

    # param file
    ...
    No models: False
    Symmetric migrations: True
    ...

Restrict some migrations
-------------------------

It is possible to restrict some number of migrations by setting the ``Migration masks`` option. It is a list of masks for each time interval after first split.

For example, if there is model structure equal to (2, 1, 1) and one want to have all zero migrations except (a) migration from population 1 to population 2 between split of ancestral population and second split and (b) migration between population 1 and population 3 after second split. Then for first interval after split mask will be ``[[None, 0], [1, None]]`` (``m[i, j]`` corresponds to the migration from the population ``j`` to the population ``i``) and for next time interval right after the second split mask will be ``[[None, 0, 1], [0, None, 0], [1, 0, None]]``.

.. code-block::

    # params_file
    Migration masks: [[[0, 0], [1, 0], [[0, 0, 1], [0, 0, 0], [1, 0, 0]]]

.. note::
    Option ``Migration masks`` is used only in case of demographic model with structure and ``Initial structure`` equal to ``Final structure``.

.. note::
    When option ``Migration masks`` is used together with ``Symmetric migration`` masks should be symmetrical. There is no such option to make some migrations symmetrical and other not.

Split fractions
------------------

Split could be set in two ways:

1) Population is split according to some ``fraction``: ``size * fraction`` becomes size of first subpopulation and ``size * (1 - fraction)`` becomes the size of the second subpopulation. In this case sizes of newly formed populations could not be greater than the size of their parent population.

2) Sizes of newly formed subpopulations are independent from the size of the parent population. In that case the demographic model will have one additional one parameter per each split in it compared to the model from the first point.

.. code-block:: none

    # param file
    Split fractions: True  # for 1) point

Upper bound of split
----------------------------

To limit time of some split one should specify an option in the parameter file. Splits are numbered from the most ancient. So split 1 is split that occurred with the ancient population and split 2 is the next division of the second population (exist only for three populations). There are two appropriate options: ``Upper bound of first split`` and ``Upper bound of second split``.

One should translate time from years into genetic units, therefore divide it by ``2 * T_g``, where ``T_g`` is time (in years) for one generation. For example, one wants to limit the last split to 2000 years. Time for one generation is estimated as 24 years, then one should specify in the parameter file:

.. code-block:: none

    # param_file
    ...
    Upper bound of second split : 83.333
    ...

Resume launch
-----------------

To resume interrupted launch one can use ``--resume`` command-line option or set ``Resume from`` in the parameter file. One needs to set the output directory of the previous run.

If neither ``Output directory`` or ``-o/--output`` is not specified, GADMA will continue evaluation in the directory: ``<previous_output_dir>_resumed``.

Only models
**************

GADMA can resume launch taking final models only from the previous run. This means, that it is not the usual resumption, but run from some initial values. It is useful, for example, when one has to run GADMA with some small grid size for dadi and then wants to restart it with a greater number of grid points. To do so, one should set the command-line option ``--only_models`` with ``--resume`` or specify ``Only models`` option in the parameter file to ``True``.

