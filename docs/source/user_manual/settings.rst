Usefull settings
=================

Time for generation
----------------------

Option ``Time for generation`` in the parameter file is corresponding to the time per one generation in Wright-Fisher model. It is responsible basically for one thing: time on the model's plots. If it is specified, then time on the pictures will be converted from genetic units by scaling with this value. Otherwise, if it is not set, time will be shown in genetic units. 

.. note::
    If ``Time for generation`` is specified, it should be consistent with another option: ``Theta0``, which is described in next section.


Relative parameters
-------------------------------

Sometimes it is more important to see parameters scaled to ``Nref = N_A``. To tell GADMA shows models with scaled parameters, option ``Relative parameters`` should be set to ``True``. By default, it is ``False``. It is conveniently when ``Theta0`` is unknown.

No migrations and symmetric migrations
-----------------------------------------

GADMA can to exclude migrations rates from optimization and consider them be equal to zero. In that case all migrations are disabled. One should set option ``No migrations`` to ``True`` in the parameter file.

To estimate symmetric migrations one should set ``Symmetric migrations`` to True.

.. code-block:: none

    # param file
    ...
    No models: False
    Symmetric migrations: True
    ...

Split frations
------------------

Split could be set in two ways:

1) Population is split according to some ``fraction``: ``size * fraction`` becomes size of first subpopulation and ``size * (1 - fraction)`` becomes size of second subpopulation. In this case sizes of newly formed populations could not be greater than size of their parent population.

2) Sizes of newly formed subpopulations are independent from size of the parent population. In that case demographic model will have additional one parameter per each split in it compared to the model from first point.

.. code-block:: none

    # param file
    Split fractions: True  # for 1) point

Upper bound of split
----------------------------

To limit time of some split one should specify option in the parameter file. Splits are numerated from the most ancient. So split 1 is split that occurred with ancient population and split 2 is next division of second population (exist only for three populations). There are two appropriate options: ``Upper bound of first split`` and ``Upper bound of second split``.

One should translate time from years into genetic units, therefore divide it by ``2 * T_g``, where ``T_g`` is time (in years) for one generation. For example, one wants to limit last split with 2000 years. Time for one generation is estimated as 24 years, then one should specify in the parameter file:

.. code-block:: none

    # param_file
    ...
    Upper bound of second split : 83.333
    ...

Resume launch
-----------------

To resume interrupted launch one can use ``--resume`` command-line option or set ``Resume from`` in the parameter file. One needs to set output directory of previous run.

If neither ``Output directory`` or ``-o/--output`` is not specified, GADMA will continue evaluation in the directory: ``<previous_output_dir>_resumed``.

Only models
**************

GADMA can resume launch taking final models only from previous run. This means, that it is not resumption, but run from some initial values. It is useful, for example, when one has to run GADMA with some small grid size for dadi and then wants to restart it with greater number of grid points. To do so, one should set command-line option ``--only_models`` with ``--resume`` or specify ``Only models`` option in the parameter file to ``True``.

