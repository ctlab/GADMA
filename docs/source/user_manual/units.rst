Units of parameters
============================

GADMA shows model parameters and their units in general ``GADMA.log`` output at the end of each model, for example:

.. code-block:: console

    [000:01:20]
    All best by log-likelihood models
    Number	log-likelihood		Model	Units
    Run 1	-1277.31	 [Nanc = 8273]	physical, time in generations

There are two possible types of units:

* Physical
* Genetic

Time units (generations or years) are specified when units are physical.

.. warning::
    Parameters specified in additional output files (e.g. in ``1/GADMA_GA.log``) are always in genetic units.

Physical Units
--------------

.. list-table:: Definition of physical units for parameters
   :widths: 25 10 65
   :header-rows: 1

   * - Parameter type
     - Symbol
     - Description of physical units
   * - Population size
     - ``N``
     - Number of individuals
   * - Time
     - ``T``
     - | Generations or years
       | (depending on ``Time for generation`` option)
   * - Migration rate
     - ``m``
     - Proportion of migrants in population per generation
   * - Inbreeding coefficient
     - ``F``
     - | Probability that two alleles at any locus in an
       | individual are identical by descent from the
       | common ancestor(s) of the two parents
   * - Fraction in population split
     - ``s``
     - | Franction of population size that goes stays at 
       | the same population

In order to get output in physical units the following options should be specified in ``params_file``:

* ``Mutation rate``
* ``Sequence length``

If ``Time for generation`` is specified then time will be presented not in generations (default), but years.


Genetic units
-------------

Parameters are presented in genetic units means that they are scaled to the reference population size ``Nref`` that is ancestral population size.

.. list-table:: Definition of genetic units for parameters
   :widths: 25 10 65
   :header-rows: 1

   * - Parameter type
     - Symbol
     - Description of physical units
   * - Population size
     - ``N``
     - Number of individuals divided by ``Nref``
   * - Time
     - ``T``
     - Generations divided by ``2 * Nref``
   * - Migration rate
     - ``m``
     - | Proportion of migrants in population per
       | generation multiplied by ``2 * Nref``
   * - Inbreeding coefficient
     - ``F``
     - Fraction, the same as for physical units
   * - Fraction in population split
     - ``s``
     - Franction, the same as for physical units


In order to get output in genetic units the ``Relative parameters`` option should be set to ``True`` in ``params_file``.
If ``Mutation rate`` or ``Sequence length`` are not specified, then ``Relative parameters`` option is set automatically to be ``True``.

To translate parameters from genetic units to physical units one should first get ``Nref`` as:

``theta`` / (4 * ``mu`` * ``L``),

where ``mu`` is mutation rate, ``L`` corresponds to sequence length and ``theta`` is given in the output of the model.
Then translate values the following way: sizes of populations — multiply by ``Nref``; time — multiply by ``2 * Nref``; migration rates — divide by ``2 * Nref``.