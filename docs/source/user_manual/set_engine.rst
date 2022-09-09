=======================
Specifying an engine
=======================

.. admonition:: Related options

    * **Base options**:

      * ``Engine`` indicates what engine will be used for demographic inference.

    * **Additional options**:

      * ``Pts`` - grid sizes for ``dadi`` engine.

GADMA provides the following choice of engines for demographic inference:

    - ``dadi``
    - ``moments`` (by default)
    - ``momi2``
    - ``momentsLD``

In order to choose engine:

.. code-block:: none

   # param file
   ...
   Engine : moments
   ...

Engine comparison
=================

.. list-table::
   :header-rows: 1

   * - Feature
     - ``dadi``
     - ``moments``
     - ``momi2``
     - ``momentsLD``

   * - Max pop. num. custom model
     - 5
     - 5
     - ∞
     - 5

   * - Max pop. num. structure model
     - 3
     - 3
     - 3
     - 3

   * - VCF input data
     - ✅
     - ✅
     - ✅
     - ✅
   * - SFS input data
     - ✅
     - ✅
     - ✅
     - ❌ (LD stats)
   * - Can simulate data
     - ✅
     - ✅
     - ✅ (``msprime``)
     - ❌
   * - Recombination rate
     - ❌
     - ❌
     - ❌
     - ✅
   * - Model plotting
     - ❌
     - ✅
     - ✅*
     - ❌
   * - Fast multinom inference
     - ✅
     - ✅
     - ❌
     - ❌
   * - Exponential size function
     - ✅
     - ✅
     - ✅
     - ✅
   * - Linear size function
     - ✅
     - ✅
     - ❌
     - ✅
   * - Continuous migration
     - ✅
     - ✅
     - ❌
     - ✅
   * - Selection coefficients
     - ✅
     - ✅
     - ❌
     - ❌
   * - Inbreeding coefficients
     - ✅
     - ❌
     - ❌
     - ❌

\* No migrations are drawn

dadi
=====

``dadi`` engine (Diffusion Approximation for Demographic Inference) was presented in [Gutenkunst2009]_ and has been developed since then. ``dadi`` engine is able to infer inbreeding coefficients ([Blischak2020]_).

When using the ``dadi`` engine, it is recommended to check the value of the ``Pts`` option in the ``params_file``. ``Pts`` is a sequence of three numbers, each of which is equal to the number of points in grid size. The greater the numbers are, the more accurate numerical solution of a partial differential equation ``dadi`` will give. However, finding such a more accurate solution takes more time. By default, GADMA takes ``Pts : n, n + 10, n + 20``, where ``n`` is the largest sample size among the populations of interest.

.. code-block:: none

    # param file
    ...
    Engine : dadi
    Pts: [40, 50, 60]
    ...

Moreover, the type of extrapolation in ``dadi`` could be changed from logarithmic ('make_extrap_log_func') to linear if desired:

.. code-block:: none

    # param file
    ...
    Engine : dadi
    Pts: [40, 50, 60]
    Dadi extrapolation: make_extrap_func
    ...

.. note:: ``Pts`` option is also used for other engines as well: for generation of python code for ``dadi``. So if one wants to use ``dadi``'s code of final models after run then maybe the ``Pts`` argument must be set too.

moments
=======

``moments`` engine [Jouganous2017] is very similar to ``dadi``. Usually it is much faster than ``dadi`` but maybe less accurate. It is the default engine for demographic inference with GADMA.

.. code-block:: none

    # param file
    ...
    Engine : moments
    ...

``moments`` engine is able to draw plots of demographic models. Be default ``demes`` engine is used as an engine for drawing but ``moments`` can be also chosen. For more information please see `Plotting model <plotting.rst>`__ section.

momi2
=====

Another engine for demographic inference is ``momi2`` ([Kamm2020]_). Although GADMA is limited to three populations with a demographic model with structure, ``momi2`` and therefore the ``momi2`` engine can be used for a lot of populations (e.g. 10) with a custom demographic model.

Custom demographic model given to GADMA will be read in a different way than models for ``dadi`` and ``moments``. The units of parameters are assumed to be in physical units. To mark that some parameter is in genetic units please add `_gen` to its name (e.g. `nu_gen`). The example ``momi2`` engine usage with custom demographic model can be found `here <https://gadma.readthedocs.io/en/latest/examples/custom_model_example_momi.html>`_.

Engines ``dadi`` and ``moments`` have a special type of demographic inference that is called multinomial inference: it is possible to infer size of ancestral population implicitly from values of other parameters. However, ``momi2`` engine is not able to perform such inference and option ``Ancestral size as parameter`` should be set to ``True``.

Unfortunately, ``momi2`` engine has some limitations on demographic parameters: it does not infer continuous migrations and linear size change. If an engine is chosen then GADMA informs about these limitations and disables migration and linear dynamic automatically.

.. code-block:: none

    # param file
    ...
    Engine : momi2
    # Mutation rate and sequence length are required
    Sequence length: 1e9
    Mutation rate: 1.25e-8

    # the following options are set automatically if momi2 engine is chosen
    Ancestral size as parameter: True
    No migrations: True
    Dynamics: [Sud, Exp]
    ...


``momi2`` engine can be also used to draw demographic models, however, it fails to draw histories with linear size change and does not draw migrations. For more information please see `Plotting model <plotting.rst>`__ section.

momentsLD
=========

This section contains all basic information about the ``momentsLD`` engine. Additional specifications are located :ref:`here <moments_ld_engine>`.

``momentsLD`` engine ([Ragsdale2019]_, [Ragsdale2020]_) is the extension of moments.
Unlike other engines, ``momentsLD`` does not work with the allele frequency spectrum,
but with LD (linkage disequilibrium) statistics and stores them in a different way than AFS.

``momentsLD`` works with data provided in VCF format. VCF file must contain all chromosomes you want to use in the analysis. If you have few separated VCF files you can concat them using bcftools or similar programs. GADMA takes only one single VCF file as input. Population map file is also required. In case of several chromosomes it is required that option ``Sequence length`` is set as a dictionary with separate lengths of each chromosome.

Parsing LD statistics from an input VCF file is a time-consuming process which is not a main part of the
GADMA genetic algorithm evaluation. If you start GADMA several times, it will spend a lot of time parsing
LD statistics from data. In this case, GADMA has the option of precomputing data before the main process starts. For more information see the :ref:`special section <precomputing_data>`.

momentsLD estimate LD statistics binning observed pairs of SNP by recombination distances.
It can bin pairs by physical distances, but genetic maps are non-uniform and do not perfectly correlate
with recombination distances. When working with momentsLD it is recommended to use **recombination rate** that will be used for auto generation of recombination maps. It is possible to set recombination maps instead of recombination rate, for more information see :ref:`additional section <rec_maps_ld>`.

LD statistics are evaluated across several regions of a given sequence. GADMA allows to specify length of one region (``Region len`` option), it can be considered in a sense very similar to the ``Pts`` option for ``dadi`` engine. The greater the number of regions is, the lower noise level in the processed data will be. But at the same time, it is important to keep balance between the number of regions and the length of each individual region. Regions that are too small are also not suitable for work. By default, GADMA has ``Region len`` equal to 6,400,000 bp that will provide ~500 regions for full human data.

.. code-block:: none

    # param file
    ...
    Engine : momentsLD
    # Data should be in VCF format, sequence length should be very specific
    Input data: path_to_vcf_file_with_chr1_chr2, path_to_popmap_file
    Sequence length: {chromosome1: 249250621, chromosome2: 243199373}
    Mutation rate: 1.25e-8
    # Recombination rate is recommended to be specified8
    Recombination rate: 1.5e-8

    # Length of each region that will be used to evaluate LD stats
    Region len: 6400000

    # the following option is set automatically if momentsLD engine is chosen with structure model
    Ancestral size as parameter: True
    ...

.. toctree::

   moments_ld_engine
