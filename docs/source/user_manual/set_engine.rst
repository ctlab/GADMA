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
    - ``momi``
    - ``moments.LD``

In order to choose engine:

.. code-block::

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
     - ``momi``
     - ``moments.LD``

   * - VCF input data
     - ✓
     - ✓
     - ✓
     - ✓
   * - SFS input data
     - ✓
     - ✓
     - ✓
     - ✕
   * - Can simulate data
     - ✓
     - ✓
     - ✓
     - ✕
   * - Recombination rate
     - ✕
     - ✕
     - ✕
     - ✕
   * - Model plotting
     - ✕
     - ✓
     - ✓ (excludes migration)
     - ✕
   * - Fast multinom inference
     - ✓
     - ✓
     - ✕
     - ✕
   * - Exponential size function
     - ✓
     - ✓
     - ✓
     - ✓
   * - Linear size function
     - ✓
     - ✓
     - ✕
     - ✓
   * - Continuous migration
     - ✓
     - ✓
     - ✕
     - ✓
   * - Selection coefficients
     - ✓
     - ✓
     - ✕
     - ✕
   * - Inbreeding coefficients
     - ✓
     - ✕
     - ✕
     - ✕

dadi
=====

``dadi`` engine (Diffusion Approximation for Demographic Inference) was presented in [Gutenkunst2009]_ and has been developed since then. ``dadi`` engine is able to infer inbreeding coefficients ([Blischak2020]_).

When using the ``dadi`` engine, it is recommended to check the value of the ``Pts`` option in the ``params_file``. ``Pts`` is a sequence of three numbers, each of which is equal to the number of points in grid size. The greater the numbers are, the more accurate numerical solution of a partial differential equation ``dadi`` will give. However, finding such a more accurate solution takes more time. By default, GADMA takes ``Pts : n, n + 10, n + 20``, where ``n`` is the largest sample size among the populations of interest.

.. code-block::

    # param file
    ...
    Engine : dadi
    Pts: [40, 50, 60]
    ...

.. note:: ``Pts`` option is also used for other engines as well: for generation of python code for ``dadi``. So if one wants to use ``dadi``'s code of final models after run then maybe the ``Pts`` argument must be set too.

moments
=======

``moments`` engine [Jouganous2017] is very similar to ``dadi``. Usually it is much faster than ``dadi`` but maybe less accurate. It is the default engine for demographic inference with GADMA.

.. code-block::

    # param file
    ...
    Engine : moments
    ...

``moments`` engine is able to draw plots of demographic models. Be default ``demes`` engine is used as an engine for drawing but ``moments`` can be also chosen. For more information please see `Plotting model <plotting.rst>`__ section.

momi
=====

Another engine for demographic inference is ``momi``. Although GADMA is limited to three populations with a demographic model with structure, ``momi`` and therefore the ``momi`` engine can be used for a lot of populations (e.g. 10) with a custom demographic model.

Custom demographic model given to GADMA will be read in a different way than models for ``dadi`` and ``moments``. The units of parameters are assumed to be in physical units. To mark that some parameter is in genetic units please add `_gen` to its name (e.g. `nu_gen`). `The example of usage of ``momi`` engine with custom demographic model `example <https://gadma.readthedocs.io/en/latest/examples/custom_model_example_momi.html>`_.

Engines ``dadi`` and ``moments`` have a special type of demographic inference that is called multinomial inference: it is possible to infer size of ancestral population implicitly from values of other parameters. However, ``momi`` engine is not able to perform such inference and option ``Ancestral size as parameter`` should be set to ``True``.

Unfortunately, ``momi`` engine has some limitations on demographic parameters: it does not infer continuous migrations and linear size change. If an engine is chosen then GADMA informs about these limitations and disables migration and linear dynamic automatically.

.. code-block::

    # param file
    ...
    Engine : momi
    # the following options are set automatically if momi engine is chosen
    Ancestral size as parameter: True
    No migrations: True
    Dynamics: [Sud, Exp]
    ...


``momi`` engine can be also used to draw demographic models, however, it fails to draw histories with linear size change and does not draw migrations. For more information please see `Plotting model <plotting.rst>`__ section.

moments.LD engine
========================

moments.LD engine is the extension of moments. moments.LD compute a large family of linkage disequilibrium statistics
in model with flexible demographic history with any number of populations.
Unlike other engines, moments.LD does not work with the allele frequency spectrum,
but with LD statistics and stores them in a different way than AFS.

.. code-block::

    # param file
    ...
    Engine : moments.LD
    ...

More about ``moments.LD engine`` :ref:`here <moments_ld_engine>`.