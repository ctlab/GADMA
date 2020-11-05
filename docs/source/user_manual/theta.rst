.. _theta:

Theta0
===========

``Theta0`` is equal to the expected number of mutations that occur in one chromosome in one generation in the infinite-sited model. GADMA can scale all values of demographic model parameters due to known value of ``Theta0``. However, it is not always possible to find it. There is a way to solve this problem: one can set ``Theta0`` to ``None`` or just not specify it at all, so GADMA will take it as ``1.0`` and after launch one can scale result values due to found ``Theta0``.

The alternative way to set ``Theta0`` is to specify both mutation rate and length of the sequence in the parameters file instead. Then ``Theta0`` will be calculated automatically.

.. code-block:: none

    # param file
    ...
    Mutation rate: 2.35e-8
    Sequence length: 4.04e6
    ...

Estimating Theta0
-------------------

If ``mu`` is the neutral mutation rate per site per generation and ``L`` is the length of the sequence, then:

``Theta_0`` = 4 \* ``mu`` \* ``L``

.. note::
    ``L`` is the effective sequence length, which accounts for losses in alignment and missed calls.

.. note::
    ``mu`` should be estimated based on generation time. One can leave ``Time per generation`` option in the parameter file unspecified (then time on the model's plots will be in the genetic units), but one should recalculate ``mu``!

**For example (Gutenkunst et al, 2009)**:
    We estimate the neutral mutation rate ``mu`` using the divergence between human and chimp. Comparing aligned sequences in our data, we estimate the divergence to be 1.13\%. Assuming a divergence time of 6 million years and a mean generation time for human and chimp over this interval of 25 years, we have

    ``mu`` = 0.0113 \* 25 / (2 \* 6 \* 10^6) = 2.35 \* 10^{-8} (per generation).

Changing Theta0
--------------------

If GADMA was launched with one ``Theta0`` and now one wants to use another or if it was launched with default ``Theta0 = 1`` and now one has estimated its real value, model's parameters can be simply scaled:

Let  ``a`` = ``Theta0_NEW`` / ``Theta0_OLD``,

    * Size of population / ``a``,
    * Time / ``a``,
    * Migration rates \* ``a``,
    * Split percent stay the same.


Examples of Theta0 and time for generation
---------------------------------------------

The following tables produce different possible values for the demographic model inference for three populations of modern people: YRI, CEU, CHB.

Examples of different values of generation time and its influence on ``mu`` and ``Theta0``:

+------------------+---------------------------+---------------------------+----------------------------+---------------------+
| FS filename      | Gen. time                 | ``mu``                    |  ``L``                     | ``Theta0``          |
|                  |                           |                           |                            |                     |
|                  | (years)                   | (per site per gen.)       | (base pair)                | (per chr. per gen.) |
+==================+===========================+===========================+============================+=====================+
| YRI\_CEU\_CHB.fs | 25                        | 2.35 \* 10^{-8}           | 4.04 \* 10^6               | 0.37976             |
|                  | (Gutenkunst et al., 2009) | (Gutenkunst et al., 2009) | (Gutenkunst et al., 2009)  |                     |
+------------------+---------------------------+---------------------------+----------------------------+---------------------+
| YRI\_CEU\_CHB.fs | 24                        | 2.26 \* 10^{-8}           | 4.04 \* 10^6               | 0.36521             |
|                  | (Lapierre et al., 2017)   | (Gutenkunst et al., 2009) | (Gutenkunst et al., 2009)  |                     |
+------------------+---------------------------+---------------------------+----------------------------+---------------------+
| YRI\_CEU\_CHB.fs | 29                        | 1.44 \* 10^(-8)           | 4.04 \* 10^6               | 0.23270             |
|                  | (Jouganous et al., 2017)  | (Jouganous et al., 2017)  | (Gutenkunst et al., 2009)  |                     |
+------------------+---------------------------+---------------------------+----------------------------+---------------------+
| YRI\_CEU\_CHB.fs | 24                        | 1.2 \* 10^(-8)            | 4.04 \* 10^6               | 0.19392             |
|                  | (Lapierre et al., 2017)   | (Jouganous et al., 2017)  | (Gutenkunst et al., 2009)  |                     |
+------------------+---------------------------+---------------------------+----------------------------+---------------------+

In Gutenkunst et al. 2009 generation time for human populations was equal to ``25`` years and mutation rate ``mu`` was estimated as ``2.35 * 10^(-8)``. If one wants to change time for one generation to ``24`` years, one needs to scale ``mu``: ``mu`` / 25 \* 24 = 2.26 \* 10^(-8).

In Jouganous et al. 2017 generation time was grater - ``29`` years and mutation rate was equal to ``1.44 \* 10^(-8)``. To change generation time to ``24``, one needs to change value of the mutation rate: ``muNEW`` = ``mu`` / 29 \* 24 = 1.2 \* 10^(-8). ``Theta0`` is calculated then by the formula above.

.. note::
    There is another more practical :ref:`example<theta_example>` of changing theta after run.
