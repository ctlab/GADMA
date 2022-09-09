.. _Dem inference for more than three pops:

Demographic inference for more than three populations
=======================================================

The demographic inference for more than three populations is a special case for GADMA.

First of all, the method for automatic inference using structure demographic model is not working. The only option for four and more populations is to use :ref:`custom model<Custom demographic model>` specified by the researcher using the utilities of the engines.

Moreover, if either ``dadi``, ``moments`` or ``momentsLD`` are specified as an engine in GADMA then Bayesian optimization (BO) should be used for faster inference instead usual genetic algorithm.

.. warning::
    if ``momi2`` is chosen as an engine in GADMA then Bayesian optimization is a bad choice. `Momi2` is very fast and it is better to use the usual genetic algorithm for it.

To change optimization to Bayesian optimization set:

.. code-block:: none

    # param_file
    ...
    # Set Bayesian optimization
    Global optimizer : SMAC_BO_combination
    # Set small initial design
    Num init const: 2
    # Set number of evaluations
    Global maxeval: 200
    ...

Setting ``Global maxeval`` tells GADMA how many evaluations of log-likelihood should be performed. It is required for Bayesian optimization. We recommend 200 evaluations for demographic inference with <15 parameters and 400 evaluations if the number of parameters is bigger than 15.

Setting ``Num init const`` should be set to 2. Bayesian optimization require much smaller initial random search and this option restricts number of evaluations there.

`The example of demographic inference for four populations. <https://gadma.readthedocs.io/en/latest/examples/inference_for_four_and_five_populations.html>`_