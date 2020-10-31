Repeats and parallel computing
=======================================

By default, GADMA runs optimization that uses Genetic algorithm once. However, it is recommended to run optimization several times and choose the best model among the resieved ones. Option ``Number of repeats`` in the parameter file tells GADMA how many times an optimization should be executed. Moreover, there is another option ``Number of processes`` that allows GADMA to run all the processes in parallel. 

.. code-block:: none

    # param_file
    ...
    Number of repeats : 6
    Number of processes : 2 # or 3 or 6
    ...
    
.. note::
    GADMA parallelize exactly several runs, not every of them. So if one asks to repeat optimization twice and parallelize in more than 2 processes, only two processes will be used eventually.

.. warning::
    ``Number of processes`` should be less than ``Number of repeats`` and it will be better if it is aliquot to the ``Number of repeats``.

.. note::
    ``Number of processes`` shouldn't be greater than the number of kernels on one's computer. Otherwise there will be no sense in parallelization.

