Inference
===========

.. toctree::
   :hidden:

   set_optimizer
   bayes_opt

.. admonition:: Related options

    * **Base options**:

      * ``Global optimizer`` - name of global optimization (`Genetic algorithm`).
      * ``Local optimizer`` - name of local search algorithm (`BFGS_log`).

    * **Additional options**:

      * ``Global maxiter`` - number of iterations for global optimization (`None`).
      * ``Global maxeval`` - number of log-likelihood evaluations for global optimization (`None`).
      * ``Global log transform`` indicates if data is log-transformed in global optimization (`False`).
      * ``Local maxiter`` - number of iterations for local search algorithm (`None`).
      * ``Local maxeval`` - number of log-likelihood evaluations for local search algorithm (`None`).
      * ``Local log transform`` indicates if data is log-transformed in local search algorithm (`True`).
      * ``Num init const`` - scaling constant (to number of model parameters) to define number of random points in initial design (`10`).

GADMA could be customized to use different to default optimization algorithms:

* :ref:`List of all available global optimizers.<Global optimizers list>`
* :ref:`List of all available local optimizers.<Local optimizers list>`

More specific information:

- `How to choose optimizer <set_optimizer.rst>`__
- `Demographic inference for more than three populations <bayes_opt.rst>`__

GADMA's inference could be run both from command-line and from Python directly.

Command-line
------------

Usage of GADMA:

.. code-block:: console

    $ gadma --help
    
    GADMA version 2.0.0	by Ekaterina Noskova (ekaterina.e.noskova@gmail.com)
    Usage: 
        	gadma	-p/--params <params_file>
            		-e/--extra <extra_params_file>


    Instead/With -p/--params and -e/--extra option you can set:
        	-o/--output <output_dir>	output directory.
        	-i/--input <in.fs>/<in.txt>	input file with AFS or in dadi format.
        	--resume <resume_dir>		resume another launch from <resume_dir>.
        	--only_models			flag to take models only from another launch (--resume option).

        	-h/--help		show this help message and exit.
        	-v/--version		show version and exit.
        	--test			run test case.

        In case of any questions or problems, please contact: ekaterina.e.noskova@gmail.com


Resume launch
-----------------

.. admonition:: Related options

    * **Base options**:

      * ``Resume from``

    * **Additional options**:

      * ``Only models``

To resume interrupted launch one can use ``--resume`` command-line option or set ``Resume from`` in the parameter file. One needs to set the output directory of the previous run.

If neither ``Output directory`` or ``-o/--output`` is not specified, GADMA will continue evaluation in the directory: ``<previous_output_dir>_resumed``.

Only models
**************

GADMA can resume launch taking final models only from the previous run. This means, that it is not the usual resumption, but run from some initial values. It is useful, for example, when one has to run GADMA with some small grid size for dadi and then wants to restart it with a greater number of grid points. To do so, one should set the command-line option ``--only_models`` with ``--resume`` or specify ``Only models`` option in the parameter file to ``True``.


Run optimizer from Python
---------------------------

Genetic algorithm pipeline from GAGMA is available from ``gadma``'s API by calling ``gadma.Inference.optimize_ga`` function. It is like usual optimization functions in ``dadi`` and ``moments``. The short notice from the API:

.. autofunction:: gadma.Inference.optimize_ga
