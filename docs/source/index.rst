.. GADMA documentation master file, created by
   sphinx-quickstart on Sun Apr 12 13:33:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GADMA's documentation!
=================================

GADMA [Noskova2020]_ implements methods for automatic inference of the joint demographic history of multiple populations from genetic data.

**GADMA is a command-line tool**. It presents a series of launches of the global and local search algorithms and infers demographic history of multiple populations. The automatic inference handles up to three populations.

GADMA provides choice of several engines for the demographic inference:

* `dadi <https://bitbucket.org/gutenkunstlab/dadi/>`_  developed by Ryan Gutenkunst [Gutenkunst2009]_
* `moments <https://bitbucket.org/simongravel/moments/>`_  developed by Simon Gravel [Jouganous2017]_
* `momi2 <https://github.com/popgenmethods/momi2/>`_ [Kamm2020]_
* `momentsLD <https://bitbucket.org/simongravel/moments/>`_ extension of ``moments`` for LD stats [Ragsdale2019]_ [Ragsdale2020]_

GADMA implements two base global search algorithms:

* Genetic algorithm — the most common choice of optimization,
* Bayesian optimization — for demographic inference with time-consuming evaluations, e.g. for four and five populations using *moments* or ∂a∂i.


GADMA features variuos optimization methods (`global <https://gadma.readthedocs.io/en/latest/api/gadma.optimizers.html#global-optimizers-list>`_ and `local <https://gadma.readthedocs.io/en/latest/api/gadma.optimizers.html#local-optimizers-list>`_ search algorithms) which may be used for `any general optimization problem <https://gadma.readthedocs.io/en/latest/api_examples/optimization_example.html>`_.


Base usage of GADMA via command-line:

.. code-block:: console

    $ gadma --help
    
    GADMA version 2.0.0	by Ekaterina Noskova (ekaterina.e.noskova@gmail.com)
    Usage: 
        gadma	-p/--params	<params_file>
                -e/--extra	<extra_params_file>


    Instead/With -p/--params and -e/--extra option you can set:
        -o/--output	<output_dir>		output directory.
        -i/--input	<in.fs>/<in.txt>/	input data for demographic inference
        		<in.vcf>,<popmap>	(AFS or dadi format or VCF).
        --resume	<resume_dir>		resume another launch from <resume_dir>.
        --only_models		flag to take models only from another
        			launch (--resume option).

        -h/--help		show this help message and exit.
        -v/--version		show version and exit.
        --test			run test case.

    In case of any questions or problems, please contact: ekaterina.e.noskova@gmail.com

What is parameter file of GADMA?
---------------------------------

A parameter file is a simple text file (created in a text editor, such as Notepad), which contains a list of parameters/options/settings with their assigned values. Create a parameter file that reflects your particular options. Hereinafter, as an example, the name of your parameter file will be defined as ``param_file``.
 
.. note::
    Each section of user manual contain list of related options for parameter file with small description.

    Full example of parameter file could be found in a section :ref:`param_file`.


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   user_manual/installation
   user_manual/hands_on
   examples/examples
   faq
   citations

.. toctree::
   :maxdepth: 2
   :caption: User manual

   user_manual/input_data/input_data
   user_manual/set_engine
   user_manual/set_model/set_model
   user_manual/inference/inference
   user_manual/run_parallel
   user_manual/output
   user_manual/plotting
   user_manual/extra_params_file
   user_manual/units
   user_manual/confidence_intervals
   user_manual/example_params_file

.. toctree::
   :maxdepth: 2
   :caption: Development documentation

   changelogs
   development_workflow
   api/gadma
   api_examples/api_examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
