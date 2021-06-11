.. GADMA documentation master file, created by
   sphinx-quickstart on Sun Apr 12 13:33:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GADMA's documentation!
=================================

GADMA implements methods for automatic inference of the joint demographic history of multiple populations from genetic data.

GADMA is based on two open source packages: `dadi <https://bitbucket.org/gutenkunstlab/dadi/>`_ developed by Ryan Gutenkunst and `moments <https://bitbucket.org/simongravel/moments/>`_ developed by Simon Gravel.

In contrast to these packages, **GADMA is a command-line tool**. It presents a series of launches of the genetic algorithm and infers demographic history from the Allele Frequency Spectrum of multiple populations (up to three).

Parameter file
--------------

A parameter file is a simple text file (created in a text editor, such as Notepad), which contains a list of parameters and variables with their assigned values. Create a parameter file that reflects your particular parameters. Hereinafter, as an example, the name of your parameter file will be defined as ``param_file``



.. toctree::
   :maxdepth: 1
   :caption: Getting started

   user_manual/installation
   user_manual/hands_on
   examples/examples
   faq

.. toctree::
   :maxdepth: 2
   :caption: User manual

   user_manual/input_data/input_data
   user_manual/set_engine
   user_manual/set_model/set_model
   user_manual/inference
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
   api/gadma
   api_examples/api_examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
