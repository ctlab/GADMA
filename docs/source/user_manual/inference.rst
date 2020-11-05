Inference
===========

GADMA's inference could be run both from command-line and from Python directly.

Command-line
--------------

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

Run optimizer from Python
---------------------------

Genetic algorithm pipeline from GAGMA is available from ``gadma``'s API by calling ``gadma.Inference.optimize_ga`` function. It is like usual optimization functions in ``dadi`` and ``moments``. The short notice from the API:

.. autofunction:: gadma.Inference.optimize_ga
