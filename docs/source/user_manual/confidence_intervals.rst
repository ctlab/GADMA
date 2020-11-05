Confidence intervals
======================

GADMA contains special scripts for confidence intervals (CI) evaluation. To get CI one will need correctly bootstrapped data. If SNP's that were used for AFS are unlinked, then usual bootstrap over them should be performed. However, if they are linked then block bootstrap should be used. It is done over the unlinked regions of the genome.

When bootstrapped data is ready, one should run two scripts ``gadma-run_ls_on_boot_data`` and ``gadma-get_confidence_intervals`` in order to get CI. One can find an example `here <https://bitbucket.org/noscode/gadma_results/src/master/YRI_CEU/model_1/>`_.

Run local search on bootstrapped data
----------------------------------------
The first script ``gadma-run_ls_on_boot_data`` runs local search from known optimum for initial AFS (the one that GADMA found) for each AFS from bootstrap. The usage is following:

.. code-block:: console

    $ gadma-run_ls_on_boot_data --help

    usage: GADMA module for runs of local search on bootstrapped data.
    Is needed for calculating confidence intervals.
           [-h] -b <dir> -d <filename> -o <dir> [-j N]
           [--opt log/powell] [-p <filename>]

    optional arguments:
      -h, --help            show this help message and exit
      -b <dir>, --boots <dir>
                            Directory where bootstrapped data is 
                            located.
      -d <filename>, --dem_model <filename>
                            File with demographic model. Should 
                            contain `model_func` or `generated_model`
                            function. One can put there several extra
                            parameters and they will be taken 
                            automatically, otherwise one will need to 
                            enter them manually. Such parameters are: 
                            1) p0 (or popt) - initial parameters values 
                            2) lower_bound - list of lower bounds for 
                            parameters values
                            4) upper_bound - list of upper bounds for 
                            parameters values 
                            5) par_labels/param_labels - list of 
                            string names for parameters 
                            6) pts - pts for dadi (if there is no pts
                            then moments will be run automatically).
      -o <dir>, --output <dir>
                            Output directory.
      -j N, --jobs N        Number of threads for parallel run.
      --opt log/powell      Local search algorithm, by now it can be: 
                            1) `log` - Inference.optimize_log 
                            2) `powell` - Inference.optimize_powell.
      -p <filename>, --params <filename>
                            Filename with parameters, should be valid
                            python file.
                            Parameters are presented in -d/--dem_model
                            option description upper.

After the run, there will be a pandas table ``result_table.pkl`` in the output directory  and its CSV version ``result_table.csv``. It contains parameters for each bootstrap. At this point of time it is possible to change its units and add new parameters with additional manipulations in Python and then run ``gadma-get_confidence_intervals`` to get CI.

Get Confidence Intervals from table
------------------------------------

After the ``gadma-run_ls_on_boot_data`` the result table will be in the output directory. One can change its columns due to what parameters should be used for CI. For example, it is possible to translate units from genetic and add one more parameter: the size of the ancestral population (N\_A). To do it the user should write a script.

To calculate confidence intervals:

.. code-block:: console

    $ gadma-get_confidence_intervals --help
    usage: GADMA module for calculating confidence 
            intervals from the result table of local 
            search runs on bootstrapped data.
           [-h] [--log] [--tex] [--acc N] <filename>

    positional arguments:
      <filename>  Filename (.csv or .pkl) with result from 
                  local search runs on bootstrapped data. 
                  Output of gadma-run_ls_on_boot_data.
    
    optional arguments:
      -h, --help  show this help message and exit
      --log       If log then logarithm will be used to
                  calculate confidence intervals.
      --tex       Tex output.
      --acc N     Accuracy of output (default: 5).
