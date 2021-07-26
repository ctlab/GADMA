Input data
=============

.. admonition:: Related options

    * **Base options**:

      * ``Input data`` holds path(s) to file(s) with data (AFS).

    * **Additional options**:

      * ``Population labels`` sets population names manually.
      * ``Projections`` corresponds to size of AFS data.
      * ``Outgroup`` indicates if data has outgroup information and thus if AFS is folded (no outgroup).
      * ``Sequence length`` holds effective length of sequence that was used to build AFS.
      * ``Linked SNP's`` enables AIC evaluation.
      * ``Directory with bootstrap`` enables CLAIC evaluation


Data formats
--------------

.. toctree::
   :hidden:

   vcf_data_format
   fs_data_format
   snp_data_format
   obs_data_format

GADMA supports several types of input data, which are familiar to anyone who has used ``dadi`` or ``moments`` in the past:

   * `VCF file format (.vcf + popmap file) <vcf_data_format.rst>`__
   * `Frequency spectrum file format (.fs or .sfs) <fs_data_format.rst>`__
   * `SNP data format (.txt) <snp_data_format.rst>`__
   * `Fastsimcoal2 frequency spectrum file format (.obs) <obs_data_format.rst>`__

Input file can be specified to GADMA in two ways:

1) Through command-line option ``-i/--input``:

   .. code-block:: console

      $ gadma -i fs_file.fs -o out_dir

   or

   .. code-block:: console

      $ gadma --input snp_file.txt -o out_dir

.. note::
    If VCF file is set as input data then popmap file should be given as well.
    Files should be separated by comma, however,
    **do not use space between two files if files are set via command line**:

    .. code-block:: console

        $ gadma --input vcf_file.vcf,popmap_file -o out_dir


2) Use a parameter ``Input data`` in the parameter file:

   .. code-block:: none

      # param_file

      # Input file path
      Input data : fs_file.fs
      ...

Extra information about data
----------------------------

Extra information about input AFS can also be put in the parameter file. For example, AFS can be projected to a smaller size with ``Projections`` option, populations can be named or their order can be changed with the ``Population labels`` option. Option ``Outgroup`` tells whether data has outgroup or not: if ``Outgroup`` is ``False`` then the spectrum will be folded. If the parameter file does not contain some options, they are automatically pulled out from the input file. Also length of sequence could be set by the ``Sequence length`` option, which could be used along with ``Mutation rate`` instead of ``Theta0``.

.. code-block:: none

   # param_file

   # Input file path
   Input data : fs_file.fs

   # (New) size of the AFS
   Projections : 20,20

   # Labels of populations
   Population labels : CEU, YRI

   # Has data outgroup
   Outgroup: True

   # Length of sequence used to build AFS
   Sequence length: 4.04e6
   ...

GADMA can be launched with a parameter file in the following way:

.. code-block:: console

   $ gadma -p params_file -o out_dir

Unlinked SNPs, AIC and CLAIC
-----------------------------

By default, SNP's that were used to build AFS are considered to be linked. In this case it is possible to compare demographic models with different number of parameters by Composite Likelihood Akaike Information Criterion (CLAIC) [Coffman2016]_. This procedure can be necessary as a model with a large number of parameters will be better able to find parameter values corresponding to the observed data than a model with a smaller number of parameters, but at the same time it will correspond less to reality, for example, due to data errors. It is called overfitting and we do not want it to happen.

Actually, CLAIC is modification of usual Akaike Information Criterion (AIC), but AIC can be used only for AFS built from unlinked SNP's. The smaller the AIC or CLAIC score is, the better the model fits the observed data.

It is possible to inform GADMA about linkage of SNP's and unlock the usage of AIC by setting ``Linked SNP's`` option to ``False``:

.. code-block:: none

   # param_file

   # Inform if SNP's are not linked
   Linked SNP's : False
   ...

If SNP's are linked and CLAIC should be evaluated (by default it is not), then the bootstrapped data should be set via ``Directory with bootstrap`` option. In order to receive reliable correct bootstrapped data, the bootstrap should be performed on the original SNP data over the unlinked regions of the genome. For example, in case of exome data one could make it over genes. Then when bootstrap is done, it is required to set the directory with it in the parameters file for CLAIC evaluation:

.. code-block:: none

   # param_file

   # Inform if SNP's are not linked
   Linked SNP's : True

   # Tell where bootstrapped data is located
   Directory with bootstrap: /home/dadi/examples/YRI_CEU/bootstraps/
   ...

.. note::
    This kind of bootstrap is called block-bootstrap and it is very important if one want to do some model selections for data with linked SNPs. **Please, be careful if it is your case**.
