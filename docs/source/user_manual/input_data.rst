Input data
=============

GADMA supports several types of input data, which are familiar to anyone who has used ``dadi`` or ``moments`` in the past:

   * Frequency spectrum file format (.fs or .sfs),
   * SNP data format (.txt).

Input file can be specified to GADMA in two ways:

1) Through command-line option ``-i/--input``:

   .. code-block:: console

      $ gadma -i fs_file.fs -o out_dir

   or

   .. code-block:: console

      $ gadma --input snp_file.txt -o out_dir

2) Use a parameter ``Input file`` in the parameter file:

   .. code-block:: none

      # param_file

      # Input file path
      Input file : fs_file.fs 
      ...

Extra information about input AFS can also be put in the parameter file. For example, AFS can be projected to a smaller size with ``Projections`` option, populations can be named or their order can be changed with the ``Population labels`` option. Option ``Outgroup`` tells whether data has outgroup or not: if ``Outgroup`` is ``False`` then the spectrum will be folded. If the parameter file does not contain some options, they are automatically pulled out from the input file. Also length of sequence could be set by the ``Sequence length`` option, which could be used along with ``Mutation rate`` instead of ``Theta0``.

.. code-block:: none

   # param_file

   # Input file path
   Input file : fs_file.fs
    
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

By default, SNP's that were used to build AFS are considered to be linked. In this case it is possible to compare demographic models with different number of parameters by Composite Likelihood Akaike Information Criterion (CLAIC). This procedure can be necessary as a model with a large number of parameters will be better able to find parameter values corresponding to the observed data than a model with a smaller number of parameters, but at the same time it will correspond less to reality, for example, due to data errors. It is called overfitting and we do not want it to happen.

Actually, CLAIC is modification of usual Akaike Information Criterion (AIC), but AIC can be used only for AFS built from unlinked SNP's. The smaller the AIC or CLAIC score is, the better the model fits the observed data.

It is possible to inform GADMA about linkage of SNP's and unlock the usage of AIC:

.. code-block:: none

   # param_file

   # Inform if SNP's are not linked
   Linked SNP's : False
   ...

If SNP's are linked and CLAIC should be evaluated (by default it is not), then the bootstrapped data should be set. In order to receive reliable correct bootstrapped data, the bootstrap should be performed on the original SNP data over the unlinked regions of the genome. For example, in case of exome data one could make it over genes. Then when bootstrap is done, it is required to set the directory with it in the parameters file for CLAIC evaluation:

.. code-block:: none

   # param_file

   # Inform if SNP's are not linked
   Linked SNP's : True

   # Tell where bootstrapped data is located
   Directory with bootstrap: /home/dadi/examples/YRI_CEU/bootstraps/

   ...

This kind of bootstrap is called block-bootstrap and it is very important if one want to do some model selections for data with linked SNPs. **Please, be careful if it is your case**.

Data formats
--------------

VCF data format
******************

To convert a VCF (.vcf) file into a SFS (.sfs) file use `easySFS <https://github.com/isaacovercast/easySFS>`_.


Frequency spectrum file format
********************************

Each file begins with any number of comment lines starting with ``#``.
The first non-comment line contains ``P`` integers giving the dimensions of the FS array, where ``P`` is the number of represented populations represented.
For an FS representing data from ``4x4x2`` samples, this would be ``5x5x3``.
(Each dimension is one larger than the number of samples because the number of observations can range, for example, from 0 to 4 if there are 4 samples, for a total of 5 possibilities.)
On the same line, the string ``folded`` or ``unfolded`` denotes whether or not the stored FS is folded.

The actual data is stored in a single line listing all the FS elements separated by spaces, in the order ``fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1]...``.
This is followed by a single line giving the elements of the mask in the same order as the data, with ``1`` indicating masked and ``0`` indicating unmasked.

SNP data format
****************

Example of a file in the SNP format:

.. code-block:: none

   Human Chimp Allele1 YRI  CEU  Allele2 YRI CEU Gene  Position
   ACG   ATG   C       29   24   T       1   0   abcb1 289
   CCT   CCT   C       29   23   G       3   2   abcb1 345


The data file begins with any number of comment lines that start with ``#``.
The first parsed line is a column header line.
Whitespace is used to separate entries within the table, so no spaces are allowed within any entry.
Individual rows maybe commented out using ``#``.

The first column contains the in-group reference sequence at that SNP, including the flanking bases.
If the flanking bases are unknown, they can be denoted by a hyphen (``-``).
The header label is arbitrary.

The second column contains the aligned outgroup reference sequence at that SNP, including the flanking bases.
Unknown entries can be denoted by ``-``.
The header label is arbitrary.

The third column gives the first segregating allele.
The column header must be exactly ``Allele1``.

Then follows an arbitrary number of columns, one for each population, each giving the number of times ``Allele1`` was observed in that population.
The header for each column should be the population identifier.

The next column gives the second segregating allele.
The column header must be exactly ``Allele2``.

Then follows one column for each population, each giving the number of times Allele2 was observed in that population.
The header for each column should be the population identifier, and the columns should be in the same order as for the Allele1 entries.

Then follows an arbitrary number of columns which will be concatenated with ``_`` to assign a label for each SNP.

The ``Allele1`` and ``Allele2`` headers must be exactly those values because the number of columns between those two is used to infer the number of populations in the file.

