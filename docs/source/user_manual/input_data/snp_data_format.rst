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

