VCF data format
****************

GADMA can load VCF file with SNP's and build SFS data from it for demographic inference.
VCF file should be set together with popmap file that describing how samples map to populations:

.. code-block:: none

   # param_file

   # Set VCF and popmap file
   Input data : path_to_vcf_file, path_to_popmap_file
   ...

VCF file
---------

.. tip::
   It is better to set options ``Sequence length`` and ``Mutation rate`` if they are known.
   Then units of  parameters of the demographic history will be in physical units.

.. note::
   Sometimes it is better to convert a VCF (.vcf) file into a SFS (.sfs) file.
   To do so use `easySFS <https://github.com/isaacovercast/easySFS>`_.

There are several conditions of successful reading of VCF file. SNP's will be
taken to SFS data if:

1. VCF file contains diploid samples;
2. `FILTER` column is equal to `PASS` or `.`;
3. `REF` and `ALT` columns are nucleotides (one letter from [`A`, `T`, `G`, `C`]);
4. When positions are repeated the last SNP will be taken.
5. Ancestral allele should be put to `INFO` column as `AA=` field.

.. _popmap_file:

Popmap file
------------

Popmap file keep information about how samples map the populations.
It is a plain-text with two tab-separated columns: sample name and corresponding population name. For example:

.. code-block:: none

    sample_1   Pop1
    sample_2   Pop1
    sample_3   Pop2
    sample_4   Pop2
    sample_5   Pop2

.. note::
    Popmap file can miss some samples from VCF file or have some extra - all of them will be ignored.

Examples
---------

In the following example of VCF file only one line will pass the restrictions of GADMA reading.
SNP on line 3 will be filtered out as FILTER column is equal to `q10` (should be `PASS` or `.`).
SNP's on lines 4, 5 and 6 has `REF` and/or `ALT` column with not-nucleotides.
And only the first SNP is okay.


.. code-block:: none
    :linenos:

    #CHROM	POS		ID			REF		ALT		QUAL	FILTER	INFO								FORMAT	  	NA00001			NA00002			NA00003
    20		14370	rs6054257	G		A		29		PASS	NS=3;DP=14;AF=0.5;DB;H2				GT:GQ:DP:HQ	0|0:48:1:51,51	1|0:48:8:51,51	1/1:43:5:.,.
    20		17330	.			T		A		3		q10 	NS=3;DP=11;AF=0.017					GT:GQ:DP:HQ	0|0:49:3:58,50	0|1:3:5:65,3	0/0:41:3
    20		1110696	rs6040355	A		G,T		67		PASS	NS=2;DP=10;AF=0.333,0.667;AA=T;DB	GT:GQ:DP:HQ	1|2:21:6:23,27	2|1:2:0:18,2	2/2:35:4
    20		1230237	.			T		.		47		PASS	NS=3;DP=13;AA=T						GT:GQ:DP:HQ	0|0:54:7:56,60	0|0:48:4:51,51	0/0:61:2
    20		1234567	microsat1	GTCT	G,GTACT	50		PASS	NS=3;DP=9;AA=G						GT:GQ:DP	0/1:35:4		0/2:17:2		1/1:40:3

Another example of VCF file with ancestral allele information, five SNP's will be successfully read (lines 2, 4, 7, 8, 9):

.. code-block:: none
    :linenos:

    #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	tsk_0	tsk_1	tsk_2	tsk_3	tsk_4	tsk_5
    1		 316	.	C	G	.	PASS	AA=C	GT	1|0	0|0	0|1	1|0	0|1	0|1
    1		1099	.	C	T	.	NOTPASS	AA=C	GT	1|0	0|0	0|1	1|0	0|1	0|1
    1		1338	.	C	T	.	PASS	AA=C	GT	0|0	0|0	0|0	0|0	1|0	1|0
    1		2276	.	0	1	.	PASS	AA=T	GT	1|0	0|0	0|1	1|0	0|1	0|1
    1		2889	.	0	1	.	NOTPASS	AA=0	GT	0|1	1|1	1|0	0|1	1|0	1|0
    1		3107	.	G	T	.	PASS	AA=G	GT	1|0	0|0	0|1	1|0	0|1	0|1
    1		3535	.	C	T	.	PASS	AA=C	GT	0|0	0|0	0|0	0|0	1|0	1|0
    1		3738	.	C	T	.	PASS	AA=C	GT	1|0	0|0	0|0	0|0	0|0	0|0
    1		3803	.	A	C	.	NOTPASS	AA=A	GT	0|0	0|0	0|0	0|0	0|1	0|0

And corresponding popmap file:

.. code-block:: none

    tsk_0   YRI
    tsk_1   YRI
    tsk_2   CEU
    tsk_3   CEU
    tsk_4   CEU
    tsk_5   CHB

