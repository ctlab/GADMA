Fastsimcoal2 data format (.obs)
********************************

SFS holding format native to fastsimcoal2. Can be generated with fastsimcoal2 or easySFS software. See fastsimcoal2 manual for more examples.

General
-------

The first line of each file starts with the number of observations ``N`` presented in the file. It can be followed with any text after a whitespace. The text is ignored.

If the FS is of minor allele (its filename contains ``MAF`` or ``MSFS``), corresponding elements of the FS are masked with ``0``.

One populaion (_{M,D}AFpop{X}.obs)
----------------------------------

``X`` should be integer value that represents population number.

The second line has tab separeted columns names of one-dimensional FS (``dX_0	dX_1	...``)

Each of the next ``N`` lines contains values for single observed FS; values can be tab- or space-separeted.

Examples
^^^^^^^^

SFS of a derived allele for single population (``YRI_DAFpop0.obs``)

.. code-block:: none

        1 observation
        d0_0    d0_1    d0_2    d0_3    d0_4
        21671 25734 10899 6520 2388

SFS of a minor allele for single population (``YRI_MAFpop0.obs``)

.. code-block:: none

        1 observation
        d0_0    d0_1    d0_2    d0_3    d0_4
        24059 32254 10899 0 0

Two populations (_joint{M,D}AFpop{X}_{Y}.obs)
----------------------------------------------

``X`` and ``Y`` should be integer values that represent population numbers.

Consider FS generated from data with 2 populations where the first one (i.e. population ``X``)  has ``A`` samples and the second (population ``Y``)  has ``B`` samples.

The second line then has ``B + 1`` tab separeted columns names for the second dimension (i.e. the second population): ``dY_0	dY_1	...``)

Then there are ``N`` blocks containing ``A+1`` lines; each line has a line name (``dX_0``, ``dX_1`` etc.) and ``B+1`` tab- or space-separated values. Value at the crossing of ``dY_j`` and ``dX_i`` indicates observed number of sites in which there are ``i`` derived (or minor) alleles in population ``X`` and ``j`` derived (or minor) alleles in population Y in current observation.

Examples
^^^^^^^^
2 observations of a joint SFS of derived allele for two populations: ``0`` with 4 samples and ``1`` with 5 samples (``MO_jointDAFpop1_0.obs``)

.. code-block:: none

        2 observation
        	d0_0	d0_1	d0_2	d0_3	d0_4
        d1_0	0 15463 2757 588 84
        d1_1	12163 2670 1269 439 111
        d1_2	3934 2041 1188 543 163
        d1_3	2476 1572 1168 554 212
        d1_4	1393 1214 983 652 314
        d1_5	830 966 949 629 377
        d1_0	0 15463 2757 588 84
        d1_1	12163 2670 1269 439 111
        d1_2	3934 2041 1188 543 163
        d1_3	2476 1572 1168 554 212
        d1_4	1393 1214 983 652 314
        d1_5	830 966 949 629 377

SFS of minor allele for two populations with 8 and 4 samples (``jointMAFpop1_0.pbs``)

.. code-block:: none

	1 observation
		d0_0	d0_1	d0_2	d0_3	d0_4
	d1_0	0 17097 3650 992 224
	d1_1	12833 3445 2083 1034 361
	d1_2	4391 2747 2066 1352 324
	d1_3	2853 2201 2117 760 0
	d1_4	1707 1866 983 0 0
	d1_5	1042 760 0 0 0
	d1_6	324 0 0 0 0
	d1_7	0 0 0 0 0
	d1_8	0 0 0 0 0


Multiple populations (_{M,D}SFS.obs)
-------------------------------------
The second line contains number of demes ``A`` and ``A`` integers - population sizes for each deme.

Each of the next ``N`` lines contain values of multidimensional FS separeted by spaces or tabs.
The order of values follows is the one used in ``dadi`` format. For example, for 3-population FS the order is as follows: ``fs[0,0,0] fs[0,0,1] ... fs[0,1,0] fs[0,1,1]``

Examples
^^^^^^^^

Multidimensional SFS of a derived allele for 3 populations having 4 samples each (``DSFS.obs``)

.. code-block:: none

	1 observations. No. of demes and sample sizes are on next line.
	3	4 4 4
	0 6749 917 119 10 5414 2100 1001 349 54 917 1163 661 253 92 193 347 429 225 71 26 94 168 179 140 15463 1430 505 140 17 1240 1118 668 277 102 418 632 625 345 159 132 283 405 438 260 12 114 212 335 404 2757 663 329 102 21 606 612 483 230 107 247 467 473 373 180 116 249 372 460 437 10 97 238 377 893 588 227 143 47 1 212 264 210 160 62 136 241 319 261 128 56 159 258 386 369 13 48 192 406 1634 84 65 53 24 2 46 73 104 87 27 37 71 158 158 113 13 66 170 239 280 1 22 105 390 0

Multidimensional SFS of a minor allele for 3 populations having 4 samples each (``MSFS.obs``)

.. code-block:: none

	1 observations. No. of demes and sample sizes are on next line.
	3	4 4 4
	0 7139 1022 141 11 5694 2339 1171 415 67 1030 1321 819 324 64.5 220 434 533 149 0 28 118 110.5 0 0 17097 1836 697 188 30 1609 1504 926 436 79 546 893 944 293 0 194 443 307.5 0 0 13 80.5 0 0 0 3650 1040 567 199 15.5 1043 1072 855 239.5 0 427 840 473 0 0 223 239.5 0 0 0 15.5 0 0 0 0 992 562 355 80.5 0 472 702 307.5 0 0 295 293 0 0 0 79 0 0 0 0 0 0 0 0 0 224 244 110.5 0 0 117 149 0 0 0 64.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0

