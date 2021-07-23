.obs (fastsimcoal2) data format
********************************

SFS holding format native to fastsimcoal2. Can be generated with fastsimcoal2 or easySFS software. See fastsimcoal2 manual for examples.

General
-------

The first line of each file starts with the number of observations ``N`` presented in the file. It can be followed with any text after a whitespace. The text is ignored.

If the FS is of minor allele (its filename contains ``MAF`` or ``MSFS``), corresponding elements of the FS are masked with ``0``.

One populaion (_{M,D}AFpop{X}.obs)
----------------------------------

``X`` should be integer value.

The second line has tab separeted columns names of one-dimensional FS (``dX_0	dX_1	...``)

Each of the next ``N`` lines contains values for single observed FS; values can be tab- or space-separeted.

Two populations (_joint{M,D}AFpop{X}_{Y}.obs)
----------------------------------------------

``X`` and ``Y`` should be integer values.

Consider FS generated from data with 2 populations where the first one (i.e. population ``X``)  has ``A`` samples and the second (population ``Y``)  has ``B`` samples.

The second line then has ``B + 1`` tab separeted columns names for the second dimension (i.e. the second population): ``dY_0	dY_1	...``)

Then there are ``N`` blocks containing ``A+1`` lines; each line has a line name (``dX_0``, ``dX_1`` etc.) and ``B+1`` tab- or space-separated values. Value at the crossing of ``dY_j`` and ``dX_i`` indicates observed number of sites in which there are ``i`` derived (or minor) alleles in population ``X`` and ``j`` derived (or minor) alleles in population Y in current observation.

Multiple observations (_{M,D}SFS.obs)
-------------------------------------
The second line contains number of demes ``A`` and ``A`` integers - population sizes for each deme.

Each of the next ``N`` lines contain values of multidimensional FS separeted by spaces or tabs.
The order of values follows is the one used in ``dadi`` format. For example, for 3-population FS the order is as follows: ``fs[0,0,0] fs[0,0,1] ... fs[0,1,0] fs[0,1,1]``

