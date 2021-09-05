.. _moments_ld_engine:

Input data
-------------------
moments LD work with data stored in VCF format and require population map and recombination map
(if available).

moments.LD estimate LD statistics binning observed SNP pairs of SNP by recombination distances.
It can bin pairs by physical distances, but genetic maps are non-uniform and do not perfectly correlate
with recombination distances. When working with moments.LD It is recommended to use recombination maps.
Recombination map - is a genetic map that measures the probability of crossing over at each position
in the genome. As default GADMA takes recombination maps in cM(centiMorgans).

When you use moments.LD as a GADMA engine you have some additional parameters in the parameter file.


Recombination maps
-----
Recombination maps - is a directory containing a recombination map for each
chromosome in a VCF file. Rename the recombination maps according to the example.

.. code-block::

    # param file
    ...
    recombinations_maps: ./some_dir
    ...

.. code-block::

    {map}_{chrom_name}.txt
    Example:
    rec_map_chrom1.txt
    rec_map_chrom2.txt


If you have a few recombination maps united into one file,
each map should have a head name according to the chromosome.


Ancestral size and Theta
-------------------
In moments.LD engine ``Theta`` and ``Ancestral population size`` (effective population size) differ
a little from what we see in moments or dadi with ASF.
Ancestral size is used as one of the model parameters.
If you use moments.LD engine:

.. code-block::

    # param file
    ...
    ancestral_size_as_parameter: True
    ...

Theta also differ from moments or dadi and calculate ``4 * Ne * mu``.
You can skip theta0 and sequence length in param files, but please specify mutation rate.

LD keyword arguments
--------------------
moments.LD engine takes several arguments using in computing LD stats.
All of these parameters have default values

Default LD kwargs:

.. code-block::

    r_bins : np.logspace(-6, -3, 7)
    report: False,
    bp_bins: np.array([ii for ii in range(0, 8275250, 1655050)]),
    use_genotypes: True,
    cM: True

If you want to change some of these arguments you can add to the parameter file ``ld_kwargs``.

.. code-block::

    # param file
    ...
    ld_kwargs: {“r_bins”: “np.logspace(-6, -3, 7)”, “report”: True}
    ...

Expressions must be enclosed in ““.

You can find more information about these arguments in the original documentation of moments.LD

As default GADMA works with recombination maps with cM units and VCF files containing unphased data.

Plotting LD curves
-------------------
GADMA saves all graphs of LD statistics. You can find them in the output directory.

.. image:: exampl_ld_curves.png
    :width: 10%

In the generated code you can find code for LD curves plotting and information about label preparation.
It will help you to plot only the curves you need.
