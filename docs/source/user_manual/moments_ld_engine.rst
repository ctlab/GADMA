.. _moments_ld_engine:

Input data
-------------------
momentsLD work with data stored in VCF format and require population map and recombination map
(if available).

VCF file must contain all chromosomes you want to use in the analysis. If you have few separated
VCF files you can concat them using bcftools or similar programs. GADMA takes only one single VCF file as input.

momentsLD estimate LD statistics binning observed SNP pairs of SNP by recombination distances.
It can bin pairs by physical distances, but genetic maps are non-uniform and do not perfectly correlate
with recombination distances. When working with momentsLD It is recommended to use recombination maps.
A recombination map - is a genetic map that measures the probability of crossing over at each position
in the genome. As default GADMA takes recombination maps in cM(centiMorgans).

Details about additional parameters for momentsLD engine and how it works are given below.


Regions and bed files generation
--------------------
GADMA has a function for the auto-generation of bed files. This bed files subset chromosomes
to equal regions. You can specify region length in the param file.

.. code-block::

    # param file
    ...
    region_len: 6400000
    ...

Take in mind that 15 is the minimum quantity of regions for computing LD statistics from data.
If you pass in a length parameter that causes less than 15 regions to be received, it will be ignored.
In the case of fewer regions, we will get a singular matrix during computing LD statistics and subsequent
computation will be impossible. The larger the number of regions, the lower the noise
level in the data. But at the same time, do not forget about the balance between the number of regions and the
length of each individual region. Regions that are too small are also not suitable for work.

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

In Structure Demographic Model and Custom Demographic Model GADMA by default evaluate the ancestral size
as a parameter of the demographic model.
The value of this parameter will be calculated as another parameter and used in theta and rhos transformation.

In Custom Demographic Model you can use fixed ancestral size. It will not be used as
a model parameter and will always only use the specified value. To fix ancestral size update your params file
like that:

.. code-block::

    # param file
    ...
    fixed_ancestral_size: 10000
    ...


In momentsLD engine ``Theta`` and ``Ancestral population size`` (effective population size) differ
a little from what we see in moments or dadi with ASF.

Theta also differs from moments or dadi and calculate ``4 * Ne * mu``.
You can skip theta0 and sequence length in param files, but please do not forget to specify the mutation rate.

LD keyword arguments
--------------------
momentsLD engine takes several arguments used in computing LD stats.
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

You can find more information about these arguments in the original documentation of momentsLD

As default GADMA works with recombination maps with cM units and VCF files containing unphased data.

Plotting LD curves
-------------------
GADMA saves all graphs of LD statistics. You can find them in the output directory.

.. image:: exampl_ld_curves.png
    :width: 10%

In the generated code you can find code for LD curves plotting and information about label preparation.
It will help you to plot only the curves you need.

Precomputing data
-------------------
Parsing LD statistics from an input VCF file is a time-consuming process which is not a main part of the
GADMA genetic algorithm evaluation. If you start GADMA several times, it will spend a lot of time parsing
LD statistics from data. In this case, GADMA has the option of precomputing data before the main process starts.

``gadma-precompute_ld_data`` script reads, precomputes, and saves precomputed data.
Use this script with the same parameters file as always and GADMA will automatically start data parsing using
a number of processes specified in the parameters file. GADMA will save precomputed data in binary format for further work
and update parameters file.


How to use your own precomputed data
-------------------
You can precompute data on your own using moments.LD library opportunities. For correct GADMA work, you should save
dictionary received after ``moments.LD.Parsing.compute_ld_statistics`` in binary using ``pickle`` library.
GADMA will read statistics and bootstrap regions from this file.

.. code-block::

    # param file
    Input data : ./some.vcf, ./some_popmap
    ...
    preprocessed_data: ./preprocessed_data.bp
    ...

For correct GADMA work please specify any VCF file and population map if you use precomputed data. They will not
be used, but GADMA needs them for correct work.
