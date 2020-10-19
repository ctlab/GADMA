Hands on
==========

Test case
-----------

GADMA has a test case for a simple demographic model for one population: just a constant size of $10000$ individuals in population. To run a test case:

.. code-block:: console

   $ gadma --test

Example 2
------------

Suppose we have SNP data for two populations. Data is in dadi's SNP file format in the ``snp_data.txt``. Suppose we want to get all output in some ``gadma_output`` directory:


.. code-block:: console

   $ gadma -i snp_data.txt -o gadma_output


Example 3
-----------

We didn't specify AFS size or labels for populations, they are taken automatically from the input file. We can see a parameter file of our run in the ``gadma_output/param_file``.


.. code-block::

   # gadma_output/param_file
   ...
   Population labels : pop_name_1, pop_name_2
   Projections: 18, 20
   ...


But we know that spectrum should be ``20 x 20``! To specify size of AFS we need to create a parameter file and set ``Projections``:

.. code-block::

   # param_file
   Projections : 20, 20

Order of populations can be changed as:

.. code-block::

   # param_file
   Projections : 20,20
   Population labels : pop_name_2, pop_name_1

If we want to rename populations, we should change names on ``snp_data.txt`` file.

Now assume we want to get the simplest demographic model as fast as we can. We will tell GADMA that we need no other dynamics of population sizes except sudden (constant) population size change and that we want to use *moments* library.

.. code-block::

   # param_file
   Projections : 20,20
   Population labels : pop_name_2, pop_name_1
   Only sudden : True
   Engine : moments


To run GADMA we need to specify ``-p/--params`` command-line option in cmd:

.. code-block:: console

   $ gadma -i snp_data.txt -o gadma_output -p params_file

Example 4
-----------

Consider some AFS file ``fs_data.fs``. There is a spectrum for three populations: YRI, CEU, CHB. However axes are mixed up: CHB, YRI, CEU. To run GADMA we should order them from most ancient to most recent:

.. code-block::

   # param_file
   Population labels : YRI, CEU, CHB


We want to allow exponential growth (it is the default behaviour) and have some extra changes in size of the ancient population. To do so we should specify ``Initial structure``. It is list of three numbers: first - number of time intervals before first split (we want here 2); second - number of time periods between first and second split events (at least 1); third - number of time periods after second split.

.. code-block::

   # param_file
   Population labels : YRI, CEU, CHB
   Initial structure : 2,1,1

Also we can put information about input file and output directory to our parameter file:

.. code-block::

   # param_file
   Input file : fs_data.fs
   Output directory : gadma_output
   Population labels : YRI, CEU, CHB
   Initial structure : 2,1,1

Now we can run GADMA in the following way:

.. code-block:: console

   $ gadma -p params


Example 5
------------

We have our GADMA launch interrupted for some reasons. We want to resume it:

.. code-block:: console

   $ gadma --resume gadma_output

where ``gadma_output`` is output directory of previous run. We can find resumed run in ``gadma_output_resumed``


Example 6
-------------

Our launch was finished, we used dadi with a default grid size which GADMA determines automatically if it is not specified by user. We found out that it would be better to find some models using greater number of grid points in dadi scheme, but we want to take final models from previous run:

.. code-block::

   # param_file
   Pts : 40, 50, 60 #Greater value of grid size than it was


And run GADMA:

.. code-block:: console

   $ gadma --resume gadma_output --only_models -p params 


Option ``--only_models`` tells GADMA to take from ``gadma_output`` final models only.

There is another way to do the same:

.. code-block::

   # param_file
   Resume from : gadma_output
   Only models : True
   Pts : 40, 50, 60 #Greater value of grid size than it was

And run GADMA in the following way:

.. code-block:: console

   $ gadma -p params


Example 7
-----------

We can add a custom model using a parameter ``Custom filename`` in the parameter file:

.. code-block::

   # param_file
   Custom filename : YRI_CEU_demographic_model.py

Our custom file need to contain a function with a fixed name ``model_func``. For example:

.. code-block:: python

   # YRI_CEU_demographic_model.py
   def model_func(params, ns, pts)
       nu1F, nu2B, nu2F, m, Tp, T = params
       n1, n2 = ns
       xx = yy = dadi.Numerics.default_grid(pts)
   
       phi = dadi.PhiManip.phi_1D(xx)
       phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F)
   
       phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
       nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
       phi = dadi.Integration.two_pops(phi, xx, T, nu1=nu1F,
                                       nu2=nu2_func, m12=m, m21=m)
   
       sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,yy))
       return sfs

In addition, we can easily specify values for lower and upper bounds through a parameter file. Let's set lower and upper bounds for the model we defined above:

.. code-block::

   # param_file
   Lower bounds : 1e-2, 1e-2, 1e-2, 0, 0, 0
   Upper bounds : 100, 100, 100, 10, 3, 3

Example 8
------------

Also, we can get the values of lower/upper bounds, both, or none of them in the parameter file automatically. For this, each identifier in the parameter file must be declared through a parameter ``Parameter identifiers``. Below is an identifier list:

.. code-block::

   # param_file
   #   if identifier starts with letter:
   #   T/t - time
   #   N/n - size of population
   #   M/m - migration
   #   S/s/F/f - split event, proportion in which population size
   #             is divided to form two new populations.
   #   G/g - selection
   #   H/h - dominance coefficient for selection.

For example, we set a lower bound for the model we defined above (see Example 7) and we want to get an upper bound automatically.

.. code-block::

   # param_file
   Lower bound : 1e-2, 1e-2, 1e-2, 0, 0, 0
   Upper bound : None

   Parameter identifiers : nu1F, nu2B, nu2F, m, Tp, T

If the custom function has first line line ``par1, par2, ... = params`` then option ``Parameter identifiers`` could be missed and GADMA will take this option from the function.

Example YRI, CEU
-------------------

GADMA has an example of the parameter file ``example_params``. To run GADMA with this parameters one should just run from the GADMA's home directory:

.. code-block:: console

   $ gadma -p example_params
