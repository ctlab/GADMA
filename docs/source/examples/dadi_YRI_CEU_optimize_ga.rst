Function optimize\_ga with ∂a∂i for (YRI, CEU) inference
========================================================

It is example from `∂a∂i <https://bitbucket.org/gutenkunstlab/dadi/>`__.
In original example close to optimal parameters are pertrubed and then
local search (optimize\_log, optimize\_powell etc.) is launched.

In our modification of this example here we use global search - Genetic
Algorithm (optimize\_ga) from `GADMA <https://github.com/ctlab/GADMA>`__
software.

You can find original python code
`here <https://bitbucket.org/gutenkunstlab/dadi/src/master/examples/YRI_CEU/YRI_CEU.py>`__
(``dadi/examples/YRI_CEU/YRI_CEU.py`` file)

Imports
-------

.. code:: ipython3

    import dadi
    import gadma
    %matplotlib inline

Data
----

Data was build originally for paper `Gutenkunst et al.
2009 <https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695>`__.

It is 20x20 spectrum for two populations:

-  YRI - Yoruba in Ibadan, Nigeria
-  CEU - Utah Residents (CEPH) with Northern and Western European
   Ancestry

.. code:: ipython3

    data = dadi.Spectrum.from_file("YRI_CEU.fs")
    ns = data.sample_sizes
    print(f"Size of spectrum: {ns}")


.. parsed-literal::

    Size of spectrum: [20 20]


.. code:: ipython3

    dadi.Plotting.plot_single_2d_sfs(data, vmin=1.0)




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x7f93040f6c18>




.. image:: dadi_YRI_CEU_optimize_ga_files/dadi_YRI_CEU_optimize_ga_6_1.png


Grid points
-----------

∂a∂i needs grid points for evaluations:

.. code:: ipython3

    # These are the grid point settings will use for extrapolation.
    pts = [40,50,60]

Demographic model
-----------------

Demographic model is saved in ``demographic_models_dadi.py`` as
``model_func`` function. But we also put it here:

.. code:: ipython3

    import numpy
    import dadi
    
    def model_func(params, ns, pts):
        """
        Model with growth, split, bottleneck in pop2, exp recovery, migration
    
        nu1F: The ancestral population size after growth. (Its initial size is
              defined to be 1.)
        nu2B: The bottleneck size for pop2
        nu2F: The final size for pop2
        m: The scaled migration rate
        Tp: The scaled time between ancestral population growth and the split.
        T: The time between the split and present
    
        n1,n2: Size of fs to generate.
        pts: Number of points to use in grid for evaluation.
        """
    
        nu1F, nu2B, nu2F, m, Tp, T = params
        # Define the grid we'll use
        xx = yy = dadi.Numerics.default_grid(pts)
    
        # phi for the equilibrium ancestral population
        phi = dadi.PhiManip.phi_1D(xx)
        # Now do the population growth event.
        phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F)
    
        # The divergence
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        # We need to define a function to describe the non-constant population 2
        # size. lambda is a convenient way to do so.
        nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
        phi = dadi.Integration.two_pops(phi, xx, T, nu1=nu1F, nu2=nu2_func, 
                                        m12=m, m21=m)
    
        # Finally, calculate the spectrum.
        sfs = dadi.Spectrum.from_phi(phi, ns, (xx,yy))
        return sfs

We can import it from file by:

.. code:: ipython3

    from demographic_models_dadi import model_func as func

or use directly from our notebook:

.. code:: ipython3

    func = model_func

Inference
---------

Now we will infer parameters for this demographic history and loaded
data.

.. code:: ipython3

    # Now let's optimize parameters for this model.
    
    # The upper_bound and lower_bound lists are for use in optimization.
    # Occasionally the optimizer will try wacky parameter values. We in particular
    # want to exclude values with very long times, very small population sizes, or
    # very high migration rates, as they will take a long time to evaluate.
    # Parameters are: (nu1F, nu2B, nu2F, m, Tp, T)
    par_labels = ('nu1F', 'nu2B', 'nu2F', 'm', 'Tp', 'T')
    upper_bound = [100, 100, 100, 10, 3, 3]
    lower_bound = [1e-2, 1e-2, 1e-2, 0, 0, 0]
    
    # Make the extrapolating version of our demographic model function.
    func_ex = dadi.Numerics.make_extrap_log_func(func)
    
    # Run our optimization
    # For more information: help(gadma.Inference.optimize_ga)
    # It is test optimization so only 10 iterations of global optimization
    # (ga_maxiter) and 1 iteration of local (ls_maxiter) are run.
    # For better optimization set those number to greater values or to None.
    print('Beginning optimization ************************************************')
    result = gadma.Inference.optimize_ga(data=data,
                                         model_func=func_ex,
                                         engine='dadi',
                                         args=(pts,),
                                         p_ids = par_labels,
                                         lower_bound=lower_bound,
                                         upper_bound=upper_bound,
                                         local_optimizer='Powell_log',
                                         ga_maxiter=10,
                                         ls_maxiter=1)
    print('Finshed optimization **************************************************')


.. parsed-literal::

    Beginning optimization ************************************************
    --Start global optimization Genetic_algorithm--


.. parsed-literal::

    WARNING:Inference:Model is masked in some entries where data is not.
    WARNING:Inference:Number of affected entries is 238. Sum of data in those entries is 2998.58:


.. parsed-literal::

    2998.5798028627855 15684.69145058756
    Generation #0.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1701.826817	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.45752,	m=0.75996,	Tp=1.18187,	T=0.86968)	r
    1	-1729.977101	(nu1F=2.8046,	nu2B=0.03398,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=1.27484)	r
    2	-3429.804929	(nu1F=10.07386,	nu2B=2.30486,	nu2F=0.42512,	m=0.83356,	Tp=0.83459,	T=0.9573)	r
    3	-3553.704116	(nu1F=0.57483,	nu2B=0.08977,	nu2F=0.28265,	m=1.15827,	Tp=1.12788,	T=1.00575)	r
    4	-3584.960802	(nu1F=2.26739,	nu2B=3.09241,	nu2F=2.52953,	m=0.58835,	Tp=0.88652,	T=0.77567)	r
    5	-3675.524195	(nu1F=12.18778,	nu2B=3.09972,	nu2F=0.33028,	m=1.00913,	Tp=1.11375,	T=0.3531)	r
    6	-3792.617760	(nu1F=0.69454,	nu2B=0.68659,	nu2F=1.04326,	m=1.37447,	Tp=1.07923,	T=1.16278)	r
    7	-4259.763741	(nu1F=0.41307,	nu2B=0.28107,	nu2F=0.18045,	m=1.40936,	Tp=1.59377,	T=1.17211)	r
    8	-4300.191417	(nu1F=0.59633,	nu2B=1.9406,	nu2F=0.38285,	m=0.45428,	Tp=0.75365,	T=0.33004)	r
    9	-4316.308906	(nu1F=3.35194,	nu2B=8.90924,	nu2F=1.66733,	m=1.2133,	Tp=1.10813,	T=0.85285)	r
    Current mean mutation rate:	 0.200000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1701.8268172514881
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.45752,	m=0.75996,	Tp=1.18187,	T=0.86968)	r
    
    
    Generation #1.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1701.826817	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.45752,	m=0.75996,	Tp=1.18187,	T=0.86968)	r
    1	-1729.977101	(nu1F=2.8046,	nu2B=0.03398,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=1.27484)	r
    2	-1760.864950	(nu1F=2.8046,	nu2B=2.06258,	nu2F=0.45752,	m=0.75996,	Tp=0.54352,	T=0.86968)	c
    3	-2264.177426	(nu1F=5.83626,	nu2B=0.13663,	nu2F=0.48804,	m=1.42326,	Tp=1.08219,	T=0.87912)	r
    4	-3063.650364	(nu1F=9.15027,	nu2B=2.33581,	nu2F=0.33028,	m=1.23806,	Tp=1.11375,	T=0.3531)	mmm
    5	-3148.592837	(nu1F=10.07386,	nu2B=2.30486,	nu2F=0.53811,	m=0.83356,	Tp=0.83459,	T=1.02132)	mm
    6	-3319.821496	(nu1F=2.8046,	nu2B=0.03398,	nu2F=0.33028,	m=1.00913,	Tp=1.11375,	T=1.27484)	c
    7	-3766.686457	(nu1F=2.30421,	nu2B=3.09241,	nu2F=2.52953,	m=0.75996,	Tp=0.88652,	T=0.77567)	c
    8	-3930.025618	(nu1F=0.59633,	nu2B=1.9406,	nu2F=0.38285,	m=0.58574,	Tp=0.75365,	T=0.33004)	m
    9	-12840.236662	(nu1F=0.2805,	nu2B=0.10287,	nu2F=0.50801,	m=0.36103,	Tp=0.98415,	T=1.5125)	r
    Current mean mutation rate:	 0.240000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1701.8268172514881
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.45752,	m=0.75996,	Tp=1.18187,	T=0.86968)	r
    
    Mean time:	4.705 sec.
    
    
    
    Generation #2.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1421.483902	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    1	-1648.737614	(nu1F=2.8046,	nu2B=2.06258,	nu2F=0.56888,	m=0.75996,	Tp=0.54352,	T=1.27484)	c
    2	-1701.826817	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.45752,	m=0.75996,	Tp=1.18187,	T=0.86968)	r
    3	-1712.040031	(nu1F=2.70395,	nu2B=2.06258,	nu2F=0.45752,	m=0.75996,	Tp=1.03441,	T=0.86968)	mm
    4	-1729.977101	(nu1F=2.8046,	nu2B=0.03398,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=1.27484)	r
    5	-2521.636599	(nu1F=7.01144,	nu2B=0.13663,	nu2F=0.48804,	m=1.42326,	Tp=1.08219,	T=0.87912)	m
    6	-3098.933038	(nu1F=9.15027,	nu2B=2.33581,	nu2F=0.38926,	m=1.23806,	Tp=1.33106,	T=0.3531)	mm
    7	-7024.808187	(nu1F=10.07386,	nu2B=0.10287,	nu2F=0.53811,	m=0.36103,	Tp=0.98415,	T=1.5125)	c
    8	-13573.782932	(nu1F=0.81134,	nu2B=3.28175,	nu2F=27.41468,	m=0,	Tp=0.79988,	T=0.63039)	r
    9	-30960.850332	(nu1F=3.42775,	nu2B=0.07771,	nu2F=2.06107,	m=0,	Tp=1.62673,	T=1.38412)	r
    Current mean mutation rate:	 0.288000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1421.4839020176794
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    
    Mean time:	4.617 sec.
    
    
    
    Generation #3.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1421.483902	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    1	-1577.591980	(nu1F=1.49263,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	m
    2	-1597.190567	(nu1F=2.8046,	nu2B=2.06258,	nu2F=0.56888,	m=0.75996,	Tp=0.54352,	T=0.86968)	c
    3	-1648.737614	(nu1F=2.8046,	nu2B=2.06258,	nu2F=0.56888,	m=0.75996,	Tp=0.54352,	T=1.27484)	c
    4	-3210.674478	(nu1F=9.15027,	nu2B=2.33581,	nu2F=0.38926,	m=0.36103,	Tp=0.98415,	T=0.3531)	c
    5	-4112.075815	(nu1F=7.87171,	nu2B=7.32061,	nu2F=0.21993,	m=1.00734,	Tp=0.54731,	T=1.03704)	r
    6	-5989.011686	(nu1F=6.25357,	nu2B=0.10287,	nu2F=0.53811,	m=0.36103,	Tp=0.98415,	T=1.5125)	m
    7	-7202.271523	(nu1F=10.07386,	nu2B=0.10287,	nu2F=0.53811,	m=0.36103,	Tp=0.98415,	T=1.32037)	m
    8	-7484.427556	(nu1F=10.07386,	nu2B=0.10287,	nu2F=0.56888,	m=0.36103,	Tp=0.98415,	T=0.86968)	c
    9	-76056.242622	(nu1F=0.11937,	nu2B=4.6946,	nu2F=0.08804,	m=0,	Tp=1.30773,	T=1.08575)	r
    Current mean mutation rate:	 0.275168
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1421.4839020176794
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    
    Mean time:	4.946 sec.
    
    
    
    Generation #4.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1421.483902	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    1	-1481.222311	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=1.45384,	Tp=0.54352,	T=0.86968)	mm
    2	-1489.444836	(nu1F=1.49263,	nu2B=2.06258,	nu2F=0.42053,	m=1.29198,	Tp=0.54352,	T=1.17575)	mm
    3	-1577.591980	(nu1F=1.49263,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	m
    4	-1680.835446	(nu1F=1.49263,	nu2B=2.06258,	nu2F=0.56888,	m=1.00734,	Tp=0.54731,	T=1.03704)	c
    5	-3326.596980	(nu1F=10.07386,	nu2B=0.10287,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    6	-4086.691541	(nu1F=7.87171,	nu2B=7.32061,	nu2F=0.21993,	m=1.00734,	Tp=0.3989,	T=1.03704)	m
    7	-9743.703382	(nu1F=45.92041,	nu2B=16.86401,	nu2F=4.64628,	m=0,	Tp=0.7852,	T=1.19254)	r
    8	-11479.011689	(nu1F=10.07386,	nu2B=0.10287,	nu2F=0.21993,	m=0.36103,	Tp=0.54731,	T=0.86968)	c
    9	-11637.931727	(nu1F=28.0868,	nu2B=12.61298,	nu2F=0.16764,	m=0,	Tp=1.26555,	T=0.73011)	r
    Current mean mutation rate:	 0.262907
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1421.4839020176794
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    
    Mean time:	4.958 sec.
    
    
    
    Generation #5.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1421.483902	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    1	-1481.222311	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=1.45384,	Tp=0.54352,	T=0.86968)	mm
    2	-1622.650743	(nu1F=1.49263,	nu2B=0.10287,	nu2F=0.42053,	m=1.29198,	Tp=0.54731,	T=1.17575)	c
    3	-1784.525601	(nu1F=1.49263,	nu2B=2.06258,	nu2F=0.38616,	m=1.00734,	Tp=0.54731,	T=1.03704)	m
    4	-2704.144441	(nu1F=6.88227,	nu2B=0.10287,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	m
    5	-4111.566184	(nu1F=7.87171,	nu2B=7.32061,	nu2F=0.21993,	m=1.00734,	Tp=0.54352,	T=1.03704)	c
    6	-6430.579304	(nu1F=1.49263,	nu2B=0.10287,	nu2F=0.42053,	m=0.36103,	Tp=0.54352,	T=0.86968)	c
    7	-9184.126127	(nu1F=3.47701,	nu2B=6.0622,	nu2F=0.19563,	m=0,	Tp=1.74888,	T=0.85791)	r
    8	-10133.145838	(nu1F=10.07386,	nu2B=0.10287,	nu2F=0.28776,	m=0.36103,	Tp=0.54731,	T=0.86968)	m
    9	-50698.797605	(nu1F=0.2162,	nu2B=0.24522,	nu2F=0.78623,	m=0,	Tp=1.18292,	T=1.05393)	r
    Current mean mutation rate:	 0.251192
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1421.4839020176794
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    
    Mean time:	5.052 sec.
    
    
    
    Generation #6.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1421.483902	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    1	-1481.222311	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=1.45384,	Tp=0.54352,	T=0.86968)	mm
    2	-6199.437036	(nu1F=1.49263,	nu2B=0.10287,	nu2F=0.42053,	m=0.36103,	Tp=0.54352,	T=0.52632)	m
    3	-6430.579304	(nu1F=1.49263,	nu2B=0.10287,	nu2F=0.42053,	m=0.36103,	Tp=0.54352,	T=0.86968)	c
    4	-6640.844060	(nu1F=6.88227,	nu2B=0.10287,	nu2F=0.56888,	m=0.36103,	Tp=0.54352,	T=0.86968)	c
    5	-8399.863314	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=0,	Tp=0.54352,	T=0.85791)	c
    6	-8832.053052	(nu1F=3.47701,	nu2B=7.33239,	nu2F=0.19563,	m=0,	Tp=1.74888,	T=0.85791)	m
    7	-9146.318921	(nu1F=4.05271,	nu2B=6.0622,	nu2F=0.19563,	m=0,	Tp=1.74888,	T=0.85791)	m
    8	-10454.658375	(nu1F=0.84381,	nu2B=10.16508,	nu2F=0.56506,	m=0,	Tp=1.43428,	T=0.71472)	r
    9	-78224.057665	(nu1F=0.57119,	nu2B=0.09234,	nu2F=0.05643,	m=0,	Tp=1.06285,	T=0.99956)	r
    Current mean mutation rate:	 0.240000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1421.4839020176794
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    
    Mean time:	4.796 sec.
    
    
    
    Generation #7.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1379.451604	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.40953,	m=1.69576,	Tp=0.54352,	T=0.86968)	mm
    1	-1421.483902	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    2	-1481.222311	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=1.45384,	Tp=0.54352,	T=0.86968)	mm
    3	-3443.242136	(nu1F=6.55397,	nu2B=1.64121,	nu2F=0.23611,	m=0.79503,	Tp=1.01632,	T=0.47258)	r
    4	-4238.287643	(nu1F=13.82932,	nu2B=0.52445,	nu2F=1.76065,	m=1.12997,	Tp=0.88447,	T=0.63745)	r
    5	-6199.437036	(nu1F=1.49263,	nu2B=0.10287,	nu2F=0.42053,	m=0.36103,	Tp=0.54352,	T=0.52632)	c
    6	-6341.457145	(nu1F=6.88227,	nu2B=0.10287,	nu2F=0.56888,	m=0.36103,	Tp=0.54352,	T=1.1935)	m
    7	-9146.318921	(nu1F=4.05271,	nu2B=6.0622,	nu2F=0.19563,	m=0,	Tp=1.74888,	T=0.85791)	c
    8	-10749.801271	(nu1F=0.65262,	nu2B=10.16508,	nu2F=0.56506,	m=0,	Tp=1.43428,	T=0.56536)	mmm
    9	-19462.131920	(nu1F=0.57119,	nu2B=10.16508,	nu2F=0.56506,	m=0,	Tp=1.06285,	T=0.99956)	c
    Current mean mutation rate:	 0.288000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1379.451603685885
    Solution:		(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.40953,	m=1.69576,	Tp=0.54352,	T=0.86968)	mm
    
    Mean time:	4.537 sec.
    
    
    
    Generation #8.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1332.756094	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	m
    1	-1379.451604	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.40953,	m=1.69576,	Tp=0.54352,	T=0.86968)	mm
    2	-1421.483902	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.56888,	m=1.29198,	Tp=0.54352,	T=0.86968)	c
    3	-1446.202441	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=1.45384,	Tp=0.54352,	T=1.1935)	c
    4	-2335.006192	(nu1F=6.55397,	nu2B=1.64121,	nu2F=0.56888,	m=0.79503,	Tp=0.54352,	T=0.47258)	c
    5	-3864.588131	(nu1F=13.82932,	nu2B=0.32695,	nu2F=1.76065,	m=1.12997,	Tp=1.08406,	T=0.51967)	mmm
    6	-4350.986386	(nu1F=1.00225,	nu2B=1.98231,	nu2F=1.51719,	m=0.62344,	Tp=0.73584,	T=0.77814)	r
    7	-11844.684249	(nu1F=4.05271,	nu2B=2.06258,	nu2F=0.19563,	m=0,	Tp=1.74888,	T=0.85791)	c
    8	-13112.320124	(nu1F=1.19806,	nu2B=1.39403,	nu2F=19.89957,	m=0,	Tp=0.9426,	T=0.9661)	r
    9	-19462.131920	(nu1F=0.57119,	nu2B=10.16508,	nu2F=0.56506,	m=0,	Tp=1.06285,	T=0.99956)	mmmmm
    Current mean mutation rate:	 0.345600
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1332.756093859653
    Solution:		(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	m
    
    Mean time:	4.328 sec.
    
    
    
    Generation #9.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1332.756094	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	m
    1	-1356.743441	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.40953,	m=1.12997,	Tp=1.08406,	T=0.51967)	c
    2	-1377.570320	(nu1F=1.9016,	nu2B=1.46395,	nu2F=0.56888,	m=1.45384,	Tp=0.54352,	T=1.1935)	m
    3	-1379.451604	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.40953,	m=1.69576,	Tp=0.54352,	T=0.86968)	mm
    4	-1409.806963	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=1.28727,	Tp=0.54352,	T=1.1935)	m
    5	-1427.591237	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.55688,	m=1.45384,	Tp=0.54352,	T=1.23065)	mm
    6	-1488.529556	(nu1F=1.9016,	nu2B=0.32695,	nu2F=0.44017,	m=1.12997,	Tp=1.08406,	T=0.51967)	c
    7	-3388.058628	(nu1F=1.9016,	nu2B=2.06258,	nu2F=1.51719,	m=1.45384,	Tp=0.54352,	T=0.77814)	c
    8	-8041.568942	(nu1F=1.95672,	nu2B=0.6205,	nu2F=4.85963,	m=0,	Tp=1.35314,	T=0.91228)	r
    9	-8214.541951	(nu1F=3.26254,	nu2B=1.69449,	nu2F=0.44299,	m=0,	Tp=0.2379,	T=0.81994)	r
    Current mean mutation rate:	 0.330201
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1332.756093859653
    Solution:		(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	m
    
    Mean time:	4.448 sec.
    
    
    
    Generation #10.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1332.756094	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	m
    1	-1335.409202	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.66314,	T=0.86968)	m
    2	-1356.743441	(nu1F=2.30421,	nu2B=2.06258,	nu2F=0.40953,	m=1.12997,	Tp=1.08406,	T=0.51967)	c
    3	-1371.555405	(nu1F=1.9016,	nu2B=1.46395,	nu2F=0.56888,	m=1.45384,	Tp=0.26909,	T=1.1935)	m
    4	-1444.528889	(nu1F=1.9016,	nu2B=1.46395,	nu2F=0.44017,	m=1.12997,	Tp=0.54352,	T=1.1935)	c
    5	-1446.202441	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.56888,	m=1.45384,	Tp=0.54352,	T=1.1935)	c
    6	-1637.447995	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.7285,	m=1.28727,	Tp=0.54352,	T=1.1935)	m
    7	-4578.167648	(nu1F=5.71242,	nu2B=0.34508,	nu2F=15.81412,	m=0,	Tp=0.65176,	T=0.63848)	r
    8	-5580.242741	(nu1F=3.94175,	nu2B=0.97538,	nu2F=4.41883,	m=0,	Tp=1.64821,	T=1.03995)	r
    9	-10564.016593	(nu1F=3.26254,	nu2B=0.6205,	nu2F=0.44299,	m=0,	Tp=1.35314,	T=0.81994)	c
    Current mean mutation rate:	 0.315488
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1332.756093859653
    Solution:		(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	m
    
    Mean time:	4.465 sec.
    
    
    
    --Finish global optimization Genetic_algorithm--
    Result:
       status: 1
     success: True
     message: OPTIMIZATION IS NOT STOPPED
           x: [1.9015956483379328 2.06257938805023 0.4401664517805921 1.45383785541038
     0.5435242533018835 0.8696827003939265]
           y: -1332.756093859653
      n_eval: 154
      n_iter: 10
    
    --Start local optimization optimize_log_powell--
    1	-1332.756093859653	(nu1F=1.9016,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    2	-2028.0661250483067	(nu1F=5.16907,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    3	-4898.459097385603	(nu1F=0.37706,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    4	-1810.533699560487	(nu1F=1.02497,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    5	-1433.348153974162	(nu1F=2.78615,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    6	-1331.2814040134494	(nu1F=2.02714,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    7	-1330.8914116934352	(nu1F=1.98867,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    8	-1330.8905075545727	(nu1F=1.98685,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    9	-1330.8906282307707	(nu1F=1.98598,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    10	-1330.8907466170276	(nu1F=1.98772,	nu2B=2.06258,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    11	-1493.8639276695478	(nu1F=1.98685,	nu2B=5.60667,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    12	-1258.422056937624	(nu1F=1.98685,	nu2B=0.40899,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    13	-1245.529008909171	(nu1F=1.98685,	nu2B=0.55927,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    14	-1254.7026644593452	(nu1F=1.98685,	nu2B=0.92069,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    15	-1244.9797708249203	(nu1F=1.98685,	nu2B=0.67657,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    16	-1244.5673649674582	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    17	-1244.5749257264883	(nu1F=1.98685,	nu2B=0.62068,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    18	-1244.5802436046356	(nu1F=1.98685,	nu2B=0.63562,	nu2F=0.44017,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    19	-2111.2380189441797	(nu1F=1.98685,	nu2B=0.6281,	nu2F=1.1965,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    20	-6307.093905715057	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.08728,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    21	-2116.1078046047132	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.23725,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    22	-1297.5281461620343	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.64492,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    23	-1216.4731192580962	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50947,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    24	-1216.40638111635	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50711,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    25	-1216.395948956531	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    26	-1216.3983018971778	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50509,	m=1.45384,	Tp=0.54352,	T=0.86968)	
    27	-2816.280367294807	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=3.95194,	Tp=0.54352,	T=0.86968)	
    28	-4413.191785940151	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=0.28828,	Tp=0.54352,	T=0.86968)	
    29	-1807.5156600216585	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=0.78362,	Tp=0.54352,	T=0.86968)	
    30	-1454.6104282471256	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=2.13011,	Tp=0.54352,	T=0.86968)	
    31	-1216.5164931025915	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.44462,	Tp=0.54352,	T=0.86968)	
    32	-1216.3828121502381	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45754,	Tp=0.54352,	T=0.86968)	
    33	-1216.3737651390086	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45812,	Tp=0.54352,	T=0.86968)	
    34	-1251.1253085831238	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.68526,	Tp=0.54352,	T=0.86968)	
    35	-1221.4720803645298	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.54102,	Tp=0.54352,	T=0.86968)	
    36	-1217.136792008684	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.48925,	Tp=0.54352,	T=0.86968)	
    37	-1216.5009769324383	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.46993,	Tp=0.54352,	T=0.86968)	
    38	-1216.3868222564906	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.46262,	Tp=0.54352,	T=0.86968)	
    39	-1216.3857536100047	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45997,	Tp=0.54352,	T=0.86968)	
    40	-1216.383717663772	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45869,	Tp=0.54352,	T=0.86968)	
    41	-1216.3681534921243	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=0.54352,	T=0.86968)	
    42	-1216.3797027301853	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45786,	Tp=0.54352,	T=0.86968)	
    43	-1216.3785601896823	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45799,	Tp=0.54352,	T=0.86968)	
    44	-1234.2589240883358	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.47745,	T=0.86968)	
    45	-1207.801950905503	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=0.10777,	T=0.86968)	
    46	-1208.7540304472927	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=0.13962,	T=0.86968)	
    47	-1204.217295391792	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=0.00786,	T=0.86968)	
    48	-1204.4649664537692	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=0.0139,	T=0.86968)	
    49	-1203.8955946917079	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=0.00011,	T=0.86968)	
    50	-1203.9239532891347	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=0.00077,	T=0.86968)	
    51	-1203.8906918264106	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.20e-07,	T=0.86968)	
    52	-1203.8908379423583	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=3.50e-06,	T=0.86968)	
    53	-1203.8906866459101	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.83e-12,	T=0.86968)	
    54	-1203.890686666229	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=4.66e-10,	T=0.86968)	
    55	-1203.8906866456591	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.95e-20,	T=0.86968)	
    56	-1203.8906866456623	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.32e-16,	T=0.86968)	
    57	-1203.8906866456587	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=7.25e-33,	T=0.86968)	
    58	-1203.890686645661	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.42e-26,	T=0.86968)	
    59	-1203.8906866456532	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.86968)	
    60	-1203.8906866456578	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.93e-15,	T=0.86968)	
    61	-1203.8906866456591	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.07e-17,	T=0.86968)	
    62	-1203.890686645661	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.58e-16,	T=0.86968)	
    63	-1203.8906866456591	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=5.42e-17,	T=0.86968)	
    64	-1203.8906866456587	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=1.92e-17,	T=0.86968)	
    65	-1246.5689058605644	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=2.36404)	
    66	-1565.1957441363834	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.17245)	
    67	-1184.6639581710049	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.46876)	
    68	-1219.6162433763666	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.31994)	
    69	-1186.4370918740005	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.56242)	
    70	-1184.7607271707827	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.46495)	
    71	-1184.4456559875418	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.49189)	
    72	-1184.8114774728335	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.51772)	
    73	-1184.4413687280955	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.48864)	
    74	-1184.446161555542	(nu1F=1.98685,	nu2B=0.6281,	nu2F=0.50579,	m=1.45808,	Tp=2.79e-17,	T=0.48583)	
    --Finish local optimization optimize_log_powell--
    Result:
       status: 2
     success: False
     message: GLOBAL OPTIMIZATION: OPTIMIZATION IS NOT STOPPED; LOCAL OPTIMIZATION: Maximum number of iterations has been exceeded.
           x: [1.9868528725580605 0.6281033050618842 0.5057936159267086
     1.4580765859361928 2.7945929091077765e-17 0.4886353404838611]
           y: -1184.4413687280955
      n_eval: 228
      n_iter: 11
    
    Finshed optimization **************************************************


.. code:: ipython3

    popt = result.x
    print(f'Found parameters: {popt}')
    print(f'With log-likelihood: {result.y}')
    
    # Now we can compare our parameters with those that were obtained before:
    print('\nFrom Gutenkunst et al 2009:')
    # These are the actual best-fit model parameters, which we found through
    # longer optimizations and confirmed by running multiple optimizations.
    # We'll work with them through the rest of this script. 
    popt = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]
    print('Best-fit parameters: {0}'.format(popt))
    
    # Calculate the best-fit model AFS.
    model = func_ex(popt, ns, pts)
    # Likelihood of the data given the model AFS.
    ll_model = dadi.Inference.ll_multinom(model, data)
    print('Maximum log composite likelihood: {0}'.format(ll_model))
    # The optimal value of theta given the model.
    theta = dadi.Inference.optimal_sfs_scaling(model, data)
    print('Optimal value of theta: {0}'.format(theta))


.. parsed-literal::

    Found parameters: [1.9868528725580605 0.6281033050618842 0.5057936159267086
     1.4580765859361928 2.7945929091077765e-17 0.4886353404838611]
    With log-likelihood: -1184.4413687280955
    
    From Gutenkunst et al 2009:
    Best-fit parameters: [1.881, 0.071, 1.845, 0.911, 0.355, 0.111]
    Maximum log composite likelihood: -1066.3460755934125
    Optimal value of theta: 2749.285796480555


Plotting
--------

Now we could draw some plots for model with best parameters.

.. code:: ipython3

    # Plot a comparison of the resulting fs with the data.
    import pylab
    pylab.figure(1)
    dadi.Plotting.plot_2d_comp_multinom(model, data, vmin=1, resid_range=3,
                                        pop_ids =('YRI','CEU'), show=True)



.. image:: dadi_YRI_CEU_optimize_ga_files/dadi_YRI_CEU_optimize_ga_21_0.png


