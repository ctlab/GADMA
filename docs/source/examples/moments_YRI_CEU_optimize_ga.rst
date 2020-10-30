Function optimize\_ga with *moments* for (YRI, CEU) inference
=============================================================

It is example from
`moments <https://bitbucket.org/simongravel/moments/>`__. In original
example close to optimal parameters are pertrubed and then local search
(optimize\_log, optimize\_powell etc.) is launched.

In our modification of this example here we use global search - Genetic
Algorithm (optimize\_ga) from `GADMA <https://github.com/ctlab/GADMA>`__
software.

You can find original python code
`here <https://bitbucket.org/simongravel/moments/src/master/examples/YRI_CEU/YRI_CEU.py>`__
(``moments/examples/YRI_CEU/YRI_CEU.py`` file)

Imports
-------

.. code:: ipython3

    import moments
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

    data = moments.Spectrum.from_file("YRI_CEU.fs")
    ns = data.sample_sizes
    print(f"Size of spectrum: {ns}")


.. parsed-literal::

    Size of spectrum: [20 20]


.. code:: ipython3

    moments.Plotting.plot_single_2d_sfs(data, vmin=1.0)



.. image:: moments_YRI_CEU_optimize_ga_files/moments_YRI_CEU_optimize_ga_6_0.png


Demographic model
-----------------

Demographic model is saved in ``demographic_models_moments.py`` as
``model_func`` function. But we also put it here:

.. code:: ipython3

    import numpy
    import moments
    
    def model_func(params, ns):
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
        # f for the equilibrium ancestral population
        sts = moments.LinearSystem_1D.steady_state_1D(ns[0]+ns[1])
        fs = moments.Spectrum(sts)
    
        
        # Now do the population growth event.
        fs.integrate([nu1F], Tp)
        # The divergence
        fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
        # We need to define a function to describe the non-constant population 2
        # size. lambda is a convenient way to do so.
        nu2_func = lambda t: [nu1F, nu2B*(nu2F/nu2B)**(t/T)]
        fs.integrate(nu2_func, T, m=numpy.array([[0, m],[m, 0]]))
    
        return fs

We can import it from file by:

.. code:: ipython3

    from demographic_models_moments import model_func as func

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
    
    # Run our optimization
    # For more information: help(gadma.Inference.optimize_ga)
    # It is test optimization so only 10 iterations of global optimization
    # (ga_maxiter) and 1 iteration of local (ls_maxiter) are run.
    # For better optimization set those number to greater values or to None.
    print('Beginning optimization ************************************************')
    result = gadma.Inference.optimize_ga(data=data,
                                         model_func=func,
                                         engine='moments',
                                         args=(),
                                         p_ids = par_labels,
                                         lower_bound=lower_bound,
                                         upper_bound=upper_bound,
                                         local_optimizer='BFGS_log',
                                         ga_maxiter=10,
                                         ls_maxiter=1)
    print('Finshed optimization **************************************************')


.. parsed-literal::

    Beginning optimization ************************************************
    --Start global optimization Genetic_algorithm--
    Generation #0.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1987.052205	(nu1F=3.43862,	nu2B=2.89219,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	r
    1	-2234.646741	(nu1F=3.50412,	nu2B=0.13566,	nu2F=0.34019,	m=1.39445,	Tp=0.6516,	T=1.26159)	r
    2	-3284.832003	(nu1F=2.36307,	nu2B=10.48031,	nu2F=0.15888,	m=0.90435,	Tp=1.40333,	T=0.77398)	r
    3	-3949.781543	(nu1F=27.61818,	nu2B=1.91526,	nu2F=0.37061,	m=1.48645,	Tp=0.77768,	T=1.62237)	r
    4	-4094.822922	(nu1F=6.88753,	nu2B=0.53698,	nu2F=4.57133,	m=0.742,	Tp=0.51412,	T=1.22419)	r
    5	-4471.930371	(nu1F=6.58858,	nu2B=0.20111,	nu2F=0.25936,	m=1.01793,	Tp=1.33963,	T=1.05038)	r
    6	-4482.327581	(nu1F=2.07049,	nu2B=3.27477,	nu2F=4.70106,	m=0,	Tp=0.43373,	T=0.24935)	r
    7	-4566.605335	(nu1F=8.27895,	nu2B=4.02306,	nu2F=2.46973,	m=0.60475,	Tp=1.18486,	T=0.79746)	r
    8	-4627.880832	(nu1F=3.15245,	nu2B=4.30329,	nu2F=2.9816,	m=0,	Tp=1.27727,	T=0.79689)	r
    9	-4814.204612	(nu1F=0.56577,	nu2B=1.21377,	nu2F=1.0236,	m=1.63148,	Tp=0.94572,	T=1.40603)	r
    Current mean mutation rate:	 0.200000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1987.0522053640375
    Solution:		(nu1F=3.43862,	nu2B=2.89219,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	r
    
    
    Generation #1.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1987.052205	(nu1F=3.43862,	nu2B=2.89219,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	r
    1	-2234.646741	(nu1F=3.50412,	nu2B=0.13566,	nu2F=0.34019,	m=1.39445,	Tp=0.6516,	T=1.26159)	r
    2	-3242.410755	(nu1F=7.73078,	nu2B=3.56599,	nu2F=0.31896,	m=1.03176,	Tp=1.27683,	T=1.43487)	r
    3	-3420.133914	(nu1F=3.50412,	nu2B=0.53698,	nu2F=4.57133,	m=0.742,	Tp=0.6516,	T=1.26159)	c
    4	-3967.672073	(nu1F=7.06096,	nu2B=3.05846,	nu2F=2.46973,	m=0.60475,	Tp=1.18486,	T=0.79746)	mm
    5	-4439.905351	(nu1F=22.25187,	nu2B=1.91526,	nu2F=0.24767,	m=1.48645,	Tp=0.77768,	T=1.62237)	mm
    6	-4482.327581	(nu1F=2.07049,	nu2B=3.27477,	nu2F=4.70106,	m=0,	Tp=0.43373,	T=0.24935)	c
    7	-4566.605335	(nu1F=8.27895,	nu2B=4.02306,	nu2F=2.46973,	m=0.60475,	Tp=1.18486,	T=0.79746)	c
    8	-4648.069388	(nu1F=3.03553,	nu2B=4.30329,	nu2F=2.9816,	m=0.00e+00,	Tp=1.12279,	T=0.75905)	mmmm
    9	-10116.700949	(nu1F=0.33253,	nu2B=1.04749,	nu2F=1.01133,	m=0.65527,	Tp=0.96124,	T=1.15813)	r
    Current mean mutation rate:	 0.240000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1987.0522053640375
    Solution:		(nu1F=3.43862,	nu2B=2.89219,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	r
    
    Mean time:	1.438 sec.
    
    
    
    Generation #2.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1985.779939	(nu1F=4.64104,	nu2B=0.14456,	nu2F=1.291,	m=1.20103,	Tp=0.72781,	T=1.05714)	r
    1	-1987.052205	(nu1F=3.43862,	nu2B=2.89219,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	r
    2	-1999.488386	(nu1F=3.43862,	nu2B=2.09585,	nu2F=1.27423,	m=0.88786,	Tp=0.72851,	T=1.64976)	mm
    3	-2234.646741	(nu1F=3.50412,	nu2B=0.13566,	nu2F=0.34019,	m=1.39445,	Tp=0.6516,	T=1.26159)	r
    4	-2646.187746	(nu1F=7.73078,	nu2B=3.56599,	nu2F=0.40538,	m=1.1536,	Tp=1.27683,	T=1.43487)	mm
    5	-3251.586303	(nu1F=7.73078,	nu2B=3.56599,	nu2F=0.31896,	m=1.03176,	Tp=1.27683,	T=1.78654)	m
    6	-3967.672073	(nu1F=7.06096,	nu2B=3.05846,	nu2F=2.46973,	m=0.60475,	Tp=1.18486,	T=0.79746)	c
    7	-5627.140413	(nu1F=2.07049,	nu2B=3.27477,	nu2F=4.57133,	m=0.742,	Tp=0.43373,	T=1.26159)	c
    8	-9524.027742	(nu1F=6.3693,	nu2B=1.24507,	nu2F=2.39034,	m=0,	Tp=1.084,	T=1.65784)	r
    9	-13989.816500	(nu1F=0.33253,	nu2B=1.04749,	nu2F=2.9816,	m=0.65527,	Tp=1.12279,	T=1.15813)	c
    Current mean mutation rate:	 0.288000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1985.7799393124901
    Solution:		(nu1F=4.64104,	nu2B=0.14456,	nu2F=1.291,	m=1.20103,	Tp=0.72781,	T=1.05714)	r
    
    Mean time:	1.752 sec.
    
    
    
    Generation #3.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1663.525411	(nu1F=3.43862,	nu2B=1.24507,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    1	-1985.779939	(nu1F=4.64104,	nu2B=0.14456,	nu2F=1.291,	m=1.20103,	Tp=0.72781,	T=1.05714)	r
    2	-1987.052205	(nu1F=3.43862,	nu2B=2.89219,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	r
    3	-2114.344901	(nu1F=1.03383,	nu2B=2.54967,	nu2F=0.30822,	m=1.37369,	Tp=0.85999,	T=1.71988)	r
    4	-2234.646741	(nu1F=3.50412,	nu2B=0.13566,	nu2F=0.34019,	m=1.39445,	Tp=0.6516,	T=1.26159)	c
    5	-2608.936199	(nu1F=7.73078,	nu2B=2.43015,	nu2F=0.40538,	m=1.1536,	Tp=1.27683,	T=1.43487)	m
    6	-8124.265720	(nu1F=6.3693,	nu2B=1.24507,	nu2F=2.39034,	m=0,	Tp=0.69147,	T=1.34897)	mm
    7	-10793.186574	(nu1F=0.33253,	nu2B=1.04749,	nu2F=1.27423,	m=0.65527,	Tp=0.72851,	T=1.15813)	c
    8	-12939.312322	(nu1F=0.33253,	nu2B=1.04749,	nu2F=2.24041,	m=0.65527,	Tp=1.12279,	T=1.15813)	m
    9	-96742.964941	(nu1F=0.09123,	nu2B=13.99824,	nu2F=0.67339,	m=0,	Tp=0.6985,	T=1.06168)	r
    Current mean mutation rate:	 0.345600
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1663.5254110392593
    Solution:		(nu1F=3.43862,	nu2B=1.24507,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    
    Mean time:	1.735 sec.
    
    
    
    Generation #4.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1663.525411	(nu1F=3.43862,	nu2B=1.24507,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    1	-1985.779939	(nu1F=4.64104,	nu2B=0.14456,	nu2F=1.291,	m=1.20103,	Tp=0.72781,	T=1.05714)	r
    2	-1987.052205	(nu1F=3.43862,	nu2B=2.89219,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    3	-2448.164133	(nu1F=4.20643,	nu2B=0.13566,	nu2F=0.34019,	m=1.39445,	Tp=0.6516,	T=1.26159)	m
    4	-4335.594328	(nu1F=3.52615,	nu2B=0.0472,	nu2F=0.25474,	m=1.06699,	Tp=0.89919,	T=0.89453)	r
    5	-4902.715633	(nu1F=6.3693,	nu2B=1.24507,	nu2F=3.40643,	m=0,	Tp=0.69147,	T=0.86596)	mm
    6	-4934.522031	(nu1F=0.33253,	nu2B=1.04749,	nu2F=0.30822,	m=1.37369,	Tp=0.72851,	T=1.15813)	c
    7	-4995.031704	(nu1F=0.33253,	nu2B=1.04749,	nu2F=0.34019,	m=1.39445,	Tp=0.72851,	T=1.26159)	c
    8	-10987.532249	(nu1F=1.26215,	nu2B=5.27405,	nu2F=1.05231,	m=0,	Tp=1.20941,	T=1.05955)	r
    9	-11772.353532	(nu1F=0.33253,	nu2B=0.47262,	nu2F=2.24041,	m=0.65527,	Tp=0.65116,	T=1.15813)	mm
    Current mean mutation rate:	 0.330201
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1663.5254110392593
    Solution:		(nu1F=3.43862,	nu2B=1.24507,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    
    Mean time:	1.682 sec.
    
    
    
    Generation #5.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1597.687926	(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.27423,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    1	-1663.525411	(nu1F=3.43862,	nu2B=1.24507,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    2	-1985.779939	(nu1F=4.64104,	nu2B=0.14456,	nu2F=1.291,	m=1.20103,	Tp=0.72781,	T=1.05714)	r
    3	-2453.652390	(nu1F=3.43862,	nu2B=1.24507,	nu2F=2.24041,	m=0.65527,	Tp=0.72851,	T=1.15813)	c
    4	-2515.310931	(nu1F=3.43862,	nu2B=5.27405,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.05955)	c
    5	-2545.396501	(nu1F=4.20643,	nu2B=0.10516,	nu2F=0.34019,	m=1.39445,	Tp=0.6516,	T=1.26159)	m
    6	-3359.041195	(nu1F=10.10477,	nu2B=2.67777,	nu2F=0.71518,	m=1.01601,	Tp=1.54076,	T=0.58929)	r
    7	-3422.167407	(nu1F=3.52615,	nu2B=0.0472,	nu2F=0.33999,	m=1.06699,	Tp=0.89919,	T=0.89453)	m
    8	-11603.959998	(nu1F=0.33253,	nu2B=0.47262,	nu2F=2.24041,	m=0.65527,	Tp=0.41536,	T=1.15813)	m
    9	-38494.309072	(nu1F=1.46343,	nu2B=0.03006,	nu2F=2.85164,	m=0,	Tp=0.99627,	T=0.85137)	r
    Current mean mutation rate:	 0.396241
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1597.6879257305714
    Solution:		(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.27423,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    
    Mean time:	1.767 sec.
    
    
    
    Generation #6.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1597.687926	(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.27423,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    1	-1663.525411	(nu1F=3.43862,	nu2B=1.24507,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    2	-1931.836224	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=1.39445,	Tp=0.72851,	T=1.26159)	c
    3	-1996.307308	(nu1F=4.64104,	nu2B=0.21664,	nu2F=1.291,	m=1.20103,	Tp=0.72781,	T=1.05714)	m
    4	-2522.765587	(nu1F=3.52615,	nu2B=5.27405,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.05955)	c
    5	-2531.702673	(nu1F=3.43862,	nu2B=5.27405,	nu2F=1.27423,	m=0.73591,	Tp=0.6516,	T=1.05955)	c
    6	-2640.415835	(nu1F=4.32674,	nu2B=5.27405,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.05955)	m
    7	-5306.442435	(nu1F=1.45793,	nu2B=0.39465,	nu2F=0.24297,	m=0.58745,	Tp=0.90613,	T=1.38515)	r
    8	-11672.705897	(nu1F=0.33253,	nu2B=0.47262,	nu2F=2.24041,	m=0.65527,	Tp=0.50409,	T=1.15813)	m
    9	-36729.679941	(nu1F=0.03946,	nu2B=3.09023,	nu2F=2.10606,	m=0.8169,	Tp=1.11355,	T=1.21844)	r
    Current mean mutation rate:	 0.378586
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1597.6879257305714
    Solution:		(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.27423,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    
    Mean time:	1.885 sec.
    
    
    
    Generation #7.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1523.930595	(nu1F=3.43862,	nu2B=0.0472,	nu2F=0.94865,	m=1.06699,	Tp=0.89919,	T=1.64976)	m
    1	-1597.687926	(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.27423,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    2	-1663.525411	(nu1F=3.43862,	nu2B=1.24507,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.64976)	c
    3	-1790.894462	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=1.39445,	Tp=0.72851,	T=0.76911)	m
    4	-2515.310931	(nu1F=3.43862,	nu2B=5.27405,	nu2F=1.27423,	m=0.73591,	Tp=0.72851,	T=1.05955)	c
    5	-2993.586871	(nu1F=3.43862,	nu2B=0.10516,	nu2F=2.24041,	m=1.39445,	Tp=0.50409,	T=1.26159)	c
    6	-4321.808193	(nu1F=3.43862,	nu2B=1.24507,	nu2F=0.24297,	m=0.73591,	Tp=0.90613,	T=1.64976)	c
    7	-30626.942399	(nu1F=0.04939,	nu2B=3.09023,	nu2F=1.13642,	m=0.8169,	Tp=1.70968,	T=1.21844)	mmm
    8	-35323.431586	(nu1F=0.92815,	nu2B=0.2353,	nu2F=0.48838,	m=0,	Tp=1.00142,	T=1.40864)	r
    9	-164418.367383	(nu1F=0.03477,	nu2B=1.40775,	nu2F=0.87201,	m=0,	Tp=0.73566,	T=0.74641)	r
    Current mean mutation rate:	 0.454303
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1523.9305953803782
    Solution:		(nu1F=3.43862,	nu2B=0.0472,	nu2F=0.94865,	m=1.06699,	Tp=0.89919,	T=1.64976)	m
    
    Mean time:	1.947 sec.
    
    
    
    Generation #8.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1433.698137	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	c
    1	-1523.930595	(nu1F=3.43862,	nu2B=0.0472,	nu2F=0.94865,	m=1.06699,	Tp=0.89919,	T=1.64976)	m
    2	-1540.854547	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    3	-1597.687926	(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.27423,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    4	-1792.477972	(nu1F=2.51022,	nu2B=0.04947,	nu2F=1.06469,	m=1.08229,	Tp=1.48334,	T=0.55765)	r
    5	-3042.188829	(nu1F=3.43862,	nu2B=0.13257,	nu2F=2.24041,	m=1.39445,	Tp=0.50409,	T=1.26159)	m
    6	-3291.036580	(nu1F=3.43862,	nu2B=5.27405,	nu2F=1.83067,	m=0.73591,	Tp=0.72851,	T=1.05955)	m
    7	-4319.493496	(nu1F=3.43862,	nu2B=1.24507,	nu2F=0.24297,	m=0.73591,	Tp=0.50409,	T=1.64976)	c
    8	-44952.010421	(nu1F=1.11858,	nu2B=0.49652,	nu2F=0.09486,	m=0,	Tp=1.78423,	T=1.2218)	r
    9	-228968.677901	(nu1F=0.03477,	nu2B=1.40775,	nu2F=0.87201,	m=0,	Tp=1.13039,	T=1.09572)	mm
    Current mean mutation rate:	 0.545164
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1433.6981366143343
    Solution:		(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	c
    
    Mean time:	1.931 sec.
    
    
    
    Generation #9.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1433.698137	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	c
    1	-1523.930595	(nu1F=3.43862,	nu2B=0.0472,	nu2F=0.94865,	m=1.06699,	Tp=0.89919,	T=1.64976)	m
    2	-1786.054774	(nu1F=1.70483,	nu2B=0.04947,	nu2F=1.06469,	m=1.08229,	Tp=1.48334,	T=0.55765)	m
    3	-1935.172762	(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.83067,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    4	-1941.771825	(nu1F=2.51022,	nu2B=0.08549,	nu2F=0.6051,	m=1.08229,	Tp=1.48334,	T=0.55765)	mm
    5	-7817.280977	(nu1F=16.94158,	nu2B=1.1129,	nu2F=2.24661,	m=0,	Tp=1.04069,	T=1.10915)	r
    6	-24611.921104	(nu1F=0.57072,	nu2B=8.74634,	nu2F=2.30423,	m=0,	Tp=0.08922,	T=1.29197)	r
    7	-32301.829825	(nu1F=1.11858,	nu2B=0.10516,	nu2F=1.27423,	m=0,	Tp=0.89919,	T=1.2218)	c
    8	-40957.277307	(nu1F=3.43862,	nu2B=0.49652,	nu2F=0.09486,	m=0,	Tp=1.70968,	T=1.2218)	c
    9	-44919.104695	(nu1F=1.11858,	nu2B=0.49652,	nu2F=0.09486,	m=0,	Tp=2.55412,	T=1.2218)	m
    Current mean mutation rate:	 0.520873
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1433.6981366143343
    Solution:		(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	c
    
    Mean time:	1.870 sec.
    
    
    
    Generation #10.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	-1433.698137	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	c
    1	-1523.930595	(nu1F=3.43862,	nu2B=0.0472,	nu2F=0.94865,	m=1.06699,	Tp=0.89919,	T=1.64976)	m
    2	-1523.930595	(nu1F=3.43862,	nu2B=0.0472,	nu2F=0.94865,	m=1.06699,	Tp=0.89919,	T=1.64976)	c
    3	-1566.988930	(nu1F=1.44261,	nu2B=0.62142,	nu2F=0.53705,	m=0.95052,	Tp=1.00765,	T=0.57655)	r
    4	-1671.759587	(nu1F=3.43862,	nu2B=0.0472,	nu2F=1.4189,	m=1.06699,	Tp=0.89919,	T=1.64976)	m
    5	-5717.397850	(nu1F=0.57072,	nu2B=0.0472,	nu2F=2.30423,	m=1.06699,	Tp=0.08922,	T=1.64976)	c
    6	-10618.701551	(nu1F=9.15494,	nu2B=1.48685,	nu2F=0.44471,	m=0,	Tp=1.06041,	T=0.99997)	r
    7	-18102.886873	(nu1F=0.57072,	nu2B=8.74634,	nu2F=2.30423,	m=0.00e+00,	Tp=0.08922,	T=0.85389)	mm
    8	-40952.635766	(nu1F=4.49249,	nu2B=0.49652,	nu2F=0.09486,	m=0,	Tp=1.70968,	T=1.2218)	m
    9	-40957.277307	(nu1F=3.43862,	nu2B=0.49652,	nu2F=0.09486,	m=0,	Tp=1.70968,	T=1.2218)	c
    Current mean mutation rate:	 0.497664
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: -1433.6981366143343
    Solution:		(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	c
    
    Mean time:	1.801 sec.
    
    
    
    --Finish global optimization Genetic_algorithm--
    Result:
       status: 1
     success: True
     message: OPTIMIZATION IS NOT STOPPED
           x: [3.4386201357914534 0.10516033779256068 1.2742316394276632
     0.8169029394972911 1.7096824153217636 1.2184355538385445]
           y: -1433.6981366143343
      n_eval: 152
      n_iter: 10
    
    --Start local optimization optimize_log--
    1	-1433.6981366143345	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	
    2	-1433.698142851489	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	
    3	-1433.6981332863768	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	
    4	-1433.698134333416	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	
    5	-1433.6981273073666	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	
    6	-1433.6981343335756	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	
    7	-1433.6981312747257	(nu1F=3.43862,	nu2B=0.10516,	nu2F=1.27423,	m=0.8169,	Tp=1.70968,	T=1.21844)	
    8	-2609.809724968078	(nu1F=2.13735,	nu2B=0.13553,	nu2F=1.51624,	m=1.6608,	Tp=2.03437,	T=1.8306)	
    9	-2609.8097231430033	(nu1F=2.13735,	nu2B=0.13553,	nu2F=1.51624,	m=1.6608,	Tp=2.03437,	T=1.8306)	
    10	-2609.809727972948	(nu1F=2.13735,	nu2B=0.13553,	nu2F=1.51624,	m=1.6608,	Tp=2.03437,	T=1.8306)	
    11	-2609.8097595038416	(nu1F=2.13735,	nu2B=0.13553,	nu2F=1.51624,	m=1.6608,	Tp=2.03437,	T=1.8306)	
    12	-2609.8097680132205	(nu1F=2.13735,	nu2B=0.13553,	nu2F=1.51624,	m=1.6608,	Tp=2.03437,	T=1.8306)	
    13	-2609.809724473851	(nu1F=2.13735,	nu2B=0.13553,	nu2F=1.51624,	m=1.6608,	Tp=2.03437,	T=1.8306)	
    14	-2609.809733259382	(nu1F=2.13735,	nu2B=0.13553,	nu2F=1.51624,	m=1.6608,	Tp=2.03437,	T=1.8306)	
    15	-1348.6547333289695	(nu1F=3.14721,	nu2B=0.11025,	nu2F=1.31617,	m=0.9323,	Tp=1.76595,	T=1.3144)	
    16	-1348.6547371517413	(nu1F=3.14721,	nu2B=0.11025,	nu2F=1.31617,	m=0.9323,	Tp=1.76595,	T=1.3144)	
    17	-1348.6547317333543	(nu1F=3.14721,	nu2B=0.11025,	nu2F=1.31617,	m=0.9323,	Tp=1.76595,	T=1.3144)	
    18	-1348.6547378448718	(nu1F=3.14721,	nu2B=0.11025,	nu2F=1.31617,	m=0.9323,	Tp=1.76595,	T=1.3144)	
    19	-1348.6547359228352	(nu1F=3.14721,	nu2B=0.11025,	nu2F=1.31617,	m=0.9323,	Tp=1.76595,	T=1.3144)	
    20	-1348.6547317207942	(nu1F=3.14721,	nu2B=0.11025,	nu2F=1.31617,	m=0.9323,	Tp=1.76595,	T=1.3144)	
    21	-1348.6547322110655	(nu1F=3.14721,	nu2B=0.11025,	nu2F=1.31617,	m=0.9323,	Tp=1.76595,	T=1.3144)	
    --Finish local optimization optimize_log--
    Result:
       status: 1
     success: False
     message: GLOBAL OPTIMIZATION: OPTIMIZATION IS NOT STOPPED; LOCAL OPTIMIZATION: Maximum number of iterations has been exceeded.
           x: [3.147213338395963 0.11024832837019047 1.3161713345388721
     0.9323024229304987 1.7659504105567985 1.3143963844420017]
           y: -1348.6547333289695
      n_eval: 173
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
    model = func(popt, ns)
    # Likelihood of the data given the model AFS.
    ll_model = moments.Inference.ll_multinom(model, data)
    print('Maximum log composite likelihood: {0}'.format(ll_model))
    # The optimal value of theta given the model.
    theta = moments.Inference.optimal_sfs_scaling(model, data)
    print('Optimal value of theta: {0}'.format(theta))


.. parsed-literal::

    Found parameters: [3.147213338395963 0.11024832837019047 1.3161713345388721
     0.9323024229304987 1.7659504105567985 1.3143963844420017]
    With log-likelihood: -1348.6547333289695
    
    From Gutenkunst et al 2009:
    Best-fit parameters: [1.881, 0.071, 1.845, 0.911, 0.355, 0.111]
    Maximum log composite likelihood: -1066.8225223685229
    Optimal value of theta: 2742.0151382905815


Plotting
--------

Now we could draw some plots for model with best parameters.

.. code:: ipython3

    # Plot a comparison of the resulting fs with the data.
    import pylab
    pylab.figure(1)
    moments.Plotting.plot_2d_comp_multinom(model, data, vmin=1, resid_range=3,
                                        pop_ids =('YRI','CEU'), show=True)



.. image:: moments_YRI_CEU_optimize_ga_files/moments_YRI_CEU_optimize_ga_19_0.png


.. code:: ipython3

    # Now that we've found the optimal parameters, we can use ModelPlot to
    # automatically generate a graph of our determined model.
    
    # First we generate the model by passing in the demographic function we used,
    # and the optimal parameters determined for it.
    model = moments.ModelPlot.generate_model(func, popt, ns)
    
    # Next, we plot the model. See ModelPlot.py for more information on the various
    # parameters that can be passed to the plotting function. In this case, we scale
    # the model to have an original starting population of size 11293 and a
    # generation time of 29 years. Results are saved to YRI_CEU_model.png.
    moments.ModelPlot.plot_model(
        model,
        save_file=None,
        fig_title="YRI CEU Example Model",
        pop_labels=["YRI", "CEU"],
        nref=11293,
        gen_time=29.0,
        gen_time_units="Years",
        reverse_timeline=True,
    )



.. image:: moments_YRI_CEU_optimize_ga_files/moments_YRI_CEU_optimize_ga_20_0.png


