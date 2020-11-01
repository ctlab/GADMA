Use optimization from GADMA
===========================

GADMA provides different optimizations algorithms with one interface
that could be used to optimize different functions.

GADMA has ``gadma.optimizers`` module with different global, local
optimizers and their combinations. Here we will try to use them for find
optimums of different functions.

All optimizations have two base options: \* ``log_transform`` -
indicates if search space of optimization is log-transformed. \*
``maximize`` - indicates what problem this optimizer solves:
maximization or minimization.

Base pipeline
-------------

To run optimization one should prepare function ``f`` that takes vector
``x`` of parameters values as first argument, list of ``variables`` that
corresponds to ``x`` and name of optimization algorithm.

Function to optimize
--------------------

For this example we will use Rosenbrook function as an example:

.. math:: f(x) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1- x_i)^2\right]

The optimum of such function is in point :math:`(1, 1, ..., 1)`.

We will use existing realization of Rosenbrook function in scipy:

.. code:: ipython3

    from scipy.optimize import rosen
    f = rosen

Now let us create variables for our Rosenbrook function. There is
additional example about different variables in GADMA API in
``examples/api_examples/variables_example.ipynb`` or in the
`documentation <https://gadma.readthedocs.io/en/latest/api_examples/variables_example.html>`__.

.. code:: ipython3

    from gadma import *
    
    var1 = ContinuousVariable('var1', [-1, 2])
    var2 = ContinuousVariable('var2', [0, 10])
    var3 = PopulationSizeVariable('var3')  # it actually suits this example by value of its bound.

Global optimizations
--------------------

GADMA has basic algorithm of global optimization - genetic algorithm.
There is also Bayesian optimization but it is not stable anough to use
now. So let us take our ``f`` and optimize it with GA. First of all get
our optimizer:

.. code:: ipython3

    opt1 = get_global_optimizer("Genetic_algorithm")
    opt1.maximize = False  # we have minimization problem

.. code:: ipython3

    help(opt1.optimize)


.. parsed-literal::

    Help on method optimize in module gadma.optimizers.genetic_algorithm:
    
    optimize(f, variables, args=(), num_init=50, X_init=None, Y_init=None, linear_constrain=None, maxiter=None, maxeval=None, verbose=0, callback=None, report_file=None, eval_file=None, save_file=None, restore_file=None, restore_points_only=False, restore_x_transform=None) method of gadma.optimizers.genetic_algorithm.GeneticAlgorithm instance
        Return best values of `variables` that minimizes/maximizes
        the function `f`.
        
        :param f: function to minimize/maximize. The usage must be the
                  following: f(x, \*args), where x is list of values.
        :param variables: list of variables (instances of
                          :class:`gadma.Variable` class) of the function.
        :param X_init: list of initial values.
        :param Y_init: value of function `f` on initial values from `X_init`.
        :param args: arguments of function `f`.
        :param maxiter: maximum number of genetic algorithm's generations.
        :param maxeval: maximum number of function evaluations.
        :param callback: callback to call after each generation.
                         It will be called as callback(x, y), where x, y -
                         best_solution of generation and its fitness.
    


So we will set our function and variables and run optimization by:

.. code:: ipython3

    f = rosen
    variables = [var1, var2, var3]
    
    # first run for 5 iterations of genetic algorithm and print all output
    res = opt1.optimize(f, variables, verbose=1, maxiter=10)


.. parsed-literal::

    Generation #0.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 56.574674	(var1=-8.02e-01,	var2=0.11707,	var3=0.5128)	r
    1	 67.648121	(var1=0.13811,	var2=0.78656,	var3=0.90075)	r
    2	 73.174806	(var1=0.90363,	var2=0.98986,	var3=0.1422)	r
    3	 87.813760	(var1=0.56828,	var2=0.00362,	var3=0.8743)	r
    4	 114.322055	(var1=0.96019,	var2=1.05319,	var3=0.0481)	r
    5	 130.260656	(var1=0.50258,	var2=0.82626,	var3=1.66796)	r
    6	 172.393181	(var1=0.76879,	var2=1.41752,	var3=3.02846)	r
    7	 230.069293	(var1=0.03052,	var2=1.2144,	var3=2.37939)	r
    8	 1681.749298	(var1=1.5708,	var2=2.14935,	var3=0.53315)	r
    9	 2022.846855	(var1=-8.85e-01,	var2=2.30096,	var3=1.06656)	r
    Current mean mutation rate:	 0.200000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 56.57467403543714
    Solution:		(var1=-8.02e-01,	var2=0.11707,	var3=0.5128)	r
    
    
    Generation #1.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 56.574674	(var1=-8.02e-01,	var2=0.11707,	var3=0.5128)	r
    1	 67.648121	(var1=0.13811,	var2=0.78656,	var3=0.90075)	r
    2	 67.648121	(var1=0.13811,	var2=0.78656,	var3=0.90075)	c
    3	 69.529045	(var1=0.90363,	var2=0.98986,	var3=0.16426)	m
    4	 83.673058	(var1=0.76879,	var2=1.41752,	var3=2.39843)	m
    5	 172.393181	(var1=0.76879,	var2=1.41752,	var3=3.02846)	c
    6	 214.039213	(var1=0.03052,	var2=1.38965,	var3=2.37939)	m
    7	 1681.749298	(var1=1.5708,	var2=2.14935,	var3=0.53315)	c
    8	 7754.096662	(var1=0.60962,	var2=2.94208,	var3=0.23596)	r
    9	 13542.056084	(var1=0.09119,	var2=1.7776,	var3=14.66097)	r
    Current mean mutation rate:	 0.240000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 56.57467403543714
    Solution:		(var1=-8.02e-01,	var2=0.11707,	var3=0.5128)	r
    
    Mean time:	0.007 sec.
    
    
    
    Generation #2.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 15.104148	(var1=0.90363,	var2=0.73737,	var3=0.16426)	m
    1	 48.206888	(var1=0.76879,	var2=0.11707,	var3=0.5128)	c
    2	 56.574674	(var1=-8.02e-01,	var2=0.11707,	var3=0.5128)	r
    3	 67.648121	(var1=0.13811,	var2=0.78656,	var3=0.90075)	r
    4	 69.916279	(var1=-8.02e-01,	var2=0.11707,	var3=0.63218)	m
    5	 191.433031	(var1=0.76879,	var2=1.41752,	var3=0.90075)	c
    6	 599.249314	(var1=-4.41e-01,	var2=1.61298,	var3=0.61294)	r
    7	 722.800104	(var1=0.76879,	var2=1.0936,	var3=3.83694)	mm
    8	 7948.366158	(var1=0.13811,	var2=2.94208,	var3=0.23596)	c
    9	 305868.434960	(var1=0.06174,	var2=7.46726,	var3=0.96438)	r
    Current mean mutation rate:	 0.288000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 15.10414752105835
    Solution:		(var1=0.90363,	var2=0.73737,	var3=0.16426)	m
    
    Mean time:	0.006 sec.
    
    
    
    Generation #3.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 15.104148	(var1=0.90363,	var2=0.73737,	var3=0.16426)	m
    1	 39.359484	(var1=-4.41e-01,	var2=0.11707,	var3=0.61294)	c
    2	 48.206888	(var1=0.76879,	var2=0.11707,	var3=0.5128)	c
    3	 48.206888	(var1=0.76879,	var2=0.11707,	var3=0.5128)	c
    4	 49.279849	(var1=0.76879,	var2=0.10812,	var3=0.5128)	m
    5	 124.926305	(var1=0.13811,	var2=1.0936,	var3=0.90075)	c
    6	 234.544529	(var1=0.76879,	var2=1.41752,	var3=0.7209)	m
    7	 8029.610396	(var1=0.13811,	var2=2.94208,	var3=0.18785)	m
    8	 180909.408864	(var1=0.45318,	var2=6.89186,	var3=5.49729)	r
    9	 469334.638352	(var1=0.13088,	var2=8.25583,	var3=0.15192)	r
    Current mean mutation rate:	 0.275168
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 15.10414752105835
    Solution:		(var1=0.90363,	var2=0.73737,	var3=0.16426)	m
    
    Mean time:	0.005 sec.
    
    
    
    Generation #4.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 9.008701	(var1=0.90363,	var2=0.56035,	var3=0.16426)	m
    1	 15.104148	(var1=0.90363,	var2=0.73737,	var3=0.16426)	m
    2	 35.253067	(var1=0.76879,	var2=0.11707,	var3=0.35947)	m
    3	 39.359484	(var1=-4.41e-01,	var2=0.11707,	var3=0.61294)	c
    4	 48.206888	(var1=0.76879,	var2=0.11707,	var3=0.5128)	c
    5	 150.202214	(var1=0.13811,	var2=1.0936,	var3=0.61294)	c
    6	 362.490609	(var1=0.13811,	var2=1.41752,	var3=0.7209)	c
    7	 458.283642	(var1=0.27063,	var2=0.40782,	var3=2.27868)	r
    8	 7913.137784	(var1=0.13811,	var2=2.94208,	var3=0.2569)	m
    9	 26221.189961	(var1=-1.02e-01,	var2=5.14986,	var3=11.1714)	r
    Current mean mutation rate:	 0.330201
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 9.008700592809172
    Solution:		(var1=0.90363,	var2=0.56035,	var3=0.16426)	m
    
    Mean time:	0.005 sec.
    
    
    
    Generation #5.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    1	 9.008701	(var1=0.90363,	var2=0.56035,	var3=0.16426)	m
    2	 9.008701	(var1=0.90363,	var2=0.56035,	var3=0.16426)	c
    3	 15.104148	(var1=0.90363,	var2=0.73737,	var3=0.16426)	m
    4	 38.320585	(var1=-4.41e-01,	var2=0.13988,	var3=0.61294)	m
    5	 71.990308	(var1=0.76879,	var2=1.0936,	var3=0.5128)	c
    6	 443.675322	(var1=0.08627,	var2=1.36533,	var3=0.2569)	mm
    7	 4654.010086	(var1=1.6115,	var2=2.65169,	var3=0.21191)	r
    8	 7374.023998	(var1=1.07976,	var2=3.34865,	var3=2.91163)	r
    9	 25456.311743	(var1=0.90363,	var2=5.14986,	var3=11.1714)	c
    Current mean mutation rate:	 0.396241
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 2.5378326176371835
    Solution:		(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    
    Mean time:	0.005 sec.
    
    
    
    Generation #6.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    1	 7.515823	(var1=0.90363,	var2=0.56035,	var3=0.22743)	m
    2	 9.008701	(var1=0.90363,	var2=0.56035,	var3=0.16426)	m
    3	 17.915913	(var1=-4.41e-01,	var2=0.56035,	var3=0.16426)	c
    4	 39.108874	(var1=1.07976,	var2=0.56035,	var3=0.16426)	c
    5	 39.878190	(var1=0.76879,	var2=1.0936,	var3=0.8144)	m
    6	 45.474861	(var1=0.76879,	var2=0.13988,	var3=0.5128)	c
    7	 67.896375	(var1=-4.41e-01,	var2=0.13988,	var3=0.82446)	m
    8	 76.898545	(var1=-1.48e-01,	var2=0.74295,	var3=0.06687)	r
    9	 407910.264188	(var1=0.88308,	var2=7.97531,	var3=0.14823)	r
    Current mean mutation rate:	 0.378586
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 2.5378326176371835
    Solution:		(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    
    Mean time:	0.004 sec.
    
    
    
    Generation #7.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 1.214084	(var1=0.73636,	var2=0.56035,	var3=0.21816)	m
    1	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    2	 2.583082	(var1=0.76879,	var2=0.56035,	var3=0.16426)	c
    3	 7.412472	(var1=-6.40e-01,	var2=0.56035,	var3=0.16426)	m
    4	 7.515823	(var1=0.90363,	var2=0.56035,	var3=0.22743)	m
    5	 7.515823	(var1=0.90363,	var2=0.56035,	var3=0.22743)	c
    6	 9.335244	(var1=-5.93e-01,	var2=0.56035,	var3=0.16426)	m
    7	 41.731136	(var1=-4.41e-01,	var2=0.56035,	var3=0.82446)	c
    8	 1510.723283	(var1=0.78144,	var2=0.03964,	var3=3.84494)	r
    9	 4155.124561	(var1=-2.53e-02,	var2=2.48096,	var3=0.20816)	r
    Current mean mutation rate:	 0.454303
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 1.2140835589750745
    Solution:		(var1=0.73636,	var2=0.56035,	var3=0.21816)	m
    
    Mean time:	0.004 sec.
    
    
    
    Generation #8.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 1.214084	(var1=0.73636,	var2=0.56035,	var3=0.21816)	m
    1	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    2	 7.515823	(var1=0.90363,	var2=0.56035,	var3=0.22743)	c
    3	 7.842367	(var1=-5.93e-01,	var2=0.56035,	var3=0.22743)	c
    4	 15.883830	(var1=0.44482,	var2=0.56035,	var3=0.16426)	m
    5	 20.640181	(var1=0.37193,	var2=0.56035,	var3=0.16426)	m
    6	 56.952460	(var1=-6.72e-01,	var2=0.97386,	var3=1.46674)	r
    7	 1478.888527	(var1=0.31412,	var2=0.03964,	var3=3.84494)	m
    8	 4207.538350	(var1=-2.53e-02,	var2=2.48096,	var3=0.16426)	c
    9	 882830.189226	(var1=-5.77e-01,	var2=9.72183,	var3=1.02946)	r
    Current mean mutation rate:	 0.434061
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 1.2140835589750745
    Solution:		(var1=0.73636,	var2=0.56035,	var3=0.21816)	m
    
    Mean time:	0.004 sec.
    
    
    
    Generation #9.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 1.214084	(var1=0.73636,	var2=0.56035,	var3=0.21816)	m
    1	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    2	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	c
    3	 5.092744	(var1=0.46282,	var2=0.42174,	var3=0.21816)	mm
    4	 15.883830	(var1=0.44482,	var2=0.56035,	var3=0.16426)	c
    5	 53.310934	(var1=0.37193,	var2=0.71254,	var3=0.06204)	mm
    6	 62.353679	(var1=-6.72e-01,	var2=1.2224,	var3=1.46674)	m
    7	 4207.538350	(var1=-2.53e-02,	var2=2.48096,	var3=0.16426)	c
    8	 127440.740649	(var1=1.37133,	var2=5.98973,	var3=0.41879)	r
    9	 494962.340131	(var1=0.03034,	var2=8.43732,	var3=1.34653)	r
    Current mean mutation rate:	 0.414720
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 1.2140835589750745
    Solution:		(var1=0.73636,	var2=0.56035,	var3=0.21816)	m
    
    Mean time:	0.004 sec.
    
    
    
    Generation #10.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 0.555263	(var1=0.73636,	var2=0.56035,	var3=0.26304)	m
    1	 1.214084	(var1=0.73636,	var2=0.56035,	var3=0.21816)	m
    2	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	m
    3	 2.537833	(var1=0.73636,	var2=0.56035,	var3=0.16426)	c
    4	 3.674001	(var1=0.54404,	var2=0.42643,	var3=0.06204)	mm
    5	 6.272040	(var1=0.46282,	var2=0.42174,	var3=0.06204)	c
    6	 16.359379	(var1=0.41742,	var2=0.56035,	var3=0.21816)	m
    7	 327.262053	(var1=-2.53e-02,	var2=1.2224,	var3=0.16426)	c
    8	 24507.166510	(var1=-8.40e-01,	var2=4.37933,	var3=3.96577)	r
    9	 403191.326218	(var1=0.39847,	var2=8.0792,	var3=2.27597)	r
    Current mean mutation rate:	 0.497664
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 0.5552629843855047
    Solution:		(var1=0.73636,	var2=0.56035,	var3=0.26304)	m
    
    Mean time:	0.004 sec.
    
    
    


.. code:: ipython3

    # now run for 1000 iterations and print every 100 iteration
    # It may converge faster than 1000 iterations
    res = opt1.optimize(f, variables, verbose=100, maxiter=1000)


.. parsed-literal::

    Generation #0.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 10.552293	(var1=-8.09e-01,	var2=0.48978,	var3=0.44764)	r
    1	 30.387173	(var1=-6.87e-01,	var2=0.41098,	var3=0.68686)	r
    2	 33.131647	(var1=0.08459,	var2=0.47698,	var3=0.54289)	r
    3	 174.058054	(var1=-2.18e-01,	var2=1.09656,	var3=0.41154)	r
    4	 231.890043	(var1=1.39326,	var2=1.40142,	var3=0.54117)	r
    5	 392.481091	(var1=0.05274,	var2=1.58042,	var3=1.30463)	r
    6	 517.699804	(var1=0.33108,	var2=1.94534,	var3=2.44504)	r
    7	 584.194038	(var1=1.0545,	var2=0.91638,	var3=3.24881)	r
    8	 645.364608	(var1=-9.81e-01,	var2=2.46343,	var3=4.03448)	r
    9	 1056.283990	(var1=-6.82e-01,	var2=1.77675,	var3=0.18879)	r
    Current mean mutation rate:	 0.200000
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 10.552293186197584
    Solution:		(var1=-8.09e-01,	var2=0.48978,	var3=0.44764)	r
    
    
    Generation #100.
    Current generation of solutions:
    N	Value of fitness function	Solution
    0	 0.061651	(var1=0.88493,	var2=0.78108,	var3=0.61094)	m
    1	 0.063339	(var1=0.88493,	var2=0.78108,	var3=0.61428)	m
    2	 0.063339	(var1=0.88493,	var2=0.78108,	var3=0.61428)	c
    3	 0.063339	(var1=0.88493,	var2=0.78108,	var3=0.61428)	c
    4	 0.090993	(var1=0.88493,	var2=0.78108,	var3=0.62723)	c
    5	 0.302246	(var1=0.9028,	var2=0.77055,	var3=0.61428)	m
    6	 373.547100	(var1=1.62753,	var2=3.01363,	var3=7.1957)	c
    7	 7707.724174	(var1=0.87255,	var2=3.01363,	var3=0.59881)	m
    8	 271046.472139	(var1=-6.66e-01,	var2=7.21326,	var3=0.41509)	r
    9	 512406.489146	(var1=0.33401,	var2=8.60858,	var3=3.03519)	r
    Current mean mutation rate:	 0.020477
    Current mean number of params to change during mutation:	  1
    
    --Best solution by value of fitness function--
    Value of fitness: 0.06333856222983994
    Solution:		(var1=0.88493,	var2=0.78108,	var3=0.61428)	m
    
    Mean time:	0.004 sec.
    
    
    


So the result we got:

.. code:: ipython3

    print(f"Full result info:\n{res}")
    print(f"Best values of parameters:\t{res.x}")
    print(f"Best value of function:\t{res.y}")
    print(f"Optimal valies of parameters:\t[1, 1, 1]")
    print(f"Optimum value of function:\t{f([1, 1, 1])}")


.. parsed-literal::

    Full result info:
      status: 0
     success: True
     message: CONVERGENCE: NO IMPROVEMENT DURING 100 ITERATIONS
           x: [0.8849317473156754 0.7810785929158809 0.6142801591640911]
           y: 0.06333856222983994
      n_eval: 1783
      n_iter: 180
    
    Best values of parameters:	[0.8849317473156754 0.7810785929158809 0.6142801591640911]
    Best value of function:	0.06333856222983994
    Optimal valies of parameters:	[1, 1, 1]
    Optimum value of function:	0.0


So our result is close but not ideal. To find better solution we could
use local optimizations.

Local optimizations
-------------------

GADMA provides several local optimizations **(local optimizations work
with continuous variables only)**:

.. code:: ipython3

    print("GADMA has following local optimizers:")
    for opt in all_local_optimizers():
        print(opt.id)


.. parsed-literal::

    GADMA has following local optimizers:
    optimize
    optimize_log_fmin
    optimize_log
    optimize_powell
    None
    optimize_lbfgsb
    optimize_log_lbfgsb
    optimize_log_powell
    optimize_fmin


Names were taken from ``dadi`` and ``moments`` software. Here are some
descriptions:

-  optimize - BFGS.
-  optimize\_lbfgsb - L-BFGS-B.
-  optimize\_powell - Powell's method.
-  optimize\_fmin - Neadler-Mead algorithm.
-  None - no optimization.

If there is ``log`` in name then this log transform for search space is
using. All these methods could be got by ``get_local_optimizer`` by one
of its names. For example Powell's method with log\_transform could be
got by:

.. code:: ipython3

    opt2 = get_local_optimizer("Powell_log")
    # or
    opt2 = get_local_optimizer("optimize_log_powell")
    print(f"Optimizer is log transformed: {opt._log_trasform}")


.. parsed-literal::

    Optimizer is log transformed: False


So the ``optimize`` function is almost the same except new ``x0``
argument for first approximation of best solution. Let us use result
from global optimizer:

.. code:: ipython3

    help(opt2.optimize)


.. parsed-literal::

    Help on method optimize in module gadma.optimizers.local_optimizer:
    
    optimize(f, variables, x0, args=(), options={}, linear_constrain=None, maxiter=None, maxeval=None, verbose=0, callback=None, eval_file=None, report_file=None, save_file=None, restore_file=None, restore_points_only=False, restore_x_transform=None) method of gadma.optimizers.local_optimizer.ManuallyConstrOptimizer instance
        Run optimization of local search algorithm.
        
        :param f: Target function to optimize.
        :type f: func
        :param variables: Variables of `f` which values should be optimized.
        :type variables: :class:`gadma.utils.VariablePool`
        :param x0: Initial point to start optimization.
        :type x0: list
        :param args: Additional arguments of target function.
        :type args: tuple
        :param options: Additional options kwargs for optimization.
        :type options: dict
        :param maxiter: Maximum number of iterations to run.
        :type maxiter: int
        :param maxeval: Maximum number of evaluations to run. If None then run
                        until converge.
        :type maxeval: int
        :param verbose: Verbosity of the output. If 0 then no reports.
        :type verbose: int
        :param callback: Callback to run after each iteration of optimization.
                         Should be called as `callback(x, y)`
        :type callback: function
        :param report_file: File to save report. Check option `verbose`.
        :type report_file: str
        :param eval_file: File to save all evaluations of the function `f`.
        :type eval_file: str
        :param save_file: File to save information during optimization for its
                          reconstruction.
        :type save_file: str
        :param restore_file: File to restore previous run.
        :type restore_file: str
        :param restore_points_only: Restore point/points from previous run and
                                    run optimization from them once more. If
                                    False then previous run will be resumed.
        :type restore_points_only: bool
        :param restore_x_transform: Restore points but transform them before
                                    usage in this run.
        :type restore_x_transform: function
    


.. code:: ipython3

    opt2.optimize(f, variables, x0=res.x, verbose=1)


.. parsed-literal::

    1	0.06333856222983994	(var1=0.88493,	var2=0.78108,	var3=0.61428)	
    2	57.0227960616551	(var1=0.17547,	var2=0.78108,	var3=0.61428)	
    3	30.96696404229835	(var1=0.47698,	var2=0.78108,	var3=0.61428)	
    4	81.13972107200483	(var1=1.29657,	var2=0.78108,	var3=0.61428)	
    5	7.438547050163425	(var1=0.71467,	var2=0.78108,	var3=0.61428)	
    6	7.198651372348043	(var1=1.02394,	var2=0.78108,	var3=0.61428)	
    7	0.29295430196062716	(var1=0.85667,	var2=0.78108,	var3=0.61428)	
    8	0.06720632937571171	(var1=0.88055,	var2=0.78108,	var3=0.61428)	
    9	0.06315252292607579	(var1=0.88425,	var2=0.78108,	var3=0.61428)	
    10	0.06315005999083027	(var1=0.88415,	var2=0.78108,	var3=0.61428)	
    11	0.06315006214326982	(var1=0.88416,	var2=0.78108,	var3=0.61428)	
    12	0.06315009580883257	(var1=0.88415,	var2=0.78108,	var3=0.61428)	
    13	1697.2887647241073	(var1=0.88415,	var2=2.12319,	var3=0.61428)	
    14	74.86616894871793	(var1=0.88415,	var2=0.15488,	var3=0.61428)	
    15	32.46086956077491	(var1=0.88415,	var2=0.421,	var3=0.61428)	
    16	61.54420179234234	(var1=0.88415,	var2=1.14441,	var3=0.61428)	
    17	5.673506040486336	(var1=0.88415,	var2=0.64839,	var3=0.61428)	
    18	5.61345551991403	(var1=0.88415,	var2=0.90377,	var3=0.61428)	
    19	0.17060254969790106	(var1=0.88415,	var2=0.76584,	var3=0.61428)	
    20	0.062429944243381066	(var1=0.88415,	var2=0.78149,	var3=0.61428)	
    21	0.060596795921373126	(var1=0.88415,	var2=0.78391,	var3=0.61428)	
    22	0.060592711044827614	(var1=0.88415,	var2=0.7838,	var3=0.61428)	
    23	0.060592850594461796	(var1=0.88415,	var2=0.78378,	var3=0.61428)	
    24	0.06059308840816433	(var1=0.88415,	var2=0.78383,	var3=0.61428)	
    25	111.45551711704908	(var1=0.88415,	var2=0.7838,	var3=1.66979)	
    26	24.320568755524313	(var1=0.88415,	var2=0.7838,	var3=0.1218)	
    27	8.08363424457279	(var1=0.88415,	var2=0.7838,	var3=0.3311)	
    28	8.221410496823118	(var1=0.88415,	var2=0.7838,	var3=0.90002)	
    29	0.5443490198370673	(var1=0.88415,	var2=0.7838,	var3=0.5448)	
    30	0.16797640077573167	(var1=0.88415,	var2=0.7838,	var3=0.64712)	
    31	0.06097784125681071	(var1=0.88415,	var2=0.7838,	var3=0.61238)	
    32	0.0605925093321193	(var1=0.88415,	var2=0.7838,	var3=0.6143)	
    33	0.060592241914231765	(var1=0.88415,	var2=0.7838,	var3=0.61435)	
    34	0.06059224193454777	(var1=0.88415,	var2=0.7838,	var3=0.61435)	
    35	0.06059224198829155	(var1=0.88415,	var2=0.7838,	var3=0.61435)	
    36	0.06477847578610435	(var1=0.88338,	var2=0.78654,	var3=0.61442)	
    37	57.43796088434369	(var1=0.17532,	var2=0.7838,	var3=0.61435)	
    38	31.31144654021446	(var1=0.47656,	var2=0.7838,	var3=0.61435)	
    39	80.11682054591802	(var1=1.29543,	var2=0.7838,	var3=0.61435)	
    40	7.529066819296717	(var1=0.71537,	var2=0.7838,	var3=0.61435)	
    41	6.953463422328221	(var1=1.02303,	var2=0.7838,	var3=0.61435)	
    42	0.2861168269898842	(var1=0.85847,	var2=0.7838,	var3=0.61435)	
    43	0.06428830228071211	(var1=0.88193,	var2=0.7838,	var3=0.61435)	
    44	0.059858770096483535	(var1=0.88587,	var2=0.7838,	var3=0.61435)	
    45	0.05984883224732027	(var1=0.8857,	var2=0.7838,	var3=0.61435)	
    46	0.059848853851534975	(var1=0.88568,	var2=0.7838,	var3=0.61435)	
    47	0.05984896091797328	(var1=0.88571,	var2=0.7838,	var3=0.61435)	
    48	1723.1480270347665	(var1=0.8857,	var2=2.1306,	var3=0.61435)	
    49	75.12838038498057	(var1=0.8857,	var2=0.15542,	var3=0.61435)	
    50	32.4478916541783	(var1=0.8857,	var2=0.42247,	var3=0.61435)	
    51	62.90870988417057	(var1=0.8857,	var2=1.1484,	var3=0.61435)	
    52	5.677105812022754	(var1=0.8857,	var2=0.64932,	var3=0.61435)	
    53	10.76225018492341	(var1=0.8857,	var2=0.9479,	var3=0.61435)	
    54	0.23815276213761033	(var1=0.8857,	var2=0.76171,	var3=0.61435)	
    55	0.06679229760215485	(var1=0.8857,	var2=0.78915,	var3=0.61435)	
    56	0.05962353034984764	(var1=0.8857,	var2=0.78452,	var3=0.61435)	
    57	0.05962009441218935	(var1=0.8857,	var2=0.78461,	var3=0.61435)	
    58	0.059620092457041	(var1=0.8857,	var2=0.78462,	var3=0.61435)	
    59	0.0596201365019107	(var1=0.8857,	var2=0.78463,	var3=0.61435)	
    60	111.22398421222725	(var1=0.8857,	var2=0.78462,	var3=1.66997)	
    61	24.44422045435629	(var1=0.8857,	var2=0.78462,	var3=0.12182)	
    62	8.1529970137971	(var1=0.8857,	var2=0.78462,	var3=0.33114)	
    63	8.153102754728097	(var1=0.8857,	var2=0.78462,	var3=0.90012)	
    64	0.5449507960340264	(var1=0.8857,	var2=0.78462,	var3=0.54595)	
    65	0.17316022247937884	(var1=0.8857,	var2=0.78462,	var3=0.64935)	
    66	0.05984867609681263	(var1=0.8857,	var2=0.78462,	var3=0.61365)	
    67	0.05945731427272312	(var1=0.8857,	var2=0.78462,	var3=0.61554)	
    68	0.059456620763198215	(var1=0.8857,	var2=0.78462,	var3=0.61563)	
    69	0.0594566313363329	(var1=0.8857,	var2=0.78462,	var3=0.61562)	
    70	0.05945664307016195	(var1=0.8857,	var2=0.78462,	var3=0.61564)	
    71	0.059063473594041196	(var1=0.88724,	var2=0.78544,	var3=0.61691)	
    72	0.060025755543307475	(var1=0.88975,	var2=0.78676,	var3=0.619)	
    73	0.05919613277444269	(var1=0.8882,	var2=0.78594,	var3=0.61771)	
    74	0.05912536477717941	(var1=0.88665,	var2=0.78512,	var3=0.61642)	
    75	0.059063245028626477	(var1=0.88728,	var2=0.78546,	var3=0.61694)	
    76	0.05906328025043809	(var1=0.8873,	var2=0.78546,	var3=0.61696)	
    77	0.05906328875914428	(var1=0.88726,	var2=0.78545,	var3=0.61693)	
    78	112.43780824198534	(var1=0.88728,	var2=0.78546,	var3=1.67703)	
    79	24.522673831410383	(var1=0.88728,	var2=0.78546,	var3=0.12233)	
    80	8.147679210348716	(var1=0.88728,	var2=0.78546,	var3=0.33253)	
    81	8.295021601686166	(var1=0.88728,	var2=0.78546,	var3=0.90392)	
    82	0.5469231486347989	(var1=0.88728,	var2=0.78546,	var3=0.54709)	
    83	0.1670055102659087	(var1=0.88728,	var2=0.78546,	var3=0.64979)	
    84	0.05945183240537135	(var1=0.88728,	var2=0.78546,	var3=0.61497)	
    85	0.059063493869065425	(var1=0.88728,	var2=0.78546,	var3=0.61689)	
    86	0.05906324305841605	(var1=0.88728,	var2=0.78546,	var3=0.61694)	
    87	0.059063243057883776	(var1=0.88728,	var2=0.78546,	var3=0.61694)	
    88	0.05906324305808255	(var1=0.88728,	var2=0.78546,	var3=0.61694)	
    89	1736.6340318745927	(var1=0.88728,	var2=2.13509,	var3=0.61694)	
    90	75.73447222260324	(var1=0.88728,	var2=0.15575,	var3=0.61694)	
    91	32.74624162552199	(var1=0.88728,	var2=0.42336,	var3=0.61694)	
    92	63.300390902270294	(var1=0.88728,	var2=1.15082,	var3=0.61694)	
    93	5.73241554589938	(var1=0.88728,	var2=0.65086,	var3=0.61694)	
    94	5.868540865051781	(var1=0.88728,	var2=0.90883,	var3=0.61694)	
    95	0.17267609444484244	(var1=0.88728,	var2=0.76836,	var3=0.61694)	
    96	0.0633672845134948	(var1=0.88728,	var2=0.79028,	var3=0.61694)	
    97	0.05861470068420056	(var1=0.88728,	var2=0.78653,	var3=0.61694)	
    98	0.05861360319190697	(var1=0.88728,	var2=0.78659,	var3=0.61694)	
    99	0.058613610873836944	(var1=0.88728,	var2=0.7866,	var3=0.61694)	
    100	0.05861368491724064	(var1=0.88728,	var2=0.78658,	var3=0.61694)	
    101	0.05856301882253134	(var1=0.88887,	var2=0.78742,	var3=0.61826)	
    102	0.060175002362247695	(var1=0.89144,	var2=0.78878,	var3=0.62039)	
    103	0.05892977872248001	(var1=0.88985,	var2=0.78794,	var3=0.61907)	
    104	0.058488811326223474	(var1=0.88826,	var2=0.7871,	var3=0.61775)	
    105	0.05848767237150018	(var1=0.88818,	var2=0.78706,	var3=0.61768)	
    106	0.05848767951529672	(var1=0.88817,	var2=0.78706,	var3=0.61768)	
    107	0.05848769059051995	(var1=0.88819,	var2=0.78706,	var3=0.61769)	
    108	0.05896518413306298	(var1=0.89066,	var2=0.78951,	var3=0.61975)	
    109	112.32848044627256	(var1=0.88818,	var2=0.78706,	var3=1.67904)	
    110	24.757505038921366	(var1=0.88818,	var2=0.78706,	var3=0.12248)	
    111	8.268109455453654	(var1=0.88818,	var2=0.78706,	var3=0.33293)	
    112	8.211725247493176	(var1=0.88818,	var2=0.78706,	var3=0.90501)	
    113	0.5495939855289741	(var1=0.88818,	var2=0.78706,	var3=0.54936)	
    114	0.07402408431425862	(var1=0.88818,	var2=0.78706,	var3=0.60687)	
    115	0.05826076360286656	(var1=0.88818,	var2=0.78706,	var3=0.62041)	
    116	0.05817110676174364	(var1=0.88818,	var2=0.78706,	var3=0.61947)	
    117	0.05817111276118113	(var1=0.88818,	var2=0.78706,	var3=0.61945)	
    118	0.05817116478022161	(var1=0.88818,	var2=0.78706,	var3=0.61949)	
    119	1750.1244261308098	(var1=0.88818,	var2=2.13945,	var3=0.61947)	
    120	76.18369080408024	(var1=0.88818,	var2=0.15606,	var3=0.61947)	
    121	32.95574130393726	(var1=0.88818,	var2=0.42423,	var3=0.61947)	
    122	63.765495290045735	(var1=0.88818,	var2=1.15317,	var3=0.61947)	
    123	5.773729543554591	(var1=0.88818,	var2=0.65214,	var3=0.61947)	
    124	5.909974377233213	(var1=0.88818,	var2=0.91069,	var3=0.61947)	
    125	0.17280038509770868	(var1=0.88818,	var2=0.7699,	var3=0.61947)	
    126	0.06252638612231508	(var1=0.88818,	var2=0.79189,	var3=0.61947)	
    127	0.0577278138226938	(var1=0.88818,	var2=0.78813,	var3=0.61947)	
    128	0.05772669444797826	(var1=0.88818,	var2=0.78818,	var3=0.61947)	
    129	0.05772670194809414	(var1=0.88818,	var2=0.78819,	var3=0.61947)	
    130	0.05772677531194169	(var1=0.88818,	var2=0.78817,	var3=0.61947)	
    131	0.057680860205617926	(var1=0.88976,	var2=0.78902,	var3=0.62079)	
    132	0.05930735532496334	(var1=0.89234,	var2=0.79038,	var3=0.62294)	
    133	0.05805215976192045	(var1=0.89075,	var2=0.78954,	var3=0.62161)	
    134	0.057604461431079154	(var1=0.88916,	var2=0.7887,	var3=0.62029)	
    135	0.05760303690579306	(var1=0.88906,	var2=0.78865,	var3=0.62021)	
    136	0.05760304366471021	(var1=0.88906,	var2=0.78865,	var3=0.6202)	
    137	0.05760305505631843	(var1=0.88907,	var2=0.78866,	var3=0.62022)	
    138	0.05672352078886539	(var1=0.88995,	var2=0.79025,	var3=0.62275)	
    139	0.05531136491723874	(var1=0.89139,	var2=0.79284,	var3=0.62687)	
    140	0.01851480494447806	(var1=1.05464,	var2=1.11413,	var3=1.24593)	
    141	1.3248253739057012	(var1=1.38445,	var2=1.93199,	var3=3.78598)	
    142	0.19410156895811675	(var1=1.17015,	var2=1.37484,	var3=1.90486)	
    143	0.0007382340401204149	(var1=0.98903,	var2=0.97836,	var3=0.9584)	
    144	0.0005111821003906816	(var1=1.00462,	var2=1.00982,	var3=1.02165)	
    145	0.00028872178968469646	(var1=0.998,	var2=0.99641,	var3=0.99443)	
    146	0.00029328027575736403	(var1=0.99915,	var2=0.99874,	var3=0.99913)	
    147	0.00029847649862004455	(var1=0.99685,	var2=0.99408,	var3=0.98974)	
    148	292.51357321737976	(var1=0.998,	var2=0.99641,	var3=2.70313)	
    149	63.30496906025734	(var1=0.998,	var2=0.99641,	var3=0.19718)	
    150	20.869239570401955	(var1=0.998,	var2=0.99641,	var3=0.536)	
    151	21.545255018733265	(var1=0.998,	var2=0.99641,	var3=1.457)	
    152	1.2638805200183942	(var1=0.998,	var2=0.99641,	var3=0.88041)	
    153	0.267533544551759	(var1=0.998,	var2=0.99641,	var3=1.04455)	
    154	0.001030686537574667	(var1=0.998,	var2=0.99641,	var3=0.98967)	
    155	3.322842370271053e-05	(var1=0.998,	var2=0.99641,	var3=0.99279)	
    156	3.305789680351054e-05	(var1=0.998,	var2=0.99641,	var3=0.99282)	
    157	3.307545539151939e-05	(var1=0.998,	var2=0.99641,	var3=0.99284)	
    158	3.30915493889246e-05	(var1=0.998,	var2=0.99641,	var3=0.99281)	
    159	0.00044779686099664484	(var1=0.99978,	var2=0.99747,	var3=0.99494)	
    160	0.002011627398846074	(var1=0.99512,	var2=0.99469,	var3=0.9894)	
    161	0.0004051169337430322	(var1=0.9969,	var2=0.99575,	var3=0.99152)	
    162	4.244799744744943e-05	(var1=0.99868,	var2=0.99681,	var3=0.99363)	
    163	1.457480567793838e-05	(var1=0.99831,	var2=0.99659,	var3=0.99319)	
    164	1.4576318867608422e-05	(var1=0.99831,	var2=0.99659,	var3=0.99319)	
    165	1.4576985361126131e-05	(var1=0.9983,	var2=0.99659,	var3=0.99318)	
    166	0.08778115430611688	(var1=1.12063,	var2=1.25912,	var3=1.59244)	
    167	0.13227805726800443	(var1=0.82802,	var2=0.68266,	var3=0.46268)	
    168	0.024507770420730733	(var1=0.92948,	var2=0.86249,	var3=0.74185)	
    169	0.010448996313275331	(var1=1.04337,	var2=1.0897,	var3=1.18945)	
    170	5.247198857653113e-05	(var1=0.9968,	var2=0.99356,	var3=0.98709)	
    171	6.047176104281367e-07	(var1=0.99975,	var2=0.9995,	var3=0.99905)	
    172	4.003149273302758e-07	(var1=0.99995,	var2=0.99991,	var3=0.99988)	
    173	4.006708748625715e-07	(var1=0.99993,	var2=0.99988,	var3=0.99982)	
    174	4.028249275713102e-07	(var1=0.99997,	var2=0.99994,	var3=0.99995)	
    175	0.09034161973753463	(var1=1.12466,	var2=1.26776,	var3=1.61198)	
    176	295.20087927767787	(var1=0.99995,	var2=0.99991,	var3=2.71796)	
    177	64.24923769822183	(var1=0.99995,	var2=0.99991,	var3=0.19826)	
    178	21.24116216005215	(var1=0.99995,	var2=0.99991,	var3=0.53894)	
    179	21.63821082419558	(var1=0.99995,	var2=0.99991,	var3=1.46499)	
    180	1.2813263193573465	(var1=0.99995,	var2=0.99991,	var3=0.88663)	
    181	0.2830800729077699	(var1=0.99995,	var2=0.99991,	var3=1.05303)	
    182	0.0010202903275264585	(var1=0.99995,	var2=0.99991,	var3=0.99663)	
    183	6.608147129021295e-07	(var1=0.99995,	var2=0.99991,	var3=0.99974)	
    184	2.405086189856941e-08	(var1=0.99995,	var2=0.99991,	var3=0.99982)	
    185	2.4085128118948434e-08	(var1=0.99995,	var2=0.99991,	var3=0.99982)	
    186	2.4091779817398097e-08	(var1=0.99995,	var2=0.99991,	var3=0.99982)	
    187	0.0006296031361739891	(var1=1.00174,	var2=1.00098,	var3=1.00196)	
    188	0.0016545317341091841	(var1=0.99706,	var2=0.99819,	var3=0.99638)	
    189	0.00024488938686373585	(var1=0.99885,	var2=0.99925,	var3=0.9985)	
    190	9.015174042643673e-05	(var1=1.00063,	var2=1.00032,	var3=1.00064)	
    191	9.076294545495014e-09	(var1=0.99996,	var2=0.99992,	var3=0.99983)	
    192	8.892000789206292e-09	(var1=0.99996,	var2=0.99992,	var3=0.99983)	
    193	8.893221737973971e-09	(var1=0.99996,	var2=0.99992,	var3=0.99983)	
    194	8.893805841189655e-09	(var1=0.99996,	var2=0.99992,	var3=0.99983)	
    195	0.09052842677379169	(var1=1.12248,	var2=1.26332,	var3=1.6031)	
    196	0.13037718486674457	(var1=0.82939,	var2=0.68494,	var3=0.46578)	
    197	0.023509618590357294	(var1=0.93102,	var2=0.86537,	var3=0.74681)	
    198	0.01128089562627626	(var1=1.04509,	var2=1.09333,	var3=1.19741)	
    199	5.4687011937690806e-05	(var1=0.99678,	var2=0.9935,	var3=0.98691)	
    200	1.0717809758906074e-07	(var1=0.99986,	var2=0.99971,	var3=0.99942)	
    201	2.7414864920312316e-10	(var1=1.0,	var2=1.0,	var3=1.0)	
    202	2.72233770917516e-10	(var1=1.0,	var2=1.0,	var3=1.0)	
    203	2.7307177564616513e-10	(var1=1.0,	var2=1.0,	var3=0.99999)	
    204	3.598630733510889e-07	(var1=1.00005,	var2=1.00008,	var3=1.00011)	
    205	295.2476321588907	(var1=1.0,	var2=1.0,	var3=2.71827)	
    206	64.2734624764209	(var1=1.0,	var2=1.0,	var3=0.19829)	
    207	21.25151230799889	(var1=1.0,	var2=1.0,	var3=0.539)	
    208	21.637583838450993	(var1=1.0,	var2=1.0,	var3=1.46516)	
    209	1.2817534302316456	(var1=1.0,	var2=1.0,	var3=0.88678)	
    210	0.2836401834492792	(var1=1.0,	var2=1.0,	var3=1.05325)	
    211	0.0010209746860975568	(var1=1.0,	var2=1.0,	var3=0.9968)	
    212	6.612685071833725e-07	(var1=1.0,	var2=1.0,	var3=0.99991)	
    213	1.8833859866696548e-11	(var1=1.0,	var2=1.0,	var3=0.99999)	
    214	1.7203557956609677e-11	(var1=1.0,	var2=1.0,	var3=0.99999)	
    215	1.7228871318367522e-11	(var1=1.0,	var2=1.0,	var3=0.99999)	
    216	1.7229318930444673e-11	(var1=1.0,	var2=1.0,	var3=0.99999)	
    217	0.0006357636213793796	(var1=1.00179,	var2=1.00106,	var3=1.00213)	
    218	0.0016451068932795663	(var1=0.99711,	var2=0.99828,	var3=0.99655)	
    219	0.00024118676237828036	(var1=0.9989,	var2=0.99934,	var3=0.99868)	
    220	9.245592746599701e-05	(var1=1.00068,	var2=1.0004,	var3=1.00081)	
    221	1.9582478409023648e-10	(var1=1.0,	var2=1.0,	var3=0.99999)	
    222	7.81705816475785e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    223	7.817087268252697e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    224	7.818888380554825e-12	(var1=1.0,	var2=1.0,	var3=0.99999)	
    225	0.09059627672437778	(var1=1.12253,	var2=1.26343,	var3=1.60336)	
    226	0.13033068226674646	(var1=0.82943,	var2=0.68499,	var3=0.46585)	
    227	0.023485387713827674	(var1=0.93106,	var2=0.86544,	var3=0.74693)	
    228	0.011301706200024126	(var1=1.04514,	var2=1.09342,	var3=1.1976)	
    229	5.4744450773146535e-05	(var1=0.99678,	var2=0.99349,	var3=0.98691)	
    230	1.048126506077899e-07	(var1=0.99986,	var2=0.99972,	var3=0.99943)	
    231	1.3705323106380515e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    232	2.3250474719700296e-13	(var1=1.0,	var2=1.0,	var3=1.0)	
    233	2.3325033560159503e-13	(var1=1.0,	var2=1.0,	var3=1.0)	
    234	2.332766288103868e-13	(var1=1.0,	var2=1.0,	var3=1.0)	
    235	2.413558204834738e-10	(var1=1.0,	var2=1.0,	var3=1.0)	
    236	295.24919611727694	(var1=1.0,	var2=1.0,	var3=2.71828)	
    237	64.27416756075355	(var1=1.0,	var2=1.0,	var3=0.19829)	
    238	21.251806194757773	(var1=1.0,	var2=1.0,	var3=0.539)	
    239	21.6375932535148	(var1=1.0,	var2=1.0,	var3=1.46516)	
    240	1.2817660354998646	(var1=1.0,	var2=1.0,	var3=0.88678)	
    241	0.28365506656692285	(var1=1.0,	var2=1.0,	var3=1.05326)	
    242	0.0010209942933596897	(var1=1.0,	var2=1.0,	var3=0.9968)	
    243	6.618943552007046e-07	(var1=1.0,	var2=1.0,	var3=0.99992)	
    244	1.714803008646205e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    245	1.712009571683522e-14	(var1=1.0,	var2=1.0,	var3=1.0)	
    246	1.714189774194186e-14	(var1=1.0,	var2=1.0,	var3=1.0)	
    247	1.714326063106875e-14	(var1=1.0,	var2=1.0,	var3=1.0)	
    248	0.0006359166932032684	(var1=1.00179,	var2=1.00107,	var3=1.00214)	
    249	0.0016448762646223418	(var1=0.99711,	var2=0.99828,	var3=0.99655)	
    250	0.00024109618308367493	(var1=0.9989,	var2=0.99934,	var3=0.99868)	
    251	9.251342440500468e-05	(var1=1.00068,	var2=1.00041,	var3=1.00081)	
    252	1.881037727283135e-10	(var1=1.0,	var2=1.0,	var3=1.0)	
    253	6.839741561293433e-15	(var1=1.0,	var2=1.0,	var3=1.0)	
    254	6.691625214211994e-15	(var1=1.0,	var2=1.0,	var3=1.0)	
    255	6.692668301616802e-15	(var1=1.0,	var2=1.0,	var3=1.0)	
    256	6.692668865547536e-15	(var1=1.0,	var2=1.0,	var3=1.0)	
    257	0.09059829142686093	(var1=1.12253,	var2=1.26343,	var3=1.60337)	
    258	0.13032930278479304	(var1=0.82943,	var2=0.68499,	var3=0.46585)	
    259	0.023484668957449988	(var1=0.93106,	var2=0.86544,	var3=0.74693)	
    260	0.011302324140043405	(var1=1.04514,	var2=1.09342,	var3=1.19761)	
    261	5.474616000254614e-05	(var1=0.99678,	var2=0.99349,	var3=0.98691)	
    262	1.0475054954015859e-07	(var1=0.99986,	var2=0.99972,	var3=0.99943)	
    263	1.1182355382286338e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    264	1.9574993633866026e-16	(var1=1.0,	var2=1.0,	var3=1.0)	
    265	1.9617737704165209e-16	(var1=1.0,	var2=1.0,	var3=1.0)	
    266	1.9662581007496488e-16	(var1=1.0,	var2=1.0,	var3=1.0)	
    267	2.063159267500598e-13	(var1=1.0,	var2=1.0,	var3=1.0)	
    268	295.2492426794891	(var1=1.0,	var2=1.0,	var3=2.71828)	
    269	64.27418829249157	(var1=1.0,	var2=1.0,	var3=0.19829)	
    270	21.25181481505033	(var1=1.0,	var2=1.0,	var3=0.539)	
    271	21.637593608860218	(var1=1.0,	var2=1.0,	var3=1.46516)	
    272	1.2817664066346002	(var1=1.0,	var2=1.0,	var3=0.88678)	
    273	0.28365550013003127	(var1=1.0,	var2=1.0,	var3=1.05326)	
    274	0.0010209948674819377	(var1=1.0,	var2=1.0,	var3=0.9968)	
    275	6.619130314300672e-07	(var1=1.0,	var2=1.0,	var3=0.99992)	
    276	1.6996742017436959e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    277	1.2854270001320846e-17	(var1=1.0,	var2=1.0,	var3=1.0)	
    278	1.2884591669368082e-17	(var1=1.0,	var2=1.0,	var3=1.0)	
    279	1.293512947234887e-17	(var1=1.0,	var2=1.0,	var3=1.0)	
    280	0.0006359218265768335	(var1=1.00179,	var2=1.00107,	var3=1.00214)	
    281	0.001644868477120546	(var1=0.99711,	var2=0.99828,	var3=0.99655)	
    282	0.0002410931333813359	(var1=0.9989,	var2=0.99934,	var3=0.99868)	
    283	9.251535602681285e-05	(var1=1.00068,	var2=1.00041,	var3=1.00082)	
    284	1.8810007825574122e-10	(var1=1.0,	var2=1.0,	var3=1.0)	
    285	1.5234099467936265e-16	(var1=1.0,	var2=1.0,	var3=1.0)	
    286	6.34333011831363e-18	(var1=1.0,	var2=1.0,	var3=1.0)	
    287	6.343993140325057e-18	(var1=1.0,	var2=1.0,	var3=1.0)	
    288	6.3439955715871806e-18	(var1=1.0,	var2=1.0,	var3=1.0)	
    289	0.09059835029273014	(var1=1.12253,	var2=1.26343,	var3=1.60337)	
    290	0.1303292624863931	(var1=0.82943,	var2=0.68499,	var3=0.46585)	
    291	0.02348464795908789	(var1=0.93106,	var2=0.86544,	var3=0.74693)	
    292	0.011302342194715972	(var1=1.04514,	var2=1.09342,	var3=1.19761)	
    293	5.474620994414608e-05	(var1=0.99678,	var2=0.99349,	var3=0.98691)	
    294	1.0474874198035864e-07	(var1=0.99986,	var2=0.99972,	var3=0.99943)	
    295	1.1174586408173103e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    296	1.6704792755897435e-19	(var1=1.0,	var2=1.0,	var3=1.0)	
    297	1.4457346156911087e-19	(var1=1.0,	var2=1.0,	var3=1.0)	
    298	1.4533302908425346e-19	(var1=1.0,	var2=1.0,	var3=1.0)	
    299	1.4533329734864395e-19	(var1=1.0,	var2=1.0,	var3=1.0)	
    300	1.7505695095377394e-16	(var1=1.0,	var2=1.0,	var3=1.0)	
    301	295.24924416324774	(var1=1.0,	var2=1.0,	var3=2.71828)	
    302	64.27418892490746	(var1=1.0,	var2=1.0,	var3=0.19829)	
    303	21.251815075710496	(var1=1.0,	var2=1.0,	var3=0.539)	
    304	21.63759362832723	(var1=1.0,	var2=1.0,	var3=1.46516)	
    305	1.2817664180110544	(var1=1.0,	var2=1.0,	var3=0.88678)	
    306	0.28365551291022767	(var1=1.0,	var2=1.0,	var3=1.05326)	
    307	0.0010209948846769965	(var1=1.0,	var2=1.0,	var3=0.9968)	
    308	6.619135775871371e-07	(var1=1.0,	var2=1.0,	var3=0.99992)	
    309	1.6997191534108103e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    310	1.3221232417780401e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    311	1.3021801724926641e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    312	3.421220155407809e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    313	0.0006359219546091228	(var1=1.00179,	var2=1.00107,	var3=1.00214)	
    314	0.0016448682851094847	(var1=0.99711,	var2=0.99828,	var3=0.99655)	
    315	0.00024109305784479078	(var1=0.9989,	var2=0.99934,	var3=0.99868)	
    316	9.251540407921042e-05	(var1=1.00068,	var2=1.00041,	var3=1.00082)	
    317	1.881001464435445e-10	(var1=1.0,	var2=1.0,	var3=1.0)	
    318	1.4595201158471513e-16	(var1=1.0,	var2=1.0,	var3=1.0)	
    319	6.812575045440988e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    320	6.813533484533279e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    321	6.813529307699675e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    322	0.09059835213825823	(var1=1.12253,	var2=1.26343,	var3=1.60337)	
    323	0.1303292612360932	(var1=0.82943,	var2=0.68499,	var3=0.46585)	
    324	0.02348464730608157	(var1=0.93106,	var2=0.86544,	var3=0.74693)	
    325	0.011302342758519915	(var1=1.04514,	var2=1.09342,	var3=1.19761)	
    326	5.474621151934561e-05	(var1=0.99678,	var2=0.99349,	var3=0.98691)	
    327	1.0474868570966901e-07	(var1=0.99986,	var2=0.99972,	var3=0.99943)	
    328	1.1174403840269652e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    329	2.6344066574629286e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    330	3.767150344268938e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    331	3.777366692784683e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    332	3.777556682636742e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    333	6.779352922309785e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    334	3.340124824826803e-19	(var1=1.0,	var2=1.0,	var3=1.0)	
    335	6.66127581578252e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    336	4.0460026566678415e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    337	1.6971913708814172e-22	(var1=1.0,	var2=1.0,	var3=1.0)	
    338	1.7018458084874403e-22	(var1=1.0,	var2=1.0,	var3=1.0)	
    339	1.7010072527964568e-22	(var1=1.0,	var2=1.0,	var3=1.0)	
    340	0.09059835217479639	(var1=1.12253,	var2=1.26343,	var3=1.60337)	
    341	0.13032926120480973	(var1=0.82943,	var2=0.68499,	var3=0.46585)	
    342	0.023484647290527287	(var1=0.93106,	var2=0.86544,	var3=0.74693)	
    343	0.011302342770775936	(var1=1.04514,	var2=1.09342,	var3=1.19761)	
    344	5.47462115460117e-05	(var1=0.99678,	var2=0.99349,	var3=0.98691)	
    345	1.0474868440348028e-07	(var1=0.99986,	var2=0.99972,	var3=0.99943)	
    346	1.1174399698632204e-12	(var1=1.0,	var2=1.0,	var3=1.0)	
    347	2.2748930710625645e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    348	1.7683514883528712e-22	(var1=1.0,	var2=1.0,	var3=1.0)	
    349	1.768246035522473e-22	(var1=1.0,	var2=1.0,	var3=1.0)	
    350	0.0006359219593141911	(var1=1.00179,	var2=1.00107,	var3=1.00214)	
    351	0.0016448682779338206	(var1=0.99711,	var2=0.99828,	var3=0.99655)	
    352	0.0002410930550405004	(var1=0.9989,	var2=0.99934,	var3=0.99868)	
    353	9.251540585186086e-05	(var1=1.00068,	var2=1.00041,	var3=1.00082)	
    354	1.8810014915764005e-10	(var1=1.0,	var2=1.0,	var3=1.0)	
    355	1.459432722432434e-16	(var1=1.0,	var2=1.0,	var3=1.0)	
    356	9.902838342434328e-24	(var1=1.0,	var2=1.0,	var3=1.0)	
    357	1.0037148715473513e-23	(var1=1.0,	var2=1.0,	var3=1.0)	
    358	1.0067668252456365e-23	(var1=1.0,	var2=1.0,	var3=1.0)	
    359	3.2445830103515833e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    360	1.0015054030053625e-20	(var1=1.0,	var2=1.0,	var3=1.0)	
    361	1.6082870759412662e-21	(var1=1.0,	var2=1.0,	var3=1.0)	
    362	3.961907619967029e-22	(var1=1.0,	var2=1.0,	var3=1.0)	
    363	7.552815123722756e-25	(var1=1.0,	var2=1.0,	var3=1.0)	
    364	7.5964687140654235e-25	(var1=1.0,	var2=1.0,	var3=1.0)	
    365	7.631397502796347e-25	(var1=1.0,	var2=1.0,	var3=1.0)	




.. parsed-literal::

      status: 0
     success: True
     message: Optimization terminated successfully.
           x: [1. 1. 1.]
           y: 7.552815123722756e-25
      n_eval: 365
      n_iter: 11



And we found our [1, 1, 1] solution!

Combination of global and local optimizers
------------------------------------------

There is a special class for combination of global and local optimizers
- ``gadma.optimizers.GlobalOptimizerAndLocalOptimizer``:

.. code:: ipython3

    opt1 = get_global_optimizer("Genetic_algorithm")
    opt2 = get_local_optimizer("BFGS_log")
    opt = GlobalOptimizerAndLocalOptimizer(opt1, opt2)

.. code:: ipython3

    help(opt.optimize)


.. parsed-literal::

    Help on method optimize in module gadma.optimizers.combinations:
    
    optimize(f, variables, args=(), global_num_init=50, X_init=None, Y_init=None, local_options={}, linear_constrain=None, global_maxiter=None, local_maxiter=None, global_maxeval=None, local_maxeval=None, verbose=0, callback=None, eval_file=None, report_file=None, save_file=None, restore_file=None, restore_points_only=False, global_x_transform=None, local_x_transform=None) method of gadma.optimizers.combinations.GlobalOptimizerAndLocalOptimizer instance
        :param f: Objective function.
        :type f: func
        :param variables: List of objective function variables.
        :type variables: list of class:`gadma.utils.VariablesPool`
        :param args: Arguments of `f`.
        :type args: tuple
        :param global_num_init: Number of initial points for global optimizer.
        :type global_num_init: int
        :param X_init: List of initial vectors.
        :type X_init: list
        :param Y_init: List of values of target function on points of `X_init`.
        :type Y_init: list
        :param local_options: Options for local optimizer.
        :type local_options: dict
        :param linear_constrain: Linear constrain on variables.
        :type linear_constrain: :class:`gadma.optimizers.LinearConstrain`
        :param global_maxiter: Maximum number of global optimizer iterations
                               to run.
        :type global_maxiter: int
        :param global_maxeval: Maximum number of function evaluation during
                               global optimization.
        :type global_maxeval: int
        :param local_maxiter: Maximum number of local optimizer iterations
                              to run.
        :type local_maxiter: int
        :param local_maxeval: Maximum number of function evaluation during
                              local optimization.
        :type local_maxeval: int
        :param verbose: Varbosity of reports. If 0 then no output.
        :type verbose: int
        :param callback: callback to run after each iteration of both
                         optimizers.
        :type callback: func
        :param eval_file: File to save of objective function evaluations.
        :type eval_file: str
        :param report_file: File to save report each `verbose` iteration. If
                            None and `verbose` > 0 then report will be printed
                            to stdout.
        :type report_file: str
        :param save_file: File to save information during the run.
        :type save_file: str
        :param restore_file: File to restore previous run that was saved by
                             :meth:`save` method.
        :type restore_file: str
        :param restore_points_only: Restore run last results and run again from
                                    it.
        :type restore_points_only: bool
        :param global_x_transform: Transformation of vectors after restore
                                   before run of global optimizer.
        :type global_x_transform: func
        :param local_x_transform: Transformation of vectors after restore
                                  before run of local optimizer.
        :type local_x_transform: bool
    


.. code:: ipython3

    f = rosen
    # Lets add discrete variable - it will be optimized by global optimization and fixed during local.
    var4 = DiscreteVariable('var4', [-1, 1])
    variables = [var1, var2, var3, var4]
    res = opt.optimize(f, variables, global_maxiter=1000, local_maxiter=50)
    print(res)


.. parsed-literal::

      status: 2
     success: False
     message: GLOBAL OPTIMIZATION: CONVERGENCE: NO IMPROVEMENT DURING 100 ITERATIONS; LOCAL OPTIMIZATION: Desired error not necessarily achieved due to precision loss.
           x: [0.999999992799917 0.999999997047826 0.9999999907298661 1]
           y: 4.8759147986991476e-14
      n_eval: 2775
      n_iter: 304
    


So this class is much simplier to use.

