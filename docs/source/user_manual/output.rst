Output
========

.. note::
    Output has changed a little in GADMA version 2.

GADMA puts all files to the directory that user set through ``-o/--output`` command-line option:

.. code-block:: console

    $ gadma -o output_dir -i input_fs.fs


or through ``Output directory`` option in the parameter file:

.. code-block:: none

    # param_file
    ...
    Output directory : output_dir
    ...

Stdout and log file
---------------------

GADMA prints its progress about every minute in stdout and in ``output_dir/GADMA.log`` file:

.. code-block:: none

    [hhh:mm:ss]
    All best logLL models:
    GA number       logLL            AIC             Model

    All best AIC models:
    GA number       logLL            AIC             Model

    --Best model by log likelihood--
    Log likelihood:       	Best logLL
    with AIC score:         AIC_score
    Model: 	representation of best logLL model

    --Best model by AIC score--
    Log likelihood:       	logLL
    with AIC score:         Best AIC score
    Model:  representation of best AIC model 

.. note::
    One can set ``Silence`` option in the parameter file to ``True`` to disable output in stdout, file \py{output_dir/GADMA.log} will still have it.


Model representation
*********************

Every model is printed as a line of parameters, each parameter followed by its name in brackets.

Consider designations:
    * ``T`` - time,
    * ``s`` - percent of split,
    * ``nu`` - size of population number \py{i},
    * ``d`` - dynamic of changing of the size of population number \py{i},
    * ``m`` - mutation rate from population \py{i} to population \py{j}.

Dynamic of population size change has string representations: 

    * ``'Sud'`` - sudden change of size for population number \py{i};
    * ``'Lin'`` - linear change of size for population number \py{i};
    * ``'Exp'`` - exponential change of size for population number \py{i}.

Model is printed as sequence of time intervals and splits that are represented in the following way:

    * First period (``NAnc`` --- size of ancestry population):

        .. code-block:: none
    
            [Nanc = VALUE]

    * Split:

        - If split divide population ``X``  of size ``NU`` into two new populations by fraction ``s1``:

            .. code-block:: none
        
                [ X pop split   VALUE (s1) [VALUE_1(s1*NU), VALUE_2((1-s1)*NU)]
            
        - If split divide population ``X``  of size ``NU`` into two new populations without any fraction parameter (Setting ``Split fractions`` is ``False``):

            .. code-block:: none
        
                [ X pop split [VALUE_1(NU_1), VALUE_2(NU_2)]

    * Usual time period:

        - If there is one population:
        
            .. code-block:: none
        
                [ T_VALUE (t), [ NU_VALUE (nu) ], [D_VALUE (dyn)] ]

        - If there are two populations:
        
            .. code-block:: none
        
                [ T_VALUE (t), [ NU1_VALUE (nu1), NU2_VALUE (nu2)], [[None, M12_VALUE(m12)], [M21_VALUE (m21), None]], [D1_VALUE (dyn1), D2_VALUE (dyn2)]]}

        - And similar if there are three populations.

    * At the end ``theta`` could be printed if length of sequence and mutation rate are known.


Also at the end of the string that corresponds to the model there is an information about model's ancestry in the genetic algorithm: 

* 'c' - for model, that is child of crossover,
* 'r' - if it was formed random way,
* 'm' - if it was mutated,
* 'f' - final model of genetic algorithm.

.. note::
    'm' is added as many times as the model is mutated.

**Example of the demographic model for two populations**:

.. code-block:: none

    [Nanc =  7214] [ [ 7211(t1), [17004(nu11)], [Lin(dyn11)] ],	[ 1 pop split   99.85% (s1) [16978.164(s1*nu11), 25.836((1-s1)*nu11)] ],	[ 1365(t2), [12570(nu21), 8922(nu22)], [[0, 6.45e-05(m2_12)], [5.98e-05(m2_21), 0]], [Sud(dyn21), Lin(dyn22)] ] ]	(theta =  2739.60)



Output directory content
--------------------------

For every repeat of the genetic algorithm GADMA creates a new folder in the output directory with corresponding number.

In every folder there is ``GADMA_GA.log``, where every iteration of the algorithm is saved, pictures and generated code of best models are saved in ``pictures`` and ``code`` directories of each run. ``eval_file`` and ``save_file`` have information about evaluations and optimization.

When the genetic algorithm finishes GADMA saves pictures and python code of obtained model in the corresponding folder.

When all GA are executed, the codes are saved in the root directory.

.. code-block:: none

    - <output_dir>
        	- 1
        		GADMA_GA.log
        		- pictures
        		- code
        			- dadi
        			- moments
        		final_best_logLL_model_dadi_code.py
        		final_best_logLL_model_moments_code.py
        		final_best_logLL_model.png
        		eval_file
        		save_file
        	- 2
        		GADMA_GA.log
        		- pictures
        		- code
        			- dadi
        			- moments
        		final_best_logLL_model_dadi_code.py
        		final_best_logLL_model_moments_code.py
        		final_best_logLL_model.png
        		eval_file
        		save_file
        	params
        	extra_params
        	GADMA.log
        	best_logLL_model.png
        	best_logLL_model_dadi_code.py
        	best_logLL_model_moments_code.py
        	best_aic_model.png
        	best_aic_model_dadi_code.py
        	best_aic_model_moments_code.py

Generated code of models
--------------------------

By default, GADMA generates Python code only for final models both for ``dadi`` and ``moments``. However, it can do it every ``N`` iteration of the genetic algorithm. In this case option ``Print models code every N iteration`` should be set in the parameter file. GADMA saves files with code to the ``output_dir/<GA_number>/python_code`` directory. Both ``dadi`` and ``moments`` code are generated and saved in different folders there. 

Each code contains the function of the model, which takes values of the parameters as input, and strings that load observed AFS, simulates expected AFS from the model's function and calculates log-likelihood of two AFS'. The calculated log-likelihood is printed to stdout. For the ``moments`` code, a picture is also drawn.

All code can be run in the following way:

.. code-block:: console

    $ python file_with_code.py

**Example of generated code**

.. code-block:: python

    import moments
    import numpy as np

    def model_func(params, ns):
        	t1, nu11, s1, t2, nu21, nu22, m2_12, m2_21 = params
        	sts = moments.LinearSystem_1D.steady_state_1D(np.sum(ns))
        	fs = moments.Spectrum(sts)
        	nu1_func = lambda t: 1.0 + (nu11 - 1.0) * (t / t1)
        	fs.integrate(tf=t1, Npop=lambda t: [nu1_func(t)], dt_fac=0.01)
        	fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
        	nu2_func = lambda t: ((1 - s1) * nu11) + (nu22 - ((1 - s1) * nu11)) * (t / t2)
        	migs = np.array([[0, m2_12], [m2_21, 0]])
        	fs.integrate(tf=t2, Npop=lambda t: [nu21, nu2_func(t)], m=migs, dt_fac=0.01)
        	return fs

    data = moments.Spectrum.from_file('YRI_CEU.fs')
    ns = data.sample_sizes

    p0 = [0.4998572004354712, 2.357114661127308, 0.9984806062829666,
          0.09461000843655785, 1.7425719794077874, 1.2368394548443258,
          0.9299543753642668, 0.8621179886837054]
    model = model_func(p0, ns)
    ll_model = moments.Inference.ll_multinom(model, data)
    print('Model log likelihood (LL(model, data)): {0}'.format(ll_model))

    theta = moments.Inference.optimal_sfs_scaling(model, data)
    print('Optimal value of theta: {0}'.format(theta))
    theta0 = 0.37976
    Nanc = int(theta / theta0)
    print('Size of ancestral population: {0}'.format(Nanc))

    plot_ns = [4 for _ in ns]  # small sizes for fast drawing
    gen_mod = moments.ModelPlot.generate_model(model_func,
                                               p0, plot_ns)
    moments.ModelPlot.plot_model(gen_mod,
                                 save_file='model_from_GADMA.png',
                                 fig_title='Demographic model from GADMA',
                                 draw_scale=True,
                                 pop_labels=['YRI', 'CEU'],
                                 nref=Nanc,
                                 gen_time=None,
                                 gen_time_units='generations',
                                 reverse_timeline=True)

