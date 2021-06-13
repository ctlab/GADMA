.. _Custom demographic model:

Specifying a model in details
----------------------------------

It is also possible to use the genetic algorithm from GADMA to a proposed model that is defined as a Python function using ``dadi`` or ``moments``. It is the way that ``dadi`` and ``moments`` work with demographic models inference. To understand how to specify a model like that one can read manuals to these packages. 

For example, consider a simple bottleneck model for one population: at time ``TF + TB`` in the past, an equilibrium population goes through a bottleneck of depth ``nuB``, recovering to relative size ``nuF``:

.. code-block:: python

    def bottleneck(params, ns, pts):
        nuB, nuF, TB, TF = params
        xx = dadi.Numerics.default_grid(pts)
    
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.Integration.one_pop(phi, xx, TB, nuB) 
        phi = Integration.one_pop(phi, xx, TF, nuF)
    
        fs = dadi.Spectrum.from_phi(phi, ns, (xx,)) 
        return fs

To run optimization from GADMA one needs to run optimization function just like in ``dadi`` and ``moments``:

.. code-block:: python

    # Import GADMA's optimization function:
    import gadma

    # Specify input data and its parameters:
    data = dadi.Spectrum.from_file("fs_filename.fs")
    ns = data.sample_sizes # size of AFS
    pts = [40,50,60] # grid size for dadi

    # Wrap our bottleneck function:
    func_ex = dadi.Numerics.make_extrap_log_func(bottleneck)
    
    # Specify upper and lower bounds for parameters:
    upper_bound = [100, 100, 3, 3]
    lower_bound = [1e-2, 1e-2, 0, 0]
    
    # Run optimizations:
    # Beginning GADMA optimization
    popt = gadma.Inference.optimize_ga(data, func_ex, engine='dadi', args=pts_l,
                                       p_ids = ['nuB', 'nuF', 'TB', 'TF'],
                                       lower_bound=lower_bound,
                                       upper_bound=upper_bound)
    # Beginning local optimization from dadi
    popt = dadi.Inference.optimize_log(popt, data, func_ex, pts_l,
                                       lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       verbose=len(p0), maxiter=3)

    print('Found parameters: {0}'.format(popt))

.. note::
    As GADMA optimization is a global search algorithm, no initial parameters ``p0`` are set in ``gadma.Inference.optimize_ga`` function. However, it is possible to specify ``X_init`` with ``p0`` as one of known starting points:

    .. code-block:: python

        # Initial parameters can be set too:
        p0 = [0.01, 1.5, 0.2, 0.2]
    
        # Beginning GADMA optimization
        popt = gadma.Inference.optimize_ga(data, func_ex, engine='dadi', args=pts_l, X_init=[p0], 
                                           p_ids = ['nuB', 'nuF', 'TB', 'TF'],
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound)
                                       
.. warning::
    Function ``gadma.Inference.optimize_ga`` changed in GADMA version 2. For full documentation see the API (:ref:`gadma.Inference`).

If one wants to find other parameters for ``gadma.Inference.optimize_ga`` function:

.. code-block:: python

    >>> import gadma
    >>> help(gadma.Inference.optimize_ga)

