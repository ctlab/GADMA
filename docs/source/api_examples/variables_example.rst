Variables
=========

GADMA has several variable classes in ``gadma.utils`` for different
types of variables. Variables are used in models and for optimization
runs. Base classes are:

-  ``ContinuousVariable`` - continuous variables.
-  ``DiscreteVariable`` - variable with discrete domain.

To create object one should set ``name``, ``domain`` and reand\_gen
(optional):

.. code:: ipython3

    from gadma import *
    
    cont_var = ContinuousVariable('var_name', domain=[-1, 1])
    
    rand_gen = lambda domain: np.random.choice(domain)
    disc_var = DiscreteVariable('some_var', domain=['one', 'two', 'three'], rand_gen=rand_gen)

Every variable also has random generator for its valuesa and
``get_possible_values``, ``get_bounds`` methods.

.. code:: ipython3

    print(f"Random value of cont_var: {cont_var.resample()}")
    print(f"Random value of cont_var: {disc_var.resample()}")


.. parsed-literal::

    Random value of cont_var: 0.8983341063764418
    Random value of cont_var: two


There are also special set of variables for demographic parameters with
specified default domain and special random generators:

-  ``PopulationSizeVariable`` - variable for holding populations size.
-  ``TimeVariable`` - variable for holding time migration rate.
-  ``MigrationVariable`` - variable for holding time.
-  ``SelectionVariable`` - variable for holding selection.
-  ``FractionVariable`` - variable for holding fraction.
-  ``DynamicVariable`` - variable for holding dynamic of size change.

The last variable class is discrete variable class.

.. code:: ipython3

    nu = PopulationSizeVariable("nu")
    t = TimeVariable("t")
    m = MigrationVariable("m")
    g = SelectionVariable("g")
    f = FractionVariable("f")
    d = DynamicVariable("d")
    
    print(f"Domain of PopulationSizeVariable: {nu.domain}")
    print(f"Domain of TimeVariable: {t.domain}")
    print(f"Domain of MigrationVariable: {m.domain}")
    print(f"Domain of SelectionVariable: {g.domain}")
    print(f"Domain of FractionVariable: {f.domain}")
    print(f"Domain of DynamicVariable: {d.domain}")


.. parsed-literal::

    Domain of PopulationSizeVariable: [1.e-02 1.e+02]
    Domain of TimeVariable: [1.e-15 5.e+00]
    Domain of MigrationVariable: [ 0 10]
    Domain of SelectionVariable: [1.e-15 1.e+01]
    Domain of FractionVariable: [0.001 0.999]
    Domain of DynamicVariable: ['Sud' 'Lin' 'Exp']


**Variables pool**

There is a special class for keeping list of unique variables. All
variables in variable pool should have different names otherwise there
will be an error.

.. code:: ipython3

    # Create variables for pool
    nu = PopulationSizeVariable("nu")
    same_nu = PopulationSizeVariable("nu")
    t = TimeVariable("t")
    
    # create pool from unique variables
    pool = VariablePool([nu, t])
    print("Variable pool from variables with unique names was created")
    
    try:
        pool.append(same_nu)
    except NameError as e:
        print("Variable pool was not updates as there is the variable with the same name:\n", e)



.. parsed-literal::

    Variable pool from variables with unique names was created
    Variable pool was not updates as there is the variable with the same name:
     VariablePool has already a Variable with the same name (nu).


