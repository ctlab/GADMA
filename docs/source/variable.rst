.. _sec_api_variable:

=========
Variables
=========

To construct demographic history one should choose its parameters. Values of
this parameters or variables will be found by optimization algorithms.

*******************
Base class Variable
*******************

.. autoclass:: gadma.Variable
    :members:

************************
Class ContinuousVariable
************************

.. autoclass:: gadma.ContinuousVariable
    :show-inheritance:
    :members: get_bounds, get_possible_values

**********************
Class DiscreteVariable
**********************

.. autoclass:: gadma.DiscreteVariable
    :show-inheritance:
    :members: get_bounds, get_possible_values

*****************************
Class PopulationSizeVariable
*****************************

.. autoclass:: gadma.PopulationSizeVariable
    :show-inheritance:

******************
Class TimeVariable
******************

.. autoclass:: gadma.TimeVariable
    :show-inheritance:

***********************
Class MigrationVariable
***********************

.. autoclass:: gadma.MigrationVariable
    :show-inheritance:

***********************
Class SelectionVariable
***********************

.. autoclass:: gadma.SelectionVariable
    :show-inheritance:

**********************
Class FractionVariable
**********************

.. autoclass:: gadma.FractionVariable
    :show-inheritance:

**************
Dynamics
**************

This section is about such values as dynamics of population change.

.. autoclass:: gadma.utils.variables.Dynamic
    :members: func_str, __str__, _inner_func

.. autoclass:: gadma.utils.variables.Exp
    :show-inheritance:
    :members: __str__, _inner_func

.. autoclass:: gadma.utils.variables.Lin
    :show-inheritance:
    :members: __str__, _inner_func

.. autoclass:: gadma.utils.variables.Sud
    :show-inheritance:
    :members: __str__, _inner_func

*********************
Class DynamicVariable
*********************

.. autoclass:: gadma.DynamicVariable
    :show-inheritance:
    :members: get_func_from_value, get_bounds
