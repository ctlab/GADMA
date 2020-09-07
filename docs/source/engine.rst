.. _sec_api_engine:

=======
Engines
=======

GADMA supports several engines of demographic inference.
Additional engine could be implemented by creating new subclass of class
:class:`.Engine` and register it with function :func:`.register_engine`.

.. autofunction:: gadma.register_engine

.. autofunction:: gadma.get_engine

.. autofunction:: gadma.all_engines


*********************
Base class Engine
*********************

.. autoclass:: gadma.Engine
    :members:

*************************
Class DadiOrMomentsEngine
*************************

.. autoclass:: gadma.engines.DadiOrMomentsEngine
    :show-inheritance:
    :members: id, supported_event_classes, supported_data_classes, data_type, read_data, get_theta, get_N_ancestral_from_theta, get_N_ancestral, draw_sfs_plots, evaluate, get_claic_component


***********
Dadi engine
***********

.. autoclass:: gadma.DadiEngine
    :show-inheritance:
    :members: id, supported_event_classes, supported_data_classes, data_type, read_data, _get_kwargs, _dadi_inner_func


**************
Moments engine
**************

.. autoclass:: gadma.MomentsEngine
    :show-inheritance:
    :members: id, supported_event_classes, supported_data_classes, data_type, read_data, _get_kwargs, _moments_inner_func