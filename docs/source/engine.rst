.. _sec_api_engine:

=================================
Engines for demographic inference
=================================

GADMA supports several engines of demographic inference.
Additional engine could be implemented by creating new subclass of class
:class:`.Engine` and register it with function :func:`.register_engine`.

.. autofunction:: gadma.register_engine

.. autofunction:: gadma.get_engine

.. autofunction:: gadma.all_engines


*********************
Abstract class Engine
*********************

.. autoclass:: gadma.Engine
    :members:

***********
Dadi engine
***********

.. autoclass:: gadma.DadiEngine
    :show-inheritance:
    :members: id, supported_event_classes, supported_data_classes, data_type, read_data, _get_kwargs, _dadi_inner_func
