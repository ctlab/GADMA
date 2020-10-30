Engines (engines package)
=========================

GADMA supports several engines of demographic inference.
Additional engine could be implemented by creating new subclass of class
:class:`.Engine` and register it with function :func:`.register_engine`.

Base class Engine
*********************

.. automodule:: gadma.engines.engine
   :members: register_engine, get_engine, all_engines, Engine
   :undoc-members:
   :show-inheritance:

Class DadiOrMomentsEngine
*************************

.. automodule:: gadma.engines.dadi_moments_common
   :members:
   :undoc-members:
   :show-inheritance:

Dadi engine
***********

.. automodule:: gadma.engines.dadi_engine
   :members:
   :undoc-members:
   :show-inheritance:

Moments engine
**************

.. automodule:: gadma.engines.moments_engine
   :members:
   :undoc-members:
   :show-inheritance:
