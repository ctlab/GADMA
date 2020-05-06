.. _sec_api_data:

====
Data
====

GADMA supports several data types for demographic inference.

One should create instance of some subclass of :class:`gadma.DataHolder`
and read data from the saved filename with some engine of
:class:`gadma.Engine` class by calling :func:`gadma.Engine.read_data`
method.


*************************
Base class DataHolder
*************************

.. autoclass:: gadma.DataHolder
    :members:

********
SFS Data
********

.. autoclass:: gadma.SFSDataHolder
    :show-inheritance:

********
VCF Data
********


.. autoclass:: gadma.VCFDataHolder
    :show-inheritance:
