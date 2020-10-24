Specifying an engine
=======================

GADMA uses either ``dadi`` or ``moments`` to simulate expected AFS from the demographic model. ``moments`` is used by default. To use ``dadi`` it is recommended to check the value of the ``Pts`` option in the ``params_file``. ``Pts`` is sequence of three numbers, each of which is equal to the number of points in grid size. The greater the numbers are, the more accurate dadi numerical solution of partial differential equation is. However, finding such a solution take more time. By default, GADMA takes ``Pts : n, n + 10, n + 20``, where ``n`` — is the largest sample size among populations of interest.

``moments`` library does not need ``Pts`` to be specified. To change moments to ``dadi`` engine, specify option in the parameters file:

.. code-block::

   # param file
   ...
   Engine : moments
   ...

or to use ``dadi``:

.. code-block::

   # param file
   ...
   Engine : dadi
   Pts: [40, 50, 60]
   ...

.. note::
    Using ``moments`` engine uses ``Pts`` settings for generation of ``dadi``'s code. So if one want to use ``dadi``'s code then maybe ``Pts`` argument must be set too.