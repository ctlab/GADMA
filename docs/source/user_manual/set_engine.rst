Specifying an engine
=======================

GADMA uses either ``dadi`` or ``moments`` to simulate expected AFS from the demographic model. ``moments`` is used by default. When using ``dadi``, it is recommended to check the value of the ``Pts`` option in the ``params_file``. ``Pts`` is a sequence of three numbers, each of which is equal to the number of points in grid size. The greater the numbers are, the more accurate numerical solution of a partial differential equation ``dadi`` will give. However, finding such a more accurate solution takes more time. By default, GADMA takes ``Pts : n, n + 10, n + 20``, where ``n`` is the largest sample size among the populations of interest.

The ``moments`` library does not need ``Pts`` to be specified. To change moments to the ``dadi`` engine, specify option in the parameters file:

.. code-block:: none

   # param file
   ...
   Engine : moments
   ...

or to use ``dadi``:

.. code-block:: none

   # param file
   ...
   Engine : dadi
   Pts: [40, 50, 60]
   ...

.. note::
    ``Pts`` option is also used for ``moments`` engine as well: for generation of python code for ``dadi``. So if one want to use ``dadi``'s code of final models after run then maybe ``Pts`` argument must be set too.
