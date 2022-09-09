Specifying a model
==================

.. toctree::
   :hidden:

   theta0
   set_model_struct
   set_model_custom

.. admonition:: Related options

    * **Common options**:

      * ``Theta0``
      * ``Mutation rate``
      * ``Recombination rate``

    Please read more about ``Theta0`` and ``Mutation rate`` :ref:`here <theta>`.

Types of demographic models
----------------------------

GADMA could infer two base types of demographic models:

* `Demographic model with structire <set_model_struct.rst>`__ (up to 3 populations). It is a more flexible model type as dynamics of population size change could be inferred and it has a lot of options for parameters.

.. admonition:: Related options

    * **Base options**:

      * ``Initial structure``
      * ``Final structure`` (optional)

    * **Additional options**:

      * ``Only sudden`` fixes constant population sizes instead of dynamics inference.
      * ``No migrations`` disables all migrations.
      * ``Symmetric migrations`` makes all migrations symmetric.
      * ``Migration masks`` enables/disables migrations selectively.
      * ``Split fractions`` infers fractions in which ancestral population is split.
      * ``Inbreeding`` infers inbreeding coefficients (only for ``dadi`` engine).
      * ``Selection`` infers selection coefficients.
      * ``Ancestral size as parameter`` disables multinomial approach of ``dadi`` and ``moments`` engines when ancestral size is inferred implicitly.
      * ``Upper bound of first split`` limits upper bound of the most ancient split.
      * ``Upper bound of second split`` limits upper bounds of next to the most ancient split.

* `Custom demographic model <set_model_custom.rst>`__. It is a usual user-specified model like in ``dadi``, ``moments`` and other tools for demographic inference. Using such a model will give more control over parameters and could be used for inference of more than 3 populations but is less flexible.

.. admonition:: Related options

    * **Base options**:

      * ``Custom filename`` - path to file with specified demographic model.

    * **Additional options**:

      * ``Lower bound`` - list of lower bounds for model parameters.
      * ``Upper bound`` - list of upper bounds for model parameters.
      * ``Parameters identifiers`` - names/identifiers of the parameters.
