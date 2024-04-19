Installation
==============

Dependencies
-------------

**The current version of GADMA (greater than 2.0.0) supports Python3 only**. `Older versions <https://github.com/ctlab/GADMA/releases/tag/1.0.2>`_ supported Python 2 as well but not any more.

Below we hightlight GADMA dependencies. All of them could be easily installed using requirements files which are specified in the brackets before lists. To install dependencies from requerement file run:

.. code-block:: console

    $ pip -r requirements_file

GADMA requires the following dependencies (`requirements/minimal.txt`):

* Python3
* NumPy (>= 1.2.0)
* Scipy (>= 0.6.0)
* ruamel.yaml (<0.18.0)
* ``dadi`` (>= 1.7.0)
* ``moments`` (>= 1.0.0)
* ``momi``
* ``moments.LD`` (is installed alongside with ``moments``)
* nlopt (for ``dadi``)
* Cython (for ``moments``)
* mpmath (for ``moments``)

To draw demographic models one should also install the following packages (`requirements/minimal.txt`):

* matplotlib (>= 0.98.1, <3.5)
* Pillow (>= 4.2.1) - optional
* ``moments`` (>= 1.0.0)

To use ``demes`` engine to draw models (`requirements/engines.txt`):

* demes
* demesdraw

To calculate confidence intervals one should install (`requirements/minimal.txt`):

* pandas

To run Bayesian optimization `smac` of specified version is requered (`requirements/bayes_opt.txt`):

* scikit-optimize
* configspace
* smac (**==0.13.1**)

.. note::
    ``momi`` package sometimes is not installed correctly for Windows and MacOS. If ``momi`` is not available please install it manually following the installation instructions in `momi's manual <https://momi2.readthedocs.io/en/latest/installation.html#>`_.

.. note::
    ``momentsLD`` - the extension of ``moments``, it is installed together with ``moments``.

Getting help for engine installation
------------------------------------
If there are some troubles installing the engine, please, first of all check the table below for the ability to install this engine on your system. You are always welcome to `open an issue <https://github.com/ctlab/GADMA/issues#>`_ on GitHub for getting help.

GADMA has automatic tests on GitHUb for engines on different systems (Linux, Windows, MacOS). The following table indicates (according to our tests) if engine could be installed on specified system:

.. list-table::
   :header-rows: 1

   * - Feature
     - ``dadi``
     - ``moments``
     - ``momi``
     - ``momentsLD``

   * - Linux
     - ✅
     - ✅
     - ✅
     - ✅
   * - Windows
     - ✅
     - ✅
     - ❌
     - ✅
   * - MacOS
     - ✅
     - ✅
     - ✅
     - ✅

Installing the latest release
------------------------------

The latest release of GADMA can be easily installed via ``pip`` or ``conda`` (``bioconda``):

.. code-block:: console

    $ pip install gadma

or

.. code-block:: console

    $ conda install -c bioconda gadma


.. warning::
    Installation via ``pip`` and ``conda`` will not install the ``moments`` library. To install it one should run:

    .. code-block:: console

        $ pip install git+https://github.com/MomentsLD/moments.git

    or

    .. code-block:: console

        $ conda install -c bioconda moments

Troubleshooting
---------------

If you experience problems with dependencies, we recommend to create an empty `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#>`_:

.. code-block:: console

    $ conda create -n gadma_env python=3.10
    $ conda activate gadma_env

It is possible to install versions that are used for testing by downloading file `minimal.txt` from `here <https://github.com/ctlab/GADMA/blob/master/requirements/minimal.txt#>`_ and install requirements using:

.. code-block:: console

    $ pip install -r minimal.txt
    $ pip install gadma

For **MacOS with M processor** we suggest the following recipe (credit to `@Enricobazzi <https://github.com/ctlab/GADMA/issues/82>`_):

.. code-block:: console

    $ pip install git+https://github.com/MomentsLD/moments.git
    $ conda install -c conda-forge dadi
    $ conda install -c conda-forge scikit-allel
    $ pip install gadma
    $ pip uninstall ruamel.yaml
    $ pip install "ruamel.yaml<0.18.0"
    $ pip uninstall matplotlib
    $ pip install "matplotlib<3.5"

Manual installation
-----------------------------

Some features are added to the GADMA project but are not released yet. One can install GADMA directly from the repository.

First clone the repository:

    .. code-block:: console

        $ git clone https://github.com/ctlab/GADMA.git
        $ cd GADMA

Dependencies could be installed either automatically or manually.

Automatic mode
**************

One could install everything with the ``install`` script:

.. code-block:: console

    $ ./install

Full-manual mode
****************

Install dependencies manually:

    * NumPy
        .. code-block:: console

            $ pip install numpy

    * Scipy
        .. code-block:: console

            $ pip install scipy

    * ruamel.yaml
        .. code-block:: console

            $ pip install ruamel.yaml

    * ``dadi``, nlopt
        .. code-block:: console

            $ pip install dadi

    * ``moments``, mpmath, Cython
        .. code-block:: console

            $ pip install --upgrade Cython
            $ pip install mpmath
            $ pip install git+https://github.com/MomentsLD/moments.git

    * ``momi``
        .. code-block:: console

            $ pip install momi

    * matplotlib
        .. code-block:: console

            $ pip install matplotlib

    * Pillow
        .. code-block:: console

            $ pip install Pillow

    * pandas
        .. code-block:: console

            $ pip install pandas

3) Install GADMA
    .. code-block:: console

        $ pip install .

Verifying installation
-------------------------

To verify the installation, run:

.. code-block:: console

    $ gadma --test


If the installation was successful, one will find the following information at the end:

.. code-block:: console

    --Finish pipeline--

    --Test passed correctly--
    Thank you for using GADMA!

