Installation
==============

Set up conda environment
------------------------

In order to avoid issues, we recommend to create an empty `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#>`_.
**The current version of GADMA (greater than 2.0.0) supports Python3 only**.
.. code-block:: console

    $ conda create -n gadma_env python=3.10
    $ conda activate gadma_env
    $ conda config --add channels bioconda
    $ conda config --add channels conda-forge


Installing the latest release
------------------------------

The latest release of GADMA can be easily installed via ``pip`` or ``conda`` (``bioconda``):

.. code-block:: console

    $ pip install gadma

or

.. code-block:: console

    $ conda install -c bioconda gadma

This will install GADMA with ``dadi``, ``moments`` and ``momentsLD`` engines.

.. warning::
    Sometimes some dependencies (e.g. ``dadi`` or ``scikit-allel``) fail to install when GADMA is installed using ``pip``. You can install them using ``conda`` before the installation of GADMA:

    .. code-block:: console

        $ conda install dadi scikit-allel

    If you faced an issue for ``moments`` (``moments-popgen``) package on MacOS, try installing it independently using:

    .. code-block:: console

        $ pip install Cython
        $ pip install "setuptools_scm>=8"
        $ pip install --no-build-isolation moments-popgen


If you want to use ``momi2`` engine, please install it:

.. code-block:: console

    $ pip install momi

.. note::
    ``momi`` package sometimes is not installed correctly for Windows and MacOS. If ``momi`` is not available please install it manually following the installation instructions in `momi's manual <https://momi2.readthedocs.io/en/latest/installation.html#>`_.




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


Dependencies
-------------

Below we hightlight GADMA dependencies:

* Python3
* numPy
* scipy
* matplotlib
* Pillow (>= 4.2.1)
* pandas (<= 2.2.2)
* ruamel.yaml (== 0.16.12)
* ``dadi``
* ``moments``
* ``momi``
* ``moments.LD`` (is installed alongside with ``moments``)
* h5py == 3.10.0 (for ``momentsLD``)
* scikit-allel (for ``moments``)

If you wish to use ``moments`` for demographic models plotting you need older
version of ``matplotlib``:


To calculate confidence intervals one should install (`requirements/minimal.txt`):

* pandas

To run Bayesian optimization `smac` of specified version is requered (`requirements/bayes_opt.txt`):

* scikit-optimize
* configspace
* bayesmark
* smac (**==0.13.1**)

Troubleshooting
---------------
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


Manual installation
-----------------------------

Some features are added to the GADMA project but are not released yet. One can install GADMA directly from the repository.

First clone the repository:

.. code-block:: console

    $ git clone https://github.com/ctlab/GADMA.git
    $ cd GADMA

Run installation:

.. code-block:: console

    $ pip install .

