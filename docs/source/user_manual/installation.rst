Installation
==============

Dependencies
-------------

**Current version of GADMA (later than 2.0.0) supports Python3 only**. Older versions supported Python 2 as well but not any more.

GADMA requires the following dependencies:

* Python3
* NumPy (>= 1.2.0)
* Scipy (>= 0.6.0)
* ruamel.yaml
* ``dadi`` (>= 1.7.0) or/and ``moments`` (>= 1.0.0)
* nlopt (for ``dadi``)
* Cython (for ``moments``)
* mpmath (for ``moments``)

To draw demographic models one should install the following:

* matplotlib (>= 0.98.1)
* Pillow (>= 4.2.1) - optional
* ``moments`` (>= 1.0.0)

To calculate Confidence Intervals:

* pandas

Installing the latest release
------------------------------

The latest release of BoTorch is easily installed either via ``pip``:

.. code-block:: console

    $ pip install gadma

.. warning::
    Installation via ``pip`` will not install ``moments`` library. To install it one should run:

    .. code-block:: console

        $ pip install --upgrade Cython
        $ pip install mpmath
        $ git clone https://bitbucket.org/simongravel/moments/
        $ cd moments
        $ python3 setup.py install
        $ cd ..

Manual installation
-----------------------------

Some features are added to GADMA project but are not released yet. One can try to install GADMA directly from the repository.

First clone repository:

    .. code-block:: console

        $ git clone https://github.com/ctlab/GADMA.git
        $ cd GADMA

Dependencies could be installed with one of two modes.

Auto mode
**********

One could install everything with ``install`` script:

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
            $ git clone https://bitbucket.org/simongravel/moments/
            $ cd moments
            $ python3 setup.py install
            $ cd ..

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

        $ python3 setup.py install

Verifying installation
-------------------------

To verify the installation, run a test:

.. code-block:: console

    $ gadma --test


If the installation was successful, one will find the following information at the end:

.. code-block:: console

    --Finish pipeline--

    --Test passed correctly--
    Thank you for using GADMA!

