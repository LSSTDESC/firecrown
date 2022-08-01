
=======================
Installation Quickstart
=======================

.. role:: bash(code)
   :language: bash

.. warning::
    Warning

    The conda packaging of Firecrown is not yet completed.
    The following instructions show how we intend the installation to work;
    however, for now please follow the "Installation of dependencies for development"
    instructions below.

The easiest way to get started is by using conda. We recommend creating a conda
environment for your use.

This will install Firecrown as well as the samplers that are currently supported.

.. code:: bash

    conda create --name fc -c conda-forge firecrown

Installation of dependencies for development
============================================

As with the quickstart installation, you need to choose how you want to use the
Firecrown code you will be working on. Simultaneous development of either Cobaya
or CosmoSIS and Firecrown is beyond the scope of these instructions.

Firecrown alone
---------------

.. code:: bash

    conda create --name for_fc -c conda-forge sacc pyccl fitsio flake8 pylint black pytest coverage

Firecrown with CosmoSIS
-----------------------

Firecrown supports CosmoSIS 2.x.
The conda installation of CosmoSIS does not include the CosmoSIS Standard Library (CSL), but almost all use of CosmoSIS will include the use of parts of the CSL.
These instructions include the instructions for building the CSL.

.. code:: bash

    conda create --name for_fc_cosmosis -c conda-forge cosmosis cosmosis-build-standard-library sacc pyccl fitsio flake8 pylint black pytest coverage
    # Note that the following will clone the CSL repository and build it in your current working directory.
    # This should be done *outside* of the directory tree managed by conda, and *outside* of the `firecrown` directory.
    conda activate for_fc_cosmosis
    source ${CONDA_PREFIX}/bin/cosmosis-configure
    cosmosis-build-standard-library
    export CSL_DIR=${PWD}/cosmosis-standard-library

Firecrown with Cobaya
---------------------

.. code:: bash

    conda create --name for_fc_cobaya -c conda-forge sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black pytest coverage
    conda activate for_fc_cobaya
    # Not all cobaya dependencies can be installed with conda.
    python -m pip install cobaya

Firecrown with both CosmoSIS and Cobaya
---------------------------------------

.. code:: bash

    conda create --name for_fc_both -c conda-forge cosmosis cosmosis-build-standard-library sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black pytest coverage
    conda activate for_fc_both
    python -m pip install cobaya
    # Note that the following will clone the CSL repository and build it in your current working directory.
    # This should be done *outside* of the directory tree managed by conda, and *outside* of the `firecrown` directory.
    source ${CONDA_PREFIX}/bin/cosmosis-configure
    cosmosis-build-standard-library
    export CSL_DIR=${PWD}/cosmosis-standard-library

Getting Firecrown for development
=================================

To install the package in developer mode, start by cloning the git repo.
Activate whichever conda environment you created for your development effort.

1. Define :bash:`CSL_DIR` appropriately if you are going to use CosmoSIS.
2. Define :bash:`FIRECROWN_DIR` to be the directory into which you have cloned the :bash:`firecrown` repository.

If you do not have :bash:`PYTHONPATH` defined: define :bash:`PYTHONPATH=${FIRECROWN_DIR}/build/lib`

If you have :bash:`PYTHONPATH` defined: define :bash:`PYTHONPATH=${FIRECROWN_DIR}/build/lib:${PYTHONPATH}`

In the active environment, you can build Firecrown by:

.. code:: bash

    cd ${FIRECROWN_DIR}
    python setup.py build

The tests can be run with :bash:`pytest`, after building:

.. code:: bash

    pytest

Some tests are marked as *slow*; those are skipped unless they are requested
using :bash:`--runslow`:

.. code:: bash

    pytest --runslow
