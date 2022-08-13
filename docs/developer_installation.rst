=========================================
Installation for development of Firecrown
=========================================

.. role:: bash(code)
   :language: bash

.. warning::

    These instructions do not work for Macs with M1 processors.
    For installation on that platform, please use the :doc:`Apple M1 installation instructions<apple_m1_instructions>`.

To do development work on Firecrown you need both the Firecrown source code and all the packages upon which Firecrown depends or is used with.
To get the Firecrown source code you will need to clone the Firecrown repository.
Most (but not all) of the dependencies of Firecrown are available through conda.

These instructions include details on how to obtain the samplers used with Firecrown.
This is important because if you are doing development it is important to make sure what you change or add works with both of the supported samplers.

You only need to clone the Firecrown repository once, and to create the conda environment once.
Every time you want to do development in a new shell session you will need to activate the conda environment and set the environment variables.


Clone the Firecrown repository
==============================

Choose a directory in which to work.
In this directory, you will be cloning the Firecrown repository and later building some of the non-Firecrown code that is not installable through conda.

.. code:: bash

    git clone https://github.com/LSSTDESC/firecrown.git
    

Installation of dependencies
============================

These instructions will create a new conda environment containing all the packages used in development.
This includes testing and code verification tools used during the development process.

It is best to execute these commands while in the same directory as you were in when you cloned the Firecrown repository above.
The `cosmosis-build-standard-library` command below will clone and then build the CosmoSIS Standard Library.
This will create a directory `cosmosis-standard-library` in whatever is your current directory when you execute the command.

.. code:: bash

    conda create --name firecrown_developer -c conda-forge cosmosis cosmosis-build-standard-library sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black pytest coverage
    conda activate firecrown_developer
    python -m pip install cobaya
    source ${CONDA_PREFIX}/bin/cosmosis-configure
    cosmosis-build-standard-library
    

Setting your environment for development
========================================

Each time you want to do development in a new shell session you need to activate the conda environment and set some environment variables.

Begin by `cd`-ing to the working directory you used during the installation (which should contain both `firecrown` and `cosmosis-standard-library` directories).

.. code:: bash

    cd /path/to/directory/you/created/above
    conda activate firecrown_developer
    export CSL_DIR=${PWD}/cosmosis-standard-library
    export FIRECROWN_DIR=${PWD}/firecrown
    export PYTHONPATH=${FIRECROWN_DIR}/build/lib
    export FIRECROWN_SITE_PACKAGES=${COSMOSIS_SRC_DIR}/..

To build the Firecrown code you should be in the Firecrown directory:

.. code:: bash

    cd ${FIRECROWN_DIR}
    python setup.py build

The tests can be run with :bash:`pytest`, after building:

.. code:: bash

    pytest

Examples can be run by `cd`-ing into the specific examples directory and following the instructions in the local README file.
