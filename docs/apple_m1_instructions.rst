======================================================================
Instructions for installation of Firecrown on Apple M1 and M2 hardware
======================================================================
.. role:: bash(code)
   :language: bash

The ability to install Firecrown on Apple M1 and M2 hardware (not on Intel hardware) is currently incomplete because of difficulties in obtaining a consistent build of the CosmoSIS Standard Library on M1 and M2 Macs, and the unavailability of an M1 or M2 build of CosmoSIS through conda.
Until this is fixed, we recommend the following procedure.
    
.. note::


    This procedure is *not* recommended for other platforms.
    It involves *building* the "head" versions of CosmoSIS and using (parts of) the "head" version of the CosmoSIS Standard Library.
    The normal installation instructions give you a specified version known to be consistent with the rest of the code.
    In addition, because this involves *building* CosmoSIS code, if you have something else in your environment that is accidentally found by the build, you may encounter a difficult-to-debug build failure, or worse yet a subtly inconsistent build that will perform incorrectly only some of the time.

Note that this installation procedure will give you an environment in which you have a copy of the Firecrown code and can do development, including producing pull requests to submit code back to Firecrown.

To do development work on Firecrown you need both the Firecrown source code and all the packages upon which Firecrown depends or is used with.
To get the Firecrown source code you will need to clone the Firecrown repository.
Most (but not all) of the dependencies of Firecrown are available through conda.
It is not currently possible to build the CosmoSIS Standard Library on M1 or M2 Macs.
However, you will be able to use the portions of the CosmoSIS Standard Library that do not require compilation.
This includes CAMB, but does not include many other modules.

These instructions include details on how to obtain the samplers used with Firecrown.
This is important because if you are doing development it is important to make sure what you change or add works with both of the supported samplers.

You only need to clone the Firecrown repository once, and to create the conda environment once.
Every time you want to do development in a new shell session you will need to activate the conda environment and set the environment variables.

Clone the Firecrown repository
==============================

Choose a directory in which to work.
In this directory, you will be cloning the Firecrown repository and later building some of the non-Firecrown code that is not installable through conda.

.. code:: bash

    cd /directory/for/firecrown/work
    git clone https://github.com/LSSTDESC/firecrown.git
    

Installation of dependencies
============================

These instructions will create a new conda environment containing all the packages used in development.
This includes testing and code verification tools used during the development process.

It is best to execute these commands while in the same directory as you were in when you cloned the Firecrown repository above.
Note that we do not build the CosmoSIS Standard Library.
Only the pure-python parts of the CSL will be available for use.

.. code:: bash

    cd /directory/for/firecrown/work
    conda create --name firecrown_developer -c conda-forge sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black pytest coverage
    conda activate firecrown_developer
    export CC=clang CXX=clang++ FC=gfortran
    python -m pip install cosmosis cobaya
    source cosmosis-configure
    git clone https://github.com/joezuntz/cosmosis-standard-library.git
    

Setting your environment for development
========================================

Each time you want to do development in a new shell session you need to activate the conda environment and set some environment variables.

Begin by `cd`-ing to the working directory you used during the installation (which should contain both `firecrown` and `cosmosis-standard-library` directories).

.. code:: bash

    cd /directory/for/firecrown/work
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
Note that any example that uses a compiled module from the CosmoSIS Standard Library will fail.
