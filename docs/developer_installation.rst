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

The continuous integration (CI) testing system used for Firecrown will run tests, execute examples, and employ a variety of code quality verification tools on every pull request.
Failure by any one of these will cause the CI system to automatically reject the pull request.
In order to make it easier to get your pull request passed, these instructions include all the necessary software in the conda environment used for development.
Please see the end of this page for a listing of what the CI system will run, and how to run the same tests yourself.


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

    # conda env update, when run as suggested, is able to create a new environment, as
    # well as updating an existing environment.
    conda env update -f firecrown/environment.yml
    conda activate firecrown_developer
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

To build the Firecrown code you should be in the Firecrown directory:

.. code:: bash

    cd ${FIRECROWN_DIR}
    python setup.py build

The tests can be run with :bash:`pytest`, after building:

.. code:: bash

    # We recommend removing the previous build and using the setup.py to build
    # to more closely match what will be done when creating a new release.
    rm -r build/
    python setup.py build
    python -m pytest -vv

Examples can be run by `cd`-ing into the specific examples directory and following the instructions in the local README file.
You can also consult `firecrown/.github/workflows/ci.yml`, which contains the full test of examples and tests run by the CI system.

Before committing code
======================

We are using several tools to help improve the quality of the Firecrown code.
Before committing any code, please use the following tools, and address any complaints they raise.
All of these are used as part of the CI system as part of the checking of all pull requests.

.. code:: bash

    # We are using type hints and mypy to help catch type-related errors.
    mypy -p firecrown -p examples -p tests

    # We are using flake8 to help verify PEP8 compliance.
    flake8 firecrown examples tests

    # We are using pylint to enforce a variety of rules.
    # Not all of the code is "clean" according to pylint; this is a work in progress
    pylint --rcfile tests/pylintrc --recursive=y tests

    # We are using black to keep consistent formatting across all python source files.
    black --check firecrown/ examples/ tests/

    # Note that this use of black does not actually change any file. If files other than
    # those you edited are complained about by black, please file an issue.

