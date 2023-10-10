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
Most (but not all) of the dependencies of Firecrown are available through `conda`.
We use `pip` to install only those dependencies not available through `conda`.

These instructions include details on how to obtain the samplers used with Firecrown.
This is important because if you are doing development it is necessary to make sure what you change or add works with both of the supported samplers.

You only need to clone the Firecrown repository once, and to create the conda environment once.
Every time you want to do development in a new shell session you will need to activate the conda environment and set the environment variables.
You may need to update your conda environment periodically, to keep up with current releases of the packages on which Firecrown depends.
Instructions for doing so are found below.

The continuous integration (CI) testing system used for Firecrown will run tests, execute examples, and employ a variety of code quality verification tools on every pull request.
Failure by any one of these will cause the CI system to automatically reject the pull request.
In order to make it easier to get your pull request passed, these instructions include all the necessary software in the conda environment used for development.
Please see the end of this page for a listing of what the CI system will run, and how to run the same tests yourself.
Note that the CI system is typically using the latest (compatible) version of all the Firecrown dependencies; this is one of the reasons it is best practice to keep the dependencies of your development environment up-to-date.

Clone the Firecrown repository
==============================

Choose a directory in which to work.
In this directory, you will be cloning the Firecrown repository and later building some of the non-Firecrown code that is not installable through conda.
Note that this is *not* the directory in which the conda environment is created, nor is it the directory in which the CosmoSIS Standard Library (CSL) will be built.

.. code:: bash

    git clone https://github.com/LSSTDESC/firecrown.git
    

Installation of dependencies
============================

These instructions will create a new conda environment containing all the packages used in development.
This includes testing and code verification tools used during the development process.
We use the command `conda` in these instructions, but you may prefer instead to use `mamba`.
The Mamba version of Conda is typically faster when "solving" environments, which is done both on installation and during updates of the environment.

It is best to execute these commands while in the same directory as you were in when you cloned the Firecrown repository above.
The `cosmosis-build-standard-library` command below will clone and then build the CosmoSIS Standard Library.
We recommend doing this in the directory in which the conda environment resides.
We have found this helps to make sure that only one version of the CSL is associated with any development efforts using the associated installation of CosmoSIS.

.. code:: bash

    # conda env update, when run as suggested, is able to create a new environment, as
    # well as updating an existing environment.
    conda env update -f firecrown/environment.yml
    conda activate firecrown_developer
    # We define two environment variables that will be defined whenever you activate
    # the conda environment.
    conda env config vars set CSL_DIR=${CONDA_PREFIX}/cosmosis-standard-library FIRECROWN_DIR=${PWD}/firecrown
    # The command above does not immediately defined the environment variables.
    # They are made available on every fresh activation of the environment.
    # So we have to deactivate and then reactivate...
    conda deactivate
    conda activate firecrown_developer
    # Now we can finish building the CosmoSIS Standard Library.
    source ${CONDA_PREFIX}/bin/cosmosis-configure
	# We want to put the CSL into the same directory as conda environment upon which it depends
	pushd ${CONDA_PREFIX}
    cosmosis-build-standard-library
	popd

Setting your environment for development
========================================

Each time you want to do development in a new shell session you need to activate the conda environment and set some environment variables.

When you activate the conda environment, you can use the environment variable you defined when creating the environment to find your Firecrown directory:

.. code:: bash

    conda activate firecrown_developer
    cd ${FIRECROWN_DIR}

To "build" the Firecrown code you should be in the Firecrown directory:

.. code:: bash

    python -m pip install --no-deps --editable ${PWD}

The tests can be run with :bash:`pytest`, after building:

.. code:: bash

    python -m pytest -vv

Examples can be run by `cd`-ing into the specific examples directory and following the instructions in the local README file.
You can also consult `firecrown/.github/workflows/ci.yml`, which contains the full test of examples and tests run by the CI system.

Before committing code
======================

We are using several tools to help improve the quality of the Firecrown code.
Before committing any code, please use the following tools, and address any complaints they raise.
All of these are used as part of the CI system as part of the checking of all pull requests.

.. code:: bash
    # We are using black to keep consistent formatting across all python source files.
    black firecrown examples tests

    # We are using flake8 to help verify PEP8 compliance.
    flake8 firecrown examples tests

    # We are using pylint to enforce a variety of rules.
    # Different directories require some different rules.
    pylint firecrown
    pylint --rcfile firecrown/models/pylintrc firecrown/models
    pylint --rcfile tests/pylintrc tests

    # We are using type hints and mypy to help catch type-related errors.
    mypy -p firecrown -p examples -p tests

Keeping your conda environment up-to-date
=========================================

Why to do it.

How to do it.


