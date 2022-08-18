=================================================
Installation for non-development use of Firecrown
=================================================


.. role:: bash(code)
   :language: bash

.. warning::

    These instructions do not work for Macs with M1 processors.
    For installation on that platform, please use the :doc:`Apple M1 installation instructions<apple_m1_instructions>`.

Using Firecrown in non-development mode does not require cloning the repository.
Instead, you can use a conda environment that contains Firecrown and its dependencies.
Most (but not all) of the dependencies of Firecrown are available through conda.

These instructions include details on how to obtain the samplers.

You only need to create the conda environment once.
Every time you want to do development in a new shell session you will need to activate the conda environment and set the environment variables.


Creation of the conda environment
=================================

These instructions will create a new conda environment containing all the packages used to support Firecrown, and Firecrown itself.

It is best to execute these commands in a new directory established for your work with Firecrown.
The `cosmosis-build-standard-library` command below will clone and then build the CosmoSIS Standard Library.
This will create a directory `cosmosis-standard-library` in whatever is your current directory when you execute the command.

.. code:: bash

    conda create --name firecrown_user -c conda-forge cosmosis cosmosis-build-standard-library sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib
    conda activate firecrown_user
    python -m pip install cobaya
    source ${CONDA_PREFIX}/bin/cosmosis-configure
    cosmosis-build-standard-library
    

Setting your environment for development
========================================

Each time you want to do development in a new shell session you need to activate the conda environment and set some environment variables.

Begin by `cd`-ing to the working directory you used above (which should contain a `cosmosis-standard-library` directory).

.. code:: bash

    cd /path/to/directory/used above
    conda activate firecrown_user
    export CSL_DIR=${PWD}/cosmosis-standard-library
    export FIRECROWN_SITE_PACKAGES=${COSMOSIS_SRC_DIR}/..

See the :doc:`example of non-developer mode usage <non-developer-mode-example/README>` for an example likelihood script and its use with both Cobaya and CosmoSIS.

