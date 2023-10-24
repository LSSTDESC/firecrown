=================================================
Installation for non-development use of Firecrown
=================================================


.. role:: bash(code)
   :language: bash

Using Firecrown in non-development mode does *not* require cloning the repository.
Instead, you can use a conda environment that contains Firecrown and its dependencies.
Most (but not all) of the dependencies of Firecrown are available through `conda`.
Some of the dependencies are available only through `pip`.
The CosmoSIS Standard Library (CSL) is not available through `conda`; instead, a Conda package that allows you to build your own copy of the CSL is available.

You only need to create the Conda environment once.
Every time you want to do development in a new shell session you will need to activate the Conda environment.


Creation of the conda environment
=================================

These instructions will create a new conda environment containing all the packages used to support Firecrown, and Firecrown itself.
We use the command `conda` in these instructions, but you may prefer instead to use `mamba`.
The Mamba version of Conda is typically faster when *solving* environments, which is done both on installation and during updates of the environment.

It is best to execute these commands in a new directory established for your work with Firecrown.
Any sampler configuration files and any new likelihood factory functions you write will go into this directory.
None of the Firecrown code, nor the code for any of its dependencies, will be in this directory.
The `cosmosis-build-standard-library` command below will clone and then build the CosmoSIS Standard Library.
We recommend doing this in the directory in which the conda environment resides.
We have found this helps to make sure that only one version of the CSL is associated with any development efforts using the associated installation of CosmoSIS.
It also makes it easier to keep all of the products in the conda environment consistent when updating is needed.

.. code:: bash

    conda create --name firecrown_user -c conda-forge firecrown
    conda activate firecrown_user
    conda env config vars set CSL_DIR=${CONDA_PREFIX}/cosmosis-standard-library
    conda deactivate
    conda activate firecrown_user
    cd ${CONDA_PREFIX}
    source ${CONDA_PREFIX}/bin/cosmosis-configure
    cosmosis-build-standard-library main

Setting your environment for work
=================================

Each time you want to do development in a new shell session you need to activate the conda environment.

.. code:: bash

    conda activate firecrown_user

See the :doc:`example of non-developer mode usage <non-developer-mode-example/README>` for an example likelihood script and its use with the samplers.
