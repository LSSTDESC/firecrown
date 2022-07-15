# firecrown: the "c" is for "cosmology"

## Introduction
 
Firecrown is a Python package that provides the DESC *framework* to implement likelihoods,
as well as specific likelihood implementations. Firecrown is intended to be usable *from*
external statistical analysis tools.

Currently, it supports both Cobaya and CosmoSIS, providing the necessary classes or modules
to allow users of Cobaya or CosmoSIS to call any Firecrown likelihood from within those
samplers.

* [Cobaya](https://github.com/CobayaSampler/cobaya)
* [CosmoSIS](https://github.com/joezuntz/cosmosis)

It can also be used as a library in other contexts, and so the installation of
Firecrown does not *require* the installation of a sampler.

## Installation Quickstart

> Warning
>
> The conda packaging of Firecrown is not yet completed.
> The following instructions show how we intend the installation to work;
> however, for now please follow the "Installation of dependencies for development"
> instructions below.

The easiest way to get started is by using conda. We recommend creating a conda
environment for your use.

This will install Firecrown as well as the samplers that are currently supported.

```bash
conda create --name fc -c conda-forge firecrown
```

## Installation of dependencies for development

As with the quickstart installation, you need to choose how you want to use the
Firecrown code you will be working on. Simultaneous development of either Cobaya
or CosmoSIS and Firecrown is beyond the scope of these instructions.

### Firecrown alone

```bash
conda create --name for_fc -c conda-forge sacc pyccl fitsio flake8 pylint black pytest coverage
```

### Firecrown with CosmoSIS

Firecrown supports CosmoSIS 2.x.
The conda installation of CosmoSIS does not include the CosmoSIS Standard Library (CSL), but almost all use of CosmoSIS will include the use of parts of the CSL.
These instructions include the instructions for building the CSL.

```bash
conda create --name for_fc_cosmosis -c conda-forge cosmosis cosmosis-build-standard-library sacc pyccl fitsio flake8 pylint black pytest coverage
# Note that the following will clone the CSL repository and build it in your current working directory.
# This should be done *outside* of the directory tree managed by conda, and *outside* of the `firecrown` directory.
conda activate for_fc_cosmosis
source ${CONDA_PREFIX}/bin/cosmosis-configure
cosmosis-build-standard-library
export CSL_DIR=${PWD}/cosmosis-standard-library
```

### Firecrown with Cobaya

```bash
conda create --name for_fc_cobaya -c conda-forge sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black pytest coverage
conda activate for_fc_cobaya
# Not all cobaya dependencies can be installed with conda.
python -m pip install cobaya
```

### Firecrown with both CosmoSIS and Cobaya

```bash
conda create --name for_fc_both -c conda-forge cosmosis cosmosis-build-standard-library sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black pytest coverage
conda activate for_fc_both
python -m pip install cobaya
# Note that the following will clone the CSL repository and build it in your current working directory.
# This should be done *outside* of the directory tree managed by conda, and *outside* of the `firecrown` directory.
source ${CONDA_PREFIX}/bin/cosmosis-configure
cosmosis-build-standard-library
export CSL_DIR=${PWD}/cosmosis-standard-library
```

## Getting Firecrown for development

To install the package in developer mode, start by cloning the git repo.
Activate whichever conda environment you created for your development effort.

1. Define `CSL_DIR` appropriately if you are going to use CosmoSIS.
2. Define `FIRECROWN_DIR` to be the directory into which you have cloned the `firecrown` repository.

If you do not have `PYTHONPATH` defined: define `PYTHONPATH=${FIRECROWN_DIR}/build/lib`

If you have `PYTHONPATH` defined: define `PYTHONPATH=${FIRECROWN_DIR}/build/lib:${PYTHONPATH}`

In the active environment, you can build Firecrown by:

```bash
cd ${FIRECROWN_DIR}
python setup.py build
```

The tests can be run with `pytest`, after building:

```bash
pytest
```

Some tests are marked as *slow*; those are skipped unless they are requested
using `--runslow`:

```bash
pytest --runslow
```

## Usage

The documentation for Firecrown is still under development.
It is [available on readthedocs](https://firecrown.readthedocs.io/).

> Warning
>
> The documentation currently online is out-of-date. We are working on updating it.
> The source for the documentation is in this repository, under the `docs/` subdirectory.
> If you have the repository cloned, and one of the conda environments described below active,
> the following steps will build the documentation into `docs/_build/html`:
>
>     # Installation of the required Python packages to build the documentation only needs to
>     # be done once per environment. Make sure you are in an active environment before doing
>     # this installation.
>     conda install -c conda-forge sphinx sphinx-autodoc-typehints sphinx_rtd_theme
>
>     cd docs/
>     make html

There are examples in the [examples folder](https://github.com/LSSTDESC/firecrown/examples)
that show some use of Firecrown with both Cobaya and CosmoSIS.

## Contributing

Contributions to Firecrown are welcome.

For any contribution, please start by [opening an issue](https://github.com/LSSTDESC/firecrown/issues),
and using the GitHub interface to create a branch for that issue.

To create a branch you will need write access; if you don't have write access, please send a request to the @LSSTDESC/firecrown-devs team.
You can also fork the repository and send a pull request from your fork.

When you have completed the task, push your commits to the branch you created for the issue and create a pull request.

We are using several tools to help keep the code tidy and correct; these are described below.

### Type checking

We are using type-hinting in (most of) the code, to help ensure correct use of the framework.
We are using `mypy` to verify the code is conforming to these type hints.
Please run:

```bash
mypy firecrown/ --ignore-missing-imports 
```

and fix any errors reported before pushing commits to the GitHub repository.

### Testing

> Warning
>
> We are working on improving the coverage of testing for Firecrown; it is currently very inadequate.
> As the coverage improves, we will provide instructions for writing tests for new code.

We are using `pytest` to run tests on Firecrown.
Before running tests, one must first build the code.
In addition, the environment variable `PYTHONPATH` must be correctly set to run the tests.
Please see the instructions, above, for this setup.

### Use of `pylint`

We are using `pylint` to check for a variety of possible problems.
Firecrown is not currently "clean" of all `pylint` issues, so we are not yet using `pylint` in the CI testing.

We are actively working on getting a "clean" report from `pylint`.
When this is achieved, we plan to activate `pylint` in the CI checking.
This will require that all new code pass `pylint`'s checks.

### Code formatting

We are using the command-line tool `black` to auto-format Python code in Firecrown.
Please make sure to run black on your code before creating any commits.

## Contact

If you have comments, questions, or feedback, please [open an issue](https://github.com/LSSTDESC/firecrown/issues).

You can also discuss Firecrown on the [#desc-firecrown](https://lsstc.slack.com/app_redirect?channel=desc-firecrown) LSSTC Slack channel.

## License and Conditions of Use

This software was developed within the LSSTDESC using LSST DESC resources, and
so meets the criteria given in, and is bound by, the LSST DESC Publication
Policy for being a “DESC product”. We welcome requests to access the code for
non-DESC use; if you wish to use the code outside DESC please contact the
developers.

The firecrown package is still under development and should be considered work
in progress. If you make use of any of the ideas or software in this package in
your own research, please cite them as "(LSST DESC, in preparation)" and provide
a link to this repository: https://github.com/LSSTDESC/firecrown. If you have
comments, questions, or feedback, please
[make an issue](https://github.com/LSSTDESC/firecrown/issues).

firecrown calls the CCL library: https://github.com/LSSTDESC/CCL, which makes
use of `CLASS`. For free use of the `CLASS` library, the `CLASS` developers
require that the `CLASS` paper be cited:

    CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram,
    arXiv:1104.2933, JCAP 1107 (2011) 034.

The `CLASS` repository can be found in http://class-code.net. CCL also uses code
from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package. We have
obtained permission from the FFTLog author to include modified versions of his
source code.
