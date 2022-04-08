# firecrown: the "c" is for "cosmology"

[![CircleCI](https://circleci.com/gh/LSSTDESC/firecrown/tree/master.svg?style=svg)](https://circleci.com/gh/LSSTDESC/firecrown/tree/master)

## Introduction

Firecrown is most often used along with a *sampler*. There are two samplers
currently supported:

* [Cobaya](https://github.com/CobayaSampler/cobaya)
* [CosmoSIS](https://bitbucket.org/joezuntz/cosmosis)

It can also be used as a library in other contexts, and so installation of
Firecrown does not *require* installation of a sampler.

## Installation Quickstart

> Warning
>
> The conda packaging of Firecrown is not yet done.
> The following instructions show how we intend the installation to work;
> however, for now please follow the "Installation of dependencies for development"
> instructions below.

The easiest way to get started is by using conda. We recommend creating a conda
environment for your use.

There are different prescriptions, depending upon the choice you make on the use
of samplers. You can use whatever environment name you prefer for any
installation.

For Firecrown alone:

```bash
conda create --name fc -c conda-forge firecrown
```

For Firecrown with CosmoSIS:

```bash
conda create --name fc_cosmosis -c conda-forge cosmosis firecrown
```

For Firecrown with Cobaya (note Cobaya is not currently available from
conda-forge):

```bash
conda create --name fc_cobaya -c conda-forge firecrown
conda activate fc_cobaya
python -m pip install cobaya
```

For Firecrown with both Cobaya and CosmoSIS:

```bash
conda create --name fc_both -c conda-forge cosmosis firecrown
conda activate fc_both
python -m pip install cobaya
```

## Installation of dependencies for development

As with the quickstart installation, you need to choose how you want to use the
Firecrown code you will be working on. Simultaneous development of either Cobaya
or CosmoSIS and Firecrown is beyond the scope of these instructions.

### Firecrown alone

```bash
conda create --name for_fc -c conda-forge sacc pyccl fitsio flake8 pylint black
```

### Firecrown with CosmoSIS

```bash
conda create --name for_fc_cosmosis -c conda-forge cosmosis sacc pyccl fitsio flake8 pylint black
```

### Firecrown with Cobaya

```bash
conda create --name for_fc_cobaya -c conda-forge sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black
conda activate for_fc_cobaya
python -m pip install cobaya
```

### Firecrown with both CosmoSIS and Cobaya

```bash
conda create --name for_fc_both -c conda-forge cosmosis sacc pyccl fitsio fuzzywuzzy urllib3 PyYAML portalocker idna dill charset-normalizer requests matplotlib flake8 pylint black
conda activate for_fc_both
python -m pip install cobaya
```

## Getting Firecrown for development

To install the package in developer mode, start by cloning the git repo.
Activate which ever conda environment you created for your development effort.

In the active environment, you can build Firecrown by:

```bash
python setup.py build
```

The tests can be run with `pytest`, after building:

```bash
python setup.py build
PYTHONPATH=./build/lib pytest
```

Some tests are marked as *slow*; those are skipped unless they are requested
using `--runslow`:

```bash
PYTHONPATH=./build/lib pytest --runslow
```

## Usage

*Updates still to come*.

See the examples in
the [examples folder](https://github.com/LSSTDESC/firecrown/examples)
for more details.

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
