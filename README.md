# firecrown: the "c" is for "cosmology"

[![CircleCI](https://circleci.com/gh/LSSTDESC/firecrown/tree/master.svg?style=svg)](https://circleci.com/gh/LSSTDESC/firecrown/tree/master)

## Installation

You need to have CCL installed first. Try:

```bash
pip install pyccl
```

Then you can download and install the `master` branch via

```
pip install git+https://github.com/LSSTDESC/firecrown.git
```

If you have already cloned the repo, simply install via

```
pip install -e .
```

## Usage

TLDR

```bash
firecrown compute <config file>
```

will run an example problem.

See the example in the examples folder for more details.

## Sampling

To use CosmoSIS to sample cosmological parameters, first install cosmosis-standalone.
You will need to specify compilers on install, for example:

```bash
CC=gcc-7 CXX=g++-7 FC=gfortran pip install cosmosis-standalone
```

Currently the Clang compiler cannot compile cosmosis.

You can then run with:

```bash
cd examples
firecrown run-cosmosis cosmicshear.yaml
```

## API

See the [API documentation](API.md) for details.

## License and Conditions of Use

This software was developed within the LSSTDESC using LSST DESC resources, and 
so meets the criteria given in, and is bound by, the LSST DESC Publication Policy 
for being a “DESC product”. We welcome requests to access the code for non-DESC use; 
if you wish to use the code outside DESC please contact the developers.

The firecrown package is still under development and should be considered work
in progress. If you make use of any of the ideas or software in this package
in your own research, please cite them as "(LSST DESC, in preparation)" and
provide a link to this repository: https://github.com/LSSTDESC/firecrown.
If you have comments, questions, or feedback, please
[make an issue](https://github.com/LSSTDESC/firecrown/issues).

firecrown calls the CCL library: https://github.com/LSSTDESC/CCL, which makes
use of `CLASS`. For free use of the `CLASS` library, the `CLASS` developers
require that the `CLASS` paper be cited:

    CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram,
    arXiv:1104.2933, JCAP 1107 (2011) 034.

The `CLASS` repository can be found in http://class-code.net. CCL also uses
code from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We
have obtained permission from the FFTLog author to include modified versions of
his source code.
