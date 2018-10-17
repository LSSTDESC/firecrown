# firecrown

The "c" is for "cosmology."

## Installation

You need to have CCL installed first. Try:

```bash
pip install pyccl
```

Then you can install the `master` branch via

```
pip install git+https://github.com/LSSTDESC/firecrown.git
```

## Usage

TLDR

```bash
firecrown compute <config file>
```

will run an example problem.

See the example in the examples folder for more details.

# API



## License

The firecrown package is still under development and should be considered work
in progress. If you make use of any of the ideas or software in this package
in your own research, please cite them as "(LSST DESC, in preparation)" and
provide a link to this repository: https://github.com/LSSTDESC/firecrown.
If you have comments, questions, or feedback, please
[make an issue](https://github.com/LSSTDESC/firecrown/issues).

firecrown calls the CCL library: https://github.com/LSSTDESC/CCL, which makes
use of `CLASS`. For free use of the `CLASS` library, the `CLASS` developers
require that the `CLASS` paper be cited:

    CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933, JCAP 1107 (2011) 034.

The `CLASS` repository can be found in http://class-code.net. CCL also uses
code from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We
have obtained permission from the FFTLog author to include modified versions of
his source code.
