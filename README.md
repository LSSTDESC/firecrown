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

## API

The firecrown API has two parts. The first is a wrapper around CCL for building
likelihood computations for CCL-supported analyses, e.g. 3x2pt, cosmic shear,
etc. The second is a generic interface for adding terms to the log-likelihood.

### CCL Wrapper



### Generic Interface

The generic API works with two functions. The first, `parse_config`, is
responsible for doing any initialization based on parsing on a YAML
configuration file. The second function, `compute_loglike`, does the actual
log-likelihood computations. Finally, a configuration section in the input
YAML file points firecrown to the python module with these functions.

#### YAML Configuration

The YAML configuration blob for your function should look like

```YAML
my_loglike_term:
  module: mymodule.import.path.to.functions
  ...
```

Any valid YAML is allowed after the `module` key. You can refer to parameters
in the parameters section by their names, e.g.

```YAML
parameters:
  a: 5

my_loglike_term:
  module: mymodule.import.path.to.functions
  input_for_a: a
  ...
```

#### Implementing `parse_config` and `compute_loglike`

The `parse_config` function should have the following signature

```python
def parse_config((analysis):
    """Parse an analysis.

    Parameters
    ----------
    analysis : dict
        Dictionary containing the parsed YAML.

    Returns
    -------
    data
    """
    ...
```

It is imported from the module path given above, e.g.
`mymodule.import.path.to.functions`. It gets as an input the parsed YAML blob
for your analysis (everything under `my_loglike_term` above). It is expected
to return one object (`data` above). This object is passed back to the
`compute_loglike` function.

The `compute_loglike` function has the signature

```python
def compute_loglike(
        *,
        cosmo,
        parameters,
        data):
    """Compute the log-likelihood of the data.

    Parameters
    ----------
    cosmo : a `pyccl.Cosmology` object
        A cosmology.
    parameters : dict
        Dictionary mapping parameters to their values.
    data : dict
        The output of `parse_config`.

    Returns
    -------
    loglike : float
        The computed log-likelihood.
    stats : dict
        Dictionary with any data to store.
    """
    ...
```

It takes as an input a `pyccl.Cosmology` object, the dictionary of
current parameter values, and the data returned from `parse_config`. This
function should use the data to compute the log-likelihood. If the
likelihood is zero, this function should return `-np.inf`. It should
also return any data to store as keys in a dictionary.

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

    CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram,
    arXiv:1104.2933, JCAP 1107 (2011) 034.

The `CLASS` repository can be found in http://class-code.net. CCL also uses
code from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We
have obtained permission from the FFTLog author to include modified versions of
his source code.
