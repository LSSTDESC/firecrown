# API

The firecrown API has two parts. The first is a wrapper around CCL for building
likelihood computations for CCL-supported analyses, e.g. 3x2pt, cosmic shear,
etc. The second is a generic interface for adding terms to the log-likelihood.
The generic interface provides a simple way to combine statistically
independent analyses (i.e., one can add the log-likelihoods). The CCL API
allows users to build full log-likelihoods using a YAML configuration file.

## CCL API

The CCL API is composed of four base classes. Subclasses of these base classes
are combined to compute the full log-likelihood based on a YAML configuration
file.

The base classes are

1. `firecrown.ccl.core.Source`: Defines a source (e.g., a set of galaxies).
  This class is used to produce `pyccl.cls.Tracer` objects (actually its
  subclasses).

The classes in this file define the firecrown-CCL API.

Notes:
 - Each subclass which inherits from a given class is expected to define any
   methods defined in the mixin with the same call signature.
 - If the class nelow  includes a class-level doc string, then
   the `__init__` function of the subclass should define at least those
   arguments and/or keyword arguments.
 - Attributed ending with an underscore are set after the call to
   `apply`/`compute`/`render`.
 - Attributes define in the `__init__` method should be considered constant
   and not changed after instantiation.
 - Objects inheriting from `Systematic` should only adjust
   source/statistic properties ending with an underscore.

Statistic
Systematic
Source

LogLike

## Generic API

The generic API works with two functions. The first, `parse_config`, is
responsible for doing any initialization based on parsing on a YAML
configuration file. The second function, `compute_loglike`, does the actual
log-likelihood computations. Finally, a configuration section in the input
YAML file points firecrown to the python module with these functions.

### YAML Configuration

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

### Implementing `parse_config` and `compute_loglike`

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

It takes as an input a `pyccl.Cosmology` object, the dictionary of current
parameter values, and the data returned from `parse_config`. This function
should use the data to compute the log-likelihood. It should
also return any data to store as keys in a dictionary. Follow the following rules
when implementing this function.

1. Always use the `pyccl.Cosmology` object for any cosmological computations (e.g., distances).
2. If the likelihood is zero, this function should return `-np.inf`.
3. The elements of dictionary of returned data must be [numpy structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html#module-numpy.doc.structured_arrays).
