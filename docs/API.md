# API

The `firecrown` API has two parts. The first is a wrapper around CCL for building
likelihood computations for CCL-supported analyses, e.g. 3x2pt, cosmic shear,
etc. The second is a generic interface for adding terms to the log-likelihood.
The generic interface provides a simple way to combine statistically
independent analyses (i.e., one can add the log-likelihoods). The CCL API
allows users to build full log-likelihoods using a YAML configuration file.
Finally, a global configuration block holds parameters and other metadata.

## Global YAML Configuration

A `firecrown` configuration file has a single parameters block plus one or more
blocks encoding the components of the log-likelihood.

```YAML
parameters:
  Omega_k: 0.0
  Omega_c: 0.27
  Omega_b: 0.045
  h: 0.67
  n_s: 0.96
  sigma8: 0.8
  w0: -1.0
  wa: 0.0

  # lens bin zero
  src0_delta_z: 0.0
  src1_delta_z: 0.0

two_point:
  module: firecrown.ccl.two_point
  ...
```

## CCL API

The CCL API is composed of four base classes. Subclasses of these base classes
are combined to compute the full log-likelihood based on a YAML configuration
file.

### Base Classes

These are:

1. `firecrown.ccl.core.Source`: Defines a source (e.g., a set of galaxies).
  This class is used to produce `pyccl.cls.Tracer` objects (actually its
  subclasses).
2. `firecrown.ccl.core.Statistic`: Defines a statistic (e.g., a 2pt function).
  This class is used to combine sources into statistics.
3. `firecrown.ccl.core.Systematic`: Defines a systematic effect to apply to
  either a `Source` or `Statistic`. This class is used to apply systematics
  that are associated either with a single source or a single statistic.
4. `firecrown.ccl.core.LogLike`: Defines various log-likelihood computations.
  This class is used to combine `Statistic`s together with a covariance matrix
  into a final likelihood computation.

Please see [firecrown/ccl/core.py](https://github.com/LSSTDESC/firecrown/firecrown/ccl/core.py) for the details of
each class.

Some Notes:

 - Each subclass which inherits from a given class is expected to define any
   methods defined in the parent with the same call signature. See the base
   class docstrings for additional instructions.
 - If a base class includes a class-level doc string, then
   the `__init__` function of the subclass should define at least those
   arguments and/or keyword arguments in the class-level doc string.
 - Attributes ending with an underscore are set after the call to
   `apply`/`compute`/`render`.
 - Attributes define in the `__init__` method should be considered constant
   and not changed after instantiation.
 - Objects inheriting from `Systematic` should only adjust source/statistic
   properties ending with an underscore.
 - The `read` methods are called after all objects are made and are used to
   read any additional data.   

### YAML Configuration

The example configuration file, [cosmicshear.yaml](https://github.com/LSSTDESC/firecrown/examples/cosmicshear.yaml),
shows how one would configure a cosmic shear analysis in Fourier space.

In general, a two-point YAML configuration file has four sections, `sources`,
`systematics`, `likelihood`, and `statistics`. The `sources`, `systematics`,
and `statistics` sections contain mappings of names to configuration
specifications for each item. Other sections of the file should refer to
these items by their names. The configuration of each item follows their
docstrings. There is a final optional key, `sacc_data`, which should contain
the path to the SACC data file if desired.

## Generic API

The generic API works with three required functions. The first, `parse_config`, is
responsible for doing any initialization based on parsing on a YAML
configuration file. The second function, `compute_loglike`, does the actual
log-likelihood computations. A configuration section in the input
YAML file points `firecrown` to the python module with these functions. Finally,
you should define a third function `write_stats`, that writes any returned data
to a specified path.

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

### Implementing `parse_config`, `compute_loglike`, and `write_stats`

The `parse_config` function should have the following signature

```python
def parse_config(analysis):
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
    measured : array-like, shape (n,)
        The measure statistics for this log-likelihood.
    predicted : array-like, shape (n,)
        The predicted statistics for this log-likelihood.
    covmat : array-like, shape (n, n)
        The covariance matrix for the measured statistics.
    inv_covmat : array-like, shape (n, n)
        The inverse of the covariance matrix for the measured statistics.
    stats : object or other data
        Any data you wish to store.
    """
    ...
```

It takes as an input a `pyccl.Cosmology` object, the dictionary of current
parameter values, and the data returned from `parse_config`. This function
should use the data to compute the log-likelihood. It shoudl then return
this log-likelihood along with the measured statistics, predicted statistics,
the covariance matrix of the measured statistics, and its inverse. It should
also return any data to store as keys in a dictionary. Follow the following rules
when implementing this function.

1. Always use the `pyccl.Cosmology` object for any cosmological computations (e.g., distances).
2. If the likelihood is zero, this function should return `-np.inf`.

The `write_stats` function has the signature

```python
def write_stats(*, output_path, data, stats):
    """Write statistics to a file at `output_path`.

    Parameters
    ----------
    output_path : str
        The path to which to write the data.
    data : dict
        The output of `parse_config`.
    stats : object or other data
        Second output of `compute_loglike`.
    """
    ...
```
