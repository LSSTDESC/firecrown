"""
A prior can be any distribution in `scipy.stats`. You specify them in YAML
as follows:

```YAML
priors:
  module: firecrown.priors
  param1:
    # here 'norm' is the name of the function/class in scipy.stats
    kind: norm
    # any keywors to this function are listd by name
    # these are passed to the `logpdf` method
    loc: 0.5
    scale: 0.5
```
"""
import copy
import scipy.stats


def parse_config(analysis):
    """Parse priors for an analysis.

    Parameters
    ----------
    analysis : dict
        Dictionary containing the parsed YAML.

    Returns
    -------
    data : dict
        The dictionary.
    """
    # we need a copy here since we are not promised that the input analysis
    # dict will not be changed
    return copy.deepcopy(analysis)


def compute_loglike(
        *,
        cosmo,
        parameters,
        data):
    """Compute the log-likelihood of the priors.

    Parameters
    ----------
    cosmo : a `pyccl.Cosmology` object
        A cosmology.
    parameters : dict
        Dictionary mapping parameters to their values.
    data : dict
        The output of `parse_config` above.

    Returns
    -------
    loglike : float
        The computed log-likelihood.
    stats : dict
        Dictionary with any data to store.
    """
    loglike = 0.0
    for param in parameters:
        if param in data and param != 'module':
            if not hasattr(scipy.stats, data[param]['kind']):
                raise ValueError("Prior dist %s not defined!" % data[param])

            dist = getattr(scipy.stats, data[param]['kind'])
            keys = {k: v for k, v in data[param].items() if k != 'kind'}
            loglike += dist.logpdf(parameters[param], **keys)
    return loglike, {}
