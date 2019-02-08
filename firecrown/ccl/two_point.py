import pandas as pd

from .parser import (
    _parse_sources,
    _parse_systematics,
    _parse_two_point_statistics,
    _parse_likelihood)


def parse_config(analysis):
    """Parse a nx2pt analysis.

    Parameters
    ----------
    analysis : dict
        Dictionary containing the Nx2pt analysis.

    Returns
    -------
    data : dict
        Dictionary holding all of the data needed for a Nx2pt analysis.
    """
    new_keys = {}
    new_keys['statistics'] = _parse_two_point_statistics(
        analysis['statistics'])
    new_keys['sources'] = _parse_sources(analysis['sources'])
    if 'likelihood' in analysis:
        new_keys['likelihood'] = _parse_likelihood(analysis['likelihood'])
    new_keys['systematics'] = _parse_systematics(analysis['systematics'])

    return new_keys


def compute_loglike(
        *,
        cosmo,
        parameters,
        data):
    """Compute the log-likelihood of 2pt data.

    Parameters
    ----------
    cosmo : a `pyccl.Cosmology` object
        A cosmology.
    parameters : dict
        Dictionary mapping parameters to their values.
    data : dict
        The output of `firecrown.ccl.two_point.parse_config`.

    Returns
    -------
    loglike : float
        The computed log-likelihood.
    stats : dict
        Dictionary with 2pt stat predictions.
    """

    for name, src in data['sources'].items():
        src.render(
            cosmo, parameters, systematics=data['systematics'])

    stats = {}
    _data = {}
    _theory = {}
    for name, stat in data['statistics'].items():
        stat.compute(
            cosmo, parameters, data['sources'],
            systematics=data['systematics'])
        _data[name] = stat.measured_statistic_
        _theory[name] = stat.predicted_statistic_
        stats[name] = pd.DataFrame({
            'ell_or_theta': stat.ell_or_theta_,
            'measured_statistic': _data[name],
            'predicted_statistic': _theory[name]}).to_records(index=False)

    # compute the log-like
    if 'likelihood' in data:
        loglike = data['likelihood'].compute(_data, _theory)
    else:
        loglike = None

    return loglike, stats
