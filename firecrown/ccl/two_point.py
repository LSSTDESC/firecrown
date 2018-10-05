import numpy as np

import pyccl as ccl
from .parser import _parse_sources, _parse_statistics
from .sources import build_ccl_source
from ._pdfs import parse_gaussian_pdf, compute_gaussian_pdf


def parse_config(analysis):
    """Parse a nx2pt analysis.

    Parameters
    ----------
    analysis : dict
        Dictionary containg the Nx2pt analysis.

    Returns
    -------
    data : dict
        Dictionary holding all of the data needed for a Nx2pt analysis.
    """
    new_keys = {}
    new_keys['statistics'] = _parse_statistics(analysis['statistics'])
    new_keys['sources'] = _parse_sources(analysis['sources'])
    new_keys['likelihood'] = parse_gaussian_pdf(**analysis['likelihood'])

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

    sources = {}
    for name, keys in data['sources'].items():
        sources[name] = build_ccl_source(
            cosmo=cosmo,
            parameters=parameters,
            **keys)

    stats = {}
    for stat, keys in data['statistics'].items():
        _srcs = [sources[k][0] for k in keys['sources']]
        scale = np.prod([sources[k][1] for k in keys['sources']])
        if keys['kind'] == 'cl':
            stats[stat] = ccl.angular_cl(cosmo, *_srcs, keys['l']) * scale
        else:
            raise ValueError(
                "Statistic kind '%s' for statistic '%s' for the 'two_point' "
                "analysis is not recognized!" % (keys['kind'], stat))

    # build the data vector
    dv = []
    for stat in data['likelihood']['data_vector']:
        if data['statistics'][stat]['kind'] == 'cl':
            dv.append(data['statistics'][stat]['cl'] - stats[stat])
        else:
            raise ValueError(
                "Statistic '%s' not computed!" % (stat))
    dv = np.concatenate(dv, axis=0)

    # compute the log-like
    if data['likelihood']['kind'] == 'gaussian':
        loglike = compute_gaussian_pdf(dv, data['likelihood']['L'])
    else:
        raise ValueError(
            "Likelihood '%s' not recognized for source "
            "'two_point'!" % data['likelihood']['kind'])

    return loglike, stats
