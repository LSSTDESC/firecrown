import numpy as np
import pandas as pd

import pyccl as ccl
from ._pdfs import compute_gaussian_pdf

__all__ = ['parse_two_point', 'compute_two_point']


def parse_two_point(statistics):
    """Parse two-point statistics from the config.

    Parameters
    ----------
    statistics : dict
        Dictionary describing the 2pt statistics. This dict expressed in YAML
        should have the form:

            ```YAML
            cl_src0_src0:
              sources: ['src0', 'src0']
              kind: 'cl'
              data: ./data/cl00.csv
              systematics:
                ...
            ```

        The kind can be any of the CCL 2pt function types ('gg', 'gl', 'l+',
        'l-') or 'cl' for Fourier space statistics.

    Returns
    -------
    parsed : dict
        The parsed two-point statistics.
    """
    stats = {}
    for stat, keys in statistics.items():
        new_keys = {}
        new_keys.update(keys)
        df = pd.read_csv(keys['data'])
        if keys['kind'] == 'cl':
            new_keys['l'] = df['l'].values.copy()
            new_keys['cl'] = df['cl'].values.copy()
        elif keys['kind'] in ['gg', 'gl', 'l+', 'l-']:
            new_keys['t'] = df['t'].values.copy()
            new_keys['xi'] = df['xi'].values.copy()

        stats[stat] = new_keys

    return stats


def compute_two_point(
        *,
        cosmo,
        parameters,
        sources,
        likelihood,
        statistics):
    """Compute the log-likelihood of 2pt data.

    Parameters
    ----------
    cosmo : a `pyccl.Cosmology` object
        A cosmology.
    parameters : dict
        Dictionary mapping parameters to their values.
    sources : dict
        Dictionary mapping source names to the built sources. See for example
        `nightvision.sources.build_ccl_source`.
    likelihood : dict
        Dictionary specifying the likelihood. This is a result of
        calling `nightvision.sources.parse_two_point` on an input two-point
        YAML config.
    statistics : dict
        Dictionary specifying the statistics to compute. This is a result of
        calling `nightvision.sources.parse_two_point` on an input two-point
        YAML config.

    Returns
    -------
    loglike : float
        The computed log-likelihood.
    stats : dict
        Dictionary with 2pt stat predictions.
    """

    stats = {}
    for stat, keys in statistics.items():
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
    for stat in likelihood['data_vector']:
        dv.append(statistics[stat]['cl'] - stats[stat])
    dv = np.concatenate(dv, axis=0)

    # compute the log-like
    if likelihood['kind'] == 'gaussian':
        loglike = compute_gaussian_pdf(dv, likelihood['L'])
    else:
        raise ValueError(
            "Likelihood '%s' not recognized for source "
            "'two_point'!" % likelihood['kind'])

    return loglike, stats
