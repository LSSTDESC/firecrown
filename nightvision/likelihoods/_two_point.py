import numpy as np
import pandas as pd

import pyccl as ccl
from ._pdfs import parse_gauss_pdf

__all__ = ['parse_two_point', 'compute_two_point']


def parse_two_point(*, likelihood, statistics):
    """Parse a two-point likelihood computation from the config.

    Parameters
    ----------
    likelihood : dict
        Dictionary describing the form of the likelihood. Must have at least
        the key 'kind'. See `nightvision.likelihoods._pdfs` for details.
    statistics : dict
        Dictionary describing the 2p statistics. This dict expressed in YAML
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
        The parsed two-point likelihood information.
    """
    new_keys = {}
    if likelihood['kind'] == 'gaussian':
        new_keys['likelihood'] = parse_gauss_pdf(likelihood)
    else:
        raise ValueError(
            "Likelihood '%s' not recognized for source "
            "'two_point'!" % likelihood['kind'])

    stats = {}
    for stat, keys in statistics.items():
        stats[stat] = _parse_two_point_statistic(keys)
    new_keys['statistics'] = stats

    return new_keys


def _parse_two_point_statistic(keys):
    new_keys = {}
    new_keys.update(keys)
    df = pd.read_csv(keys['data'])
    if keys['kind'] == 'cl':
        new_keys['l'] = df['l'].values.copy()
        new_keys['cl'] = df['cl'].values.copy()
    elif keys['kind'] in ['gg', 'gl', 'l+', 'l+']:
        new_keys['t'] = df['t'].values.copy()
        new_keys['xi'] = df['xi'].values.copy()

    return new_keys


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
    TODO: Write shit here.

    Returns
    -------
    loglike : float
        The computed log-likelihood.
    stats : dict
        Dictionary with 2pt stat predictions.
    """
    srcs = {}
    for src, keys in sources.items():
        srcs[src] = keys['build_func'](
            cosmo=cosmo,
            params=parameters,
            src_name=src,
            **keys
        )

    stats = {}
    for stat, keys in statistics.items():
        _srcs = [srcs[k][0] for k in keys['sources']]
        scale = np.prod([srcs[k][1] for k in keys['sources']])
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
    loglike = likelihood['comp'](dv)

    return loglike, stats
