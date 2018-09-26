import numpy as np
import scipy.linalg

import pyccl as ccl


def two_point(
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
        _srcs = [srcs[k] for k in keys['sources']]
        if keys['kind'] == 'cl':
            stats[stat] = ccl.angular_cl(cosmo, *_srcs, keys['l'])
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
    x = scipy.linalg.solve_triangular(likelihood['L'], dv)
    loglike = -0.5 * np.dot(x, x)

    return loglike, stats
