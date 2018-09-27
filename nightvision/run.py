from .likelihoods import compute_two_point


def compute_loglike(*, cosmo, data):
    """Compute the log-likelihood.

    Parameters
    ----------
    cosmo : a `pyccl.Cosmology` object
        A cosmology.
    data : dict
        The result of calling `nightvision.config.parse` on an input YAML
        config.

    Returns
    -------
    loglike : float
        The loglikelihood of the data.
    statistics : dict
        A dictionary of output statistics from each analysis.
    """
    # first build the sources
    srcs = {}
    for src, keys in data['sources'].items():
        srcs[src] = keys['build_func'](
            cosmo=cosmo,
            params=data['parameters'],
            src_name=src,
            **keys)

    # now accumulate stats and loglike
    loglike = 0.0
    statistics = {}
    analyses = list(
        set(list(data.keys())) -
        set(['sources', 'parameters', 'run_metadata']))
    for analysis in analyses:
        if analysis == 'two_point':
            _ll, _stats = compute_two_point(
                cosmo=cosmo,
                parameters=data['parameters'],
                sources=srcs,
                **data[analysis])
        else:
            raise ValueError("Analysis '%s' not recognized!" % analysis)

        loglike += _ll
        statistics[analysis] = _stats

    return loglike, statistics
