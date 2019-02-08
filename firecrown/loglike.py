

def compute_loglike(*, cosmo, data):
    """Compute the log-likelihood.

    Parameters
    ----------
    cosmo : a `pyccl.Cosmology` object
        A cosmology.
    data : dict
        The result of calling `firecrown.config.parse` on an input YAML
        config.

    Returns
    -------
    loglike : float
        The log-likelihood of the data.
    statistics : dict
        A dictionary of output statistics from each analysis.
    """
    loglike = None
    statistics = {}

    analyses = list(
        set(list(data.keys())) -
        set(['parameters', 'cosmosis']))
    for analysis in analyses:
        _ll, _stats = data[analysis]['eval'](
            cosmo=cosmo,
            parameters=data['parameters'],
            data=data[analysis]['data'])
        if _ll is not None:
            if loglike is None:
                loglike = 0.0
            loglike += _ll
        statistics[analysis] = _stats

    return loglike, statistics
