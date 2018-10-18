

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
    loglike = 0.0
    statistics = {}

    analyses = list(
        set(list(data.keys())) -
        set(['parameters', 'cosmosis']))
    for analysis in analyses:
        _ll, _stats = data[analysis]['eval'](
            cosmo=cosmo,
            parameters=data['parameters'],
            data=data[analysis]['data'])
        loglike += _ll
        statistics[analysis] = _stats

    return loglike, statistics
