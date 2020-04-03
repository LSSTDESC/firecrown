from .parser_constants import FIRECROWN_RESERVED_NAMES


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
    loglike : dict of floats
        The log-likelihood of the analyses.
    measured : dict of array-like, shape (n,)
      The measure statistics for all log-likelihoods.
    predicted : dict of array-like, shape (n,)
      The predicted statistics for all log-likelihoods.
    covmat : dict of array-like, shape (n, n)
      The covariance matrices for the measured statistics.
    inv_covmat : dict of array-like, shape (n, n)
      The inverse covariance matrices for the measured statistics.
    statistics : dict
        A dictionary of custom output statistics from each analysis.
    """
    loglike = {}
    statistics = {}
    meas = {}
    pred = {}
    cov = {}
    inv_cov = {}

    analyses = list(set(list(data.keys())) - set(FIRECROWN_RESERVED_NAMES))
    for analysis in analyses:
        _ll, _meas, _pred, _cov, _inv_cov, _stats = data[analysis]['eval'](
            cosmo=cosmo,
            parameters=data['parameters'],
            data=data[analysis]['data'])
        loglike[analysis] = _ll
        statistics[analysis] = _stats
        meas[analysis] = _meas
        pred[analysis] = _pred
        cov[analysis] = _cov
        inv_cov[analysis] = _inv_cov

    return loglike, meas, pred, cov, inv_cov, statistics
