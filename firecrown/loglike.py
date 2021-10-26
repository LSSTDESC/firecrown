from __future__ import annotations
from typing import Dict, Set
from pyccl.core import CosmologyCalculator
from .parser_constants import FIRECROWN_RESERVED_NAMES


def compute_loglike(*, cosmo: CosmologyCalculator, data: Dict[str, Dict]):
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
    analyses: Set[str] = set(data.keys()) - set(FIRECROWN_RESERVED_NAMES)
    parameters: Dict[str, float] = data["parameters"]
    for analysis in analyses:
        # It appears that current_analysis will always have keys "data", "eval" and "write".
        # Does it really never lack these keys?
        # Will it ever have others keys?
        current_analysis = data[analysis]
        assert set(current_analysis.keys()) == {"data", "eval", "write"}
        current_eval = current_analysis["eval"]
        current_data = current_analysis["data"]
        _ll, _meas, _pred, _cov, _inv_cov, _stats = current_eval(
            cosmo=cosmo, parameters=parameters, data=current_data
        )
        loglike[analysis] = _ll
        statistics[analysis] = _stats
        meas[analysis] = _meas
        pred[analysis] = _pred
        cov[analysis] = _cov
        inv_cov[analysis] = _inv_cov

    return loglike, meas, pred, cov, inv_cov, statistics
