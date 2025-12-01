"""Utility functions for data_functions module."""

import hashlib

import sacc


def cov_hash(sacc_data: sacc.Sacc) -> str:
    """Return a hash of the covariance matrix.

    :param sacc_data: The SACC data object containing the covariance matrix.
    :return: The hash of the covariance matrix.
    """
    assert sacc_data.covariance is not None
    return hashlib.sha256(sacc_data.covariance.dense).hexdigest()
