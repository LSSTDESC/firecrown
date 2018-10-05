import numpy as np
import pandas as pd
import scipy.linalg


def parse_gaussian_pdf(
        *,
        kind,
        data,
        data_vector):
    """Parse a Gaussian likelihood config.

    Parameters
    ----------
    kind : str
        The kind of likelihood. (Always Gaussian!)
    data : str
        The path to the covariance matrix in csv format.
    data_vector : list of str
        The list of statistics that form the data vector.

    Returns
    -------
    parsed : dict
        A dictionary of the parsed data. Includes the input keys plus
            'cov': the covariance matrix as a numpy array
            'L': the Cholesky decomposition of the covariance matrix
    """
    new_keys = {'kind': kind, 'data_vector': data_vector, 'data': data}
    df = pd.read_csv(data)
    dim = max(np.max(df['i']), np.max(df['j'])) + 1
    cov = np.zeros((dim, dim))
    cov[df['i'].values, df['j'].values] = df['cov'].values
    new_keys['cov'] = cov
    new_keys['L'] = scipy.linalg.cholesky(cov, lower=True)
    return new_keys


def compute_gaussian_pdf(dv, L):
    """Compute a Gaussian PDF.

    Parameters
    ----------
    dv : array-like, shape (n,)
        The difference between the point and the mean of the Gaussian.
    L : array-like, shape (n, n)
        The Cholesky decomposition of the covariance matrix.

    Returns
    -------
    loglike : float
        The negative loglike excluding the covariance terms.
    """
    x = scipy.linalg.solve_triangular(L, dv, lower=True)
    return -0.5 * np.dot(x, x)
