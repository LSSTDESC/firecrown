import numpy as np
import pandas as pd
import scipy.linalg

from ..core import LogLikeMixin


class TdistLogLike(LogLikeMixin):
    """A T-distribution for the log-likelihood.

    This distribution is appropriate when the covariance has been obtained
    from a finite number of simulations. See Sellentin & Heavens
    (2016; arXiv:1511.05969). As the number of simulations increases, the
    T-distribution approaches a Gaussian.

    Parameters
    ----------
    data : str
        The path to the covariance matrix in CSV format. The columns should be
        {'i', 'j', 'cov'} giving the indices of each matrix element and its
        value.
    data_vector : list of str
        A list of the statistics in the config file in the order they appear in
        the covariance matrix.
    nu: int
        The shape parameter. Set to the number of simulations.

    Attributes
    ----------
    cov : np.ndarray, shape (n, n)
        The covariance matrix as a numpy array.
    cholesky : np.ndarray, shape (n, n)
        The (lower triangular) Cholesky decomposition of the covariance matrix.

    Methods
    -------
    compute_loglike : compute the log-likelihood
    """
    def __init__(self, data, data_vector, nu):
        self.data = data
        self.data_vector = data_vector
        self.nu = nu

        df = pd.read_csv(data)
        dim = max(np.max(df['i']), np.max(df['j'])) + 1
        cov = np.zeros((dim, dim))
        cov[df['i'].values, df['j'].values] = df['cov'].values
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(cov, lower=True)

    def compute_loglike(self, data, theory, **kwargs):
        """Compute the log-likelihood.

        Parameters
        ----------
        data : dict of arrays
            A dictionary mapping the names of the statistics to their
            values in the data.
        theory : dict of arrays
            A dictionary mapping the names of the statistics to their
            predictions.
        **kwargs : extra keyword arguments
            Any extra keyword arguments are ignored.

        Returns
        -------
        loglike : float
            The log-likelihood.
        """
        dv = []
        for stat in self.data_vector:
            dv.append(np.atleast_1d(data[stat] - np.atleast_1d(theory[stat])))
        dv = np.concatenate(dv, axis=0)
        x = scipy.linalg.solve_triangular(self.cholesky, dv, lower=True)
        chi2 = np.dot(x, x)
        return -0.5 * self.nu * np.log(1.0 + chi2 / (self.nu - 1.0))
