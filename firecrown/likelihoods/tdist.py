import numpy as np
import pandas as pd
import scipy.linalg


class TDistLikelihood(object):
    """A class for computing a T distribution likelihood with a constant
    covariance matrix. See Sellentin & Heavens (2016; arXiv:1511.05969).

    Parameters
    ----------
    covariance : str
        The path to the CSV file with the covariance information. The
        columns in the file must be ['i', 'j', 'cov'] giving the indices
        of the matrix elements and the value.
    n : int
        The shape parameter. Usually corresponds to the number of simulations
        used.
    data_vector : list of str
        List of the sources that make up the covariance matrix. They should be
        listed in the order in which they appear in the covariance matrix.

    Attributes
    ----------
    covariance : array-like
        The covariance matrix.
    n : int
        The shape parameter.
    data_vector : list of str
        The sources that make up the data vector for this covariance matrix.
    cholesky : array-like
        The Cholesky decomposition of the covariance matrix.

    Methods
    -------
    __call__ : compute the log-likelihood
    """

    def __init__(self, *, covariance, n, data_vector):
        self._covariance = covariance
        self.n = n
        self.data_vector = data_vector

        df = pd.read_csv(covariance)
        n = max(np.max(df['i']), np.max(df['j'])) + 1
        self.covariance = np.zeros((n, n))
        self.covariance[df['i'].values, df['j'].values] = df['cov'].values
        self.cholesky = scipy.linalg.cholesky(self.covariance, lower=True)

    def __call__(self, data, theory):
        """Compute the log-likelihood.

        Parameters
        ----------
        data : array-like
            The data vector.
        theory : array-like
            The prediction of the data vector.

        Returns
        -------
        loglike : float
            The loglikelihood.
        """
        dv = data - theory
        x = scipy.linalg.solve_triangular(self.cholesky, dv, lower=True)
        chi2 = np.dot(x, x)
        return -0.5 * self.n * np.log(1.0 + chi2 / (self.n - 1.0))
