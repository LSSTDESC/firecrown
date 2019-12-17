import numpy as np
import scipy.linalg
import sacc

from ..core import LogLike


class ConstGaussianLogLike(LogLike):
    """A Gaussian log-likelihood with a constant covariance matrix.

    Parameters
    ----------
    data_vector : list of str
        A list of the statistics in the config file in the order you want them
        to appear in the covariance matrix.

    Attributes
    ----------
    cov : np.ndarray, shape (n, n)
        The covariance matrix.
    cholesky : np.ndarray, shape (n, n)
        The (lower triangular) Cholesky decomposition of the covariance matrix.

    Methods
    -------
    compute_loglike : compute the log-likelihood
    """
    def __init__(self, data_vector):
        self.data_vector = data_vector

    def read(self, sacc_data, sources, statistics):
        """Read the covariance matrirx for this likelihood from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        sources : dict
            A dictionary mapping sources to their objects. These sources do
            not have to have been rendered.
        statistics : dict
            A dictionary mapping statistics to their objects. These statistics do
            not have to have been rendered.
        """
        if not isinstance(sacc_data.covariance, sacc.covariance.FullCovariance):
            raise RuntimeError(
                "Currently, the SACC covariance must be a 'FullCovariance'. "
                "This may change in a future firecrown release.")
        _sd = sacc_data.copy()
        inds = []
        for stat in self.data_vector:
            inds.append(
                _sd.indices(
                    statistics[stat].sacc_data_type,
                    statistics[stat].sacc_tracers)
            )
        inds = np.concatenate(inds, axis=0)
        _sd.keep_indices(inds)
        self.cov = _sd.covariance.covmat.copy()
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)

    def compute(self, data, theory, **kwargs):
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
        return -0.5 * np.dot(x, x)
