import numpy as np
from .base import BaseLikelihood


class Tdist(BaseLikelihood):
    def __init__(self, data):
        BaseLikelihood.__init__(self, data)
        if self.data.nsims <= 0:
            raise ValueError(
                'nsims needs to be a positive integer read in from SACC.')

    def run(self, theory_results):
        """T-distribution for the likelihood in the case of numerical
        covariance (Sellentin & Heavens 2016; arXiv:1511.05969). As the
        number of simulations increases, the
        t-distribution approaches a Gaussian. Hartlap factor no longer needed.
        The number of simulations are meant to be obtained from SACC.
        """
        d = theory_results.data_vector()
        mu = self.data.data_vector
        P = self.data.precision
        nsims = self.data.nsims
        delta = d-mu
        chi2 = float(np.einsum('i,ij,j', delta, P, delta))
        like = -0.5 * nsims*np.log(1.0 + chi2/(nsims-1.0))
        return like
