import numpy as np
from .base import BaseLikelihood

class Tdist(BaseLikelihood):
    def run(self, theory_results):
        """T-distribution for the likelihood in the case of numerical covariance (Sellentin & Heavens 2016; arXiv:1511.05969). 
        As the number of simulations increases, the t-distribution approaches a Gaussian. Hartlap factor no longer needed.
        """
        d = theory_results.data_vector()
        mu = self.data.data_vector
        P = self.data.precision
        nsims = 10**4 #can change this number to actual number of simulations
        delta = d-mu
        chi2 = float(np.einsum('i,ij,j', delta, P, delta))
        like = -0.5 * nsims*np.log(1.0 + chi2/(nsims-1.0))
        return like
