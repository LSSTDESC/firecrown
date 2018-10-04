import numpy as np
from .base import BaseLikelihood


class Gaussian(BaseLikelihood):
    def run(self, theory_results):
        d = theory_results.data_vector()
        mu = self.data.data_vector
        P = self.data.precision
        delta = d - mu
        chi2 = float(np.einsum('i,ij,j', delta, P, delta))
        like = -0.5 * chi2
        return like
