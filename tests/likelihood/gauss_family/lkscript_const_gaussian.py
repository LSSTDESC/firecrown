"""
Provides a trivial likelihood factory function for a ConstGaussian
for testing purposes.
"""

import numpy as np

from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.statistic.supernova import Supernova
from firecrown.likelihood.gauss_family.statistic.statistic import DataVector


def build_likelihood(_):
    """Return a ConstGaussian (likelihood) object."""
    statistic = Supernova(sacc_tracer="no-tracer")
    likelihood = ConstGaussian(statistics=[statistic])
    likelihood.cov = np.array([1.0])
    likelihood.cov.shape = (1, 1)
    statistic.data_vector = DataVector.create(np.array([0.0]))

    return likelihood
