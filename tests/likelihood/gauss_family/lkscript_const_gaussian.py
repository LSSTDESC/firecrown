"""
Provides a trivial likelihood factory function for a ConstGaussian
for testing purposes.
"""

import sacc

from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.statistic.supernova import Supernova


def build_likelihood(_):
    """Return a ConstGaussian (likelihood) object."""
    statistic = Supernova(sacc_tracer="pantheon")
    likelihood = ConstGaussian(statistics=[statistic])

    # We need the Sacc object to contain everything that the Supernova stat wil
    # read.
    sacc_data = sacc.Sacc()
    sacc_data.add_tracer("misc", "pantheon")
    sacc_data.add_data_point("supernova_distance_mu", (), 1.0)
    sacc_data.add_data_point("supernova_distance_mu", (), 4.0)
    sacc_data.add_data_point("supernova_distance_mu", (), -3.0)
    sacc_data.add_covariance([4.0, 9.0, 16.0])
    likelihood.read(sacc_data)
    return likelihood
