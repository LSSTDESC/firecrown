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
    tracer_tuple = ("pantheon",)
    sacc_data.add_tracer("misc", tracer_tuple[0])
    sacc_data.add_data_point("supernova_distance_mu", tracer_tuple, 1.0, z=0.1)
    sacc_data.add_data_point("supernova_distance_mu", tracer_tuple, 4.0, z=0.2)
    sacc_data.add_data_point("supernova_distance_mu", tracer_tuple, -3.0, z=0.3)
    sacc_data.add_covariance([4.0, 9.0, 16.0])
    likelihood.read(sacc_data)
    return likelihood
