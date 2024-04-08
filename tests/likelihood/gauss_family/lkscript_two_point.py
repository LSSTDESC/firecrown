"""
Provides a trivial likelihood factory function for a ConstGaussian
for testing purposes.
"""

import numpy as np
import sacc

from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.statistic.source.number_counts import (
    NumberCounts,
)


def build_likelihood(_):
    """Return a ConstGaussian (likelihood) object."""

    src0 = NumberCounts(sacc_tracer="lens0")
    two_point = TwoPoint("galaxy_density_cl", source0=src0, source1=src0)
    likelihood = ConstGaussian(statistics=[two_point])

    sacc_data = sacc.Sacc()

    z = np.linspace(0, 0.8, 50) + 0.05
    ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))

    dndz = np.exp(-0.5 * (z - 0.4) ** 2 / 0.02 / 0.02)
    sacc_data.add_tracer("NZ", "lens0", z, dndz)

    Cells = np.random.normal(size=ells.shape[0])
    sacc_data.add_ell_cl("galaxy_density_cl", "lens0", "lens0", ells, Cells)

    cov = np.diag(np.ones_like(Cells) * 0.01)
    sacc_data.add_covariance(cov)

    likelihood.read(sacc_data)
    return likelihood
