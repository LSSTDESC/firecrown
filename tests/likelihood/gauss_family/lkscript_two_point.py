"""
Provides a trivial likelihood factory function for a ConstGaussian
for testing purposes.
"""

import numpy as np
import sacc

from firecrown.likelihood.likelihood import NamedParameters
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.number_counts import (
    NumberCounts,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory, PoweSpecAmplitudeParameter


def build_likelihood(params: NamedParameters):
    """Return a ConstGaussian (likelihood) object."""
    sacc_data = sacc.Sacc()

    src0 = NumberCounts(sacc_tracer="lens0")
    z = np.linspace(0, 0.8, 50) + 0.05
    dndz = np.exp(-0.5 * (z - 0.4) ** 2 / 0.02 / 0.02)
    sacc_data.add_tracer("NZ", "lens0", z, dndz)

    if params.get_string("projection") == "harmonic":
        two_point = TwoPoint("galaxy_density_cl", source0=src0, source1=src0)
        ells = np.unique(np.logspace(1, 3, 10).astype(np.int64))
        Cells = np.random.normal(size=ells.shape[0])
        sacc_data.add_ell_cl("galaxy_density_cl", "lens0", "lens0", ells, Cells)
        cov = np.diag(np.ones_like(Cells) * 0.01)
        sacc_data.add_covariance(cov)
    elif params.get_string("projection") == "real":
        two_point = TwoPoint("galaxy_density_xi", source0=src0, source1=src0)
        thetas = np.linspace(0.0, np.pi, 10)
        xis = np.random.normal(size=thetas.shape[0])
        sacc_data.add_theta_xi("galaxy_density_xi", "lens0", "lens0", thetas, xis)
        cov = np.diag(np.ones_like(xis) * 0.01)
        sacc_data.add_covariance(cov)
    else:
        raise ValueError("Invalid projection")

    likelihood = ConstGaussian(statistics=[two_point])
    likelihood.read(sacc_data)

    tools = ModelingTools(
        ccl_factory=CCLFactory(amplitude_parameter=PoweSpecAmplitudeParameter.SIGMA8)
    )

    return likelihood, tools
