"""Shared types for test fixtures.

Frozen records bind named fields, but do not freeze nested arrays, maps, or SACC.
Array annotations specify dtype, not semantic role, shape, or physical units.
"""

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

from numcosmo_py import Nc
import numpy as np
import numpy.typing as npt
import sacc

from firecrown.metadata_types import TracerNames

@dataclass(frozen=True, kw_only=True)
class SampledRedshiftDistribution:
    """Independent sampled redshift coordinates and unnormalized number density.

    :param z: Sampled redshift coordinates.
    :param dndz: Number-density values on the redshift grid; not normalized.
    """

    z: npt.NDArray[np.float64]
    dndz: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class HarmonicAngularPowerSpectrum:
    """Independent unwindowed harmonic coordinates and measured cells.

    :param ells: Integer multipoles for the measurements.
    :param cells: Measured angular power spectrum values, not theory predictions.
    """

    ells: npt.NDArray[np.int64]
    cells: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SaccGalaxyHarmonicFixture:
    """Harmonic SACC data with independently generated named expectations.

    :param sacc_data: SACC data used by production extraction and construction.
    :param distributions: Redshift distributions keyed by tracer name.
    :param spectra: Measured spectra keyed by ordered tracer names and data type.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]
    spectra: dict[tuple[TracerNames, str], HarmonicAngularPowerSpectrum]


@dataclass(frozen=True, kw_only=True)
class RealAngularCorrelation:
    """Independent unwindowed angular coordinates and measured real-space xis.

    :param thetas: Angular coordinates for the measurements; no unit is enforced.
    :param xis: Measured correlations, not theory predictions or number densities.
    """

    thetas: npt.NDArray[np.float64]
    xis: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SaccGalaxyRealFixture:
    """Real SACC data with independently generated named expectations.

    :param sacc_data: SACC data used by production extraction and construction.
    :param distributions: Redshift distributions keyed by tracer name.
    :param spectra: Measured correlations keyed by ordered tracer names and data type.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]
    spectra: dict[tuple[TracerNames, str], RealAngularCorrelation]


RedshiftDistribution: TypeAlias = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]
"""Sampled redshift distribution as (z, dndz), without a normalization guarantee."""

WindowedAngularPowerSpectrum: TypeAlias = tuple[
    npt.NDArray[np.int64], npt.NDArray[np.float64], sacc.BandpowerWindow
]
"""Window ell grid, measured bandpowers, and their bandpower window."""

SaccGalaxyWindowedFixture: TypeAlias = tuple[
    sacc.Sacc,
    dict[str, RedshiftDistribution],
    dict[tuple[TracerNames, str], WindowedAngularPowerSpectrum],
]
"""SACC data, distributions by tracer name, and spectra by tracer pair/data type."""


class NumCosmoCosmologyFixture(TypedDict):
    """Values provided by the NumCosmo cosmology fixtures."""

    cosmo: Nc.HICosmo
    p_ml: Nc.PowspecML
    p_mnl: Nc.PowspecMNL
    dist: Nc.Distance
