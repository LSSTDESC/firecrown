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


@dataclass(frozen=True, kw_only=True)
class WindowedAngularPowerSpectrum:
    """Independent window support, measured bandpowers, and their shared window.

    Support multipoles need not match the number of measured bandpowers.
    SACC observation labels are weighted support sums, not the support grid
    or normalized band centers.

    :param support_ells: Integer multipoles on which the window is defined.
    :param bandpowers: Measured values, one per window column, not predictions.
    :param window: SACC BandpowerWindow shared with the fixture's data points;
        its values are support multipoles and its weights are unnormalized.
    """

    support_ells: npt.NDArray[np.int64]
    bandpowers: npt.NDArray[np.float64]
    window: sacc.BandpowerWindow


@dataclass(frozen=True, kw_only=True)
class SaccGalaxyWindowedFixture:
    """Windowed SACC data with independently generated named expectations.

    :param sacc_data: SACC data used by production extraction and construction.
    :param distributions: Redshift distributions keyed by tracer name.
    :param spectra: Windowed spectra keyed by ordered tracer names and data type.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]
    spectra: dict[tuple[TracerNames, str], WindowedAngularPowerSpectrum]


class NumCosmoCosmologyFixture(TypedDict):
    """Values provided by the NumCosmo cosmology fixtures."""

    cosmo: Nc.HICosmo
    p_ml: Nc.PowspecML
    p_mnl: Nc.PowspecMNL
    dist: Nc.Distance
