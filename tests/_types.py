"""Shared types for test fixtures.

Expected arrays retain the inputs supplied to SACC, or copies of those inputs.
"""

from dataclasses import dataclass
from typing import TypedDict

from numcosmo_py import Nc
import numpy as np
import numpy.typing as npt
import sacc

from firecrown.metadata_types import TracerNames


@dataclass(frozen=True, kw_only=True)
class SampledRedshiftDistribution:
    """Sampled redshift coordinates and unnormalized number density.

    :param z: Sampled redshift coordinates.
    :param dndz: Number-density values on the redshift grid; not normalized.
    """

    z: npt.NDArray[np.float64]
    dndz: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SaccGalaxyDistributionFixture:
    """SACC tracer data and its sampled redshift distributions.

    :param sacc_data: SACC data used by production extraction and construction.
    :param distributions: Redshift distributions keyed by tracer name.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]


# Harmonic and real-space cross-bin records have distinct types because they
# represent different physical observables. Their shared fields can evolve
# independently as fixture values become more physically realistic.
@dataclass(frozen=True, kw_only=True)
class SaccHarmonicCrossBinFixture:
    """Harmonic cross-bin SACC data and expected redshift distributions.

    Both tracer distributions share their original redshift coordinate array,
    but retain distinct unnormalized number densities. No measurement metadata
    is inferred here; consumers still exercise production extraction.

    :param sacc_data: Unwindowed harmonic SACC data for the two tracers.
    :param distributions: Sampled redshift distributions keyed by tracer name.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]


@dataclass(frozen=True, kw_only=True)
class SaccRealCrossBinFixture:
    """Real cross-bin SACC data and expected redshift distributions.

    Both tracer distributions share their original redshift coordinate array,
    but retain distinct unnormalized number densities. No measurement metadata
    is inferred here; consumers still exercise production extraction.

    :param sacc_data: Unwindowed real-space SACC data for the two tracers.
    :param distributions: Sampled redshift distributions keyed by tracer name.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]


@dataclass(frozen=True, kw_only=True)
class HarmonicAngularPowerSpectrum:
    """Unwindowed harmonic coordinates and synthetic observations.

    :param ells: Integer multipoles for the synthetic observations.
    :param cells: Synthetic observed values, not survey data or theory predictions.
    """

    ells: npt.NDArray[np.int64]
    cells: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SaccGalaxyHarmonicFixture:
    """Harmonic SACC data with named expected distributions and spectra.

    :param sacc_data: SACC data used by production extraction and construction.
    :param distributions: Redshift distributions keyed by tracer name.
    :param spectra: Synthetic spectra keyed by ordered tracer names and data type.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]
    spectra: dict[tuple[TracerNames, str], HarmonicAngularPowerSpectrum]


@dataclass(frozen=True, kw_only=True)
class RealAngularCorrelation:
    """Unwindowed angular coordinates and synthetic observations.

    :param thetas: Angular coordinates for the observations.
    :param xis: Synthetic correlations, not survey data or theory predictions.
    """

    thetas: npt.NDArray[np.float64]
    xis: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class SaccGalaxyRealFixture:
    """Real SACC data with named expected distributions and correlations.

    :param sacc_data: SACC data used by production extraction and construction.
    :param distributions: Redshift distributions keyed by tracer name.
    :param spectra: Synthetic correlations keyed by ordered tracer names and data type.
    """

    sacc_data: sacc.Sacc
    distributions: dict[str, SampledRedshiftDistribution]
    spectra: dict[tuple[TracerNames, str], RealAngularCorrelation]


@dataclass(frozen=True, kw_only=True)
class WindowedAngularPowerSpectrum:
    """Window support, synthetic bandpowers, and their shared window.

    Support multipoles need not match the number of bandpowers.
    SACC observation labels are weighted support sums, not the support grid
    or normalized band centers.

    :param support_ells: Integer multipoles on which the window is defined.
    :param bandpowers: Synthetic observed values, not survey data or predictions;
        one per window column.
    :param window: SACC BandpowerWindow shared with the fixture's data points;
        its values are support multipoles and its weights are unnormalized.
    """

    support_ells: npt.NDArray[np.int64]
    bandpowers: npt.NDArray[np.float64]
    window: sacc.BandpowerWindow


@dataclass(frozen=True, kw_only=True)
class SaccGalaxyWindowedFixture:
    """Windowed SACC data with named expected distributions and spectra.

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
