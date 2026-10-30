"""Shared types for test fixtures."""

from typing import TypeAlias, TypedDict

from numcosmo_py import Nc
import numpy as np
import numpy.typing as npt
import sacc

from firecrown.metadata_types import TracerNames

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
