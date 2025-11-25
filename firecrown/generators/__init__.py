"""Classes used to generate metadata for likelihoods and cosmological models.

This package provides generators for creating various types of metadata used in
Firecrown analyses. It includes:

- Two-point statistics generators for creating ell and theta values
- Inferred galaxy redshift distribution generators for LSST surveys

Public API
----------

Two-Point Generators
^^^^^^^^^^^^^^^^^^^^
These classes and functions generate ell and theta values for two-point statistics:

- LogLinearElls: Generator for log-linear integral ell values
- EllOrThetaConfig: Configuration dictionary for ell or theta generation
- generate_bin_centers: Generate bin centers for ell or theta values
- generate_ells_cells: Generate ells and Cells from configuration
- generate_reals: Generate theta and xi values from configuration

Inferred Galaxy Redshift Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These classes generate inferred galaxy redshift distributions for LSST:

- ZDistLSSTSRD: LSST inferred galaxy redshift distributions
- ZDistLSSTSRDBin: Single bin configuration for LSST redshift distributions
- ZDistLSSTSRDBinCollection: Collection of bins for LSST redshift distributions
- LinearGrid1D: 1D linear grid generator
- RawGrid1D: 1D grid from explicit values
- Grid1D: Type alias for LinearGrid1D or RawGrid1D

Constants
^^^^^^^^^
Constants for LSST Year 1 and Year 10 surveys:

- Y1_LENS_ALPHA, Y1_LENS_BETA, Y1_LENS_Z0: Year 1 lens distribution parameters
- Y1_SOURCE_ALPHA, Y1_SOURCE_BETA, Y1_SOURCE_Z0: Year 1 source parameters
- Y10_LENS_ALPHA, Y10_LENS_BETA, Y10_LENS_Z0: Year 10 lens parameters
- Y10_SOURCE_ALPHA, Y10_SOURCE_BETA, Y10_SOURCE_Z0: Year 10 source parameters
- Y1_LENS_BINS: Year 1 lens bins
- Y1_SOURCE_BINS: Year 1 source bins
- Y10_LENS_BINS: Year 10 lens bins
- Y10_SOURCE_BINS: Year 10 source bins
- LSST_Y1_LENS_HARMONIC_BIN_COLLECTION: Year 1 lens harmonic bin collection
- LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION: Year 1 source harmonic bin collection
- LSST_Y10_LENS_HARMONIC_BIN_COLLECTION: Year 10 lens harmonic bin collection
- LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION: Year 10 source harmonic bin collection

Type Definitions
^^^^^^^^^^^^^^^^
- BinsType: Type defining redshift bin configuration
- ZDistLSSTSRDOpt: Optional parameters for LSST redshift distributions
"""

# Two-point statistics generators
from ._two_point import (
    LogLinearElls,
    EllOrThetaConfig,
    generate_bin_centers,
    generate_ells_cells,
    generate_reals,
)

# Inferred galaxy redshift distribution generators
from ._inferred_galaxy_zdist import (
    # Core classes
    ZDistLSSTSRD,
    ZDistLSSTSRDBin,
    ZDistLSSTSRDBinCollection,
    LinearGrid1D,
    RawGrid1D,
    Grid1D,
    # Type definitions
    BinsType,
    ZDistLSSTSRDOpt,
    # Year 1 constants
    Y1_LENS_ALPHA,
    Y1_LENS_BETA,
    Y1_LENS_Z0,
    Y1_SOURCE_ALPHA,
    Y1_SOURCE_BETA,
    Y1_SOURCE_Z0,
    # Year 10 constants
    Y10_LENS_ALPHA,
    Y10_LENS_BETA,
    Y10_LENS_Z0,
    Y10_SOURCE_ALPHA,
    Y10_SOURCE_BETA,
    Y10_SOURCE_Z0,
    # Lazy-loaded bins (via __getattr__)
    Y1_LENS_BINS,
    Y1_SOURCE_BINS,
    Y10_LENS_BINS,
    Y10_SOURCE_BINS,
    LSST_Y1_LENS_HARMONIC_BIN_COLLECTION,
    LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION,
    LSST_Y10_LENS_HARMONIC_BIN_COLLECTION,
    LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION,
)

__all__ = [
    # Two-point generators
    "LogLinearElls",
    "EllOrThetaConfig",
    "generate_bin_centers",
    "generate_ells_cells",
    "generate_reals",
    # Inferred galaxy zdist - Core classes
    "ZDistLSSTSRD",
    "ZDistLSSTSRDBin",
    "ZDistLSSTSRDBinCollection",
    "LinearGrid1D",
    "RawGrid1D",
    "Grid1D",
    # Inferred galaxy zdist - Type definitions
    "BinsType",
    "ZDistLSSTSRDOpt",
    # Year 1 constants
    "Y1_LENS_ALPHA",
    "Y1_LENS_BETA",
    "Y1_LENS_Z0",
    "Y1_SOURCE_ALPHA",
    "Y1_SOURCE_BETA",
    "Y1_SOURCE_Z0",
    # Year 10 constants
    "Y10_LENS_ALPHA",
    "Y10_LENS_BETA",
    "Y10_LENS_Z0",
    "Y10_SOURCE_ALPHA",
    "Y10_SOURCE_BETA",
    "Y10_SOURCE_Z0",
    # Lazy-loaded bins
    "Y1_LENS_BINS",
    "Y1_SOURCE_BINS",
    "Y10_LENS_BINS",
    "Y10_SOURCE_BINS",
    "LSST_Y1_LENS_HARMONIC_BIN_COLLECTION",
    "LSST_Y1_SOURCE_HARMONIC_BIN_COLLECTION",
    "LSST_Y10_LENS_HARMONIC_BIN_COLLECTION",
    "LSST_Y10_SOURCE_HARMONIC_BIN_COLLECTION",
]
