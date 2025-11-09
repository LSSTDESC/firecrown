"""Cosmic shear likelihood factory for Firecrown analysis.

This module defines a generalized likelihood factory function for cosmic shear
analysis with configurable number of tomographic bins. It demonstrates how to:
- Set up weak lensing sources with photo-z systematics
- Create all auto and cross-correlation power spectra
- Build a complete Gaussian likelihood from SACC data

This template can be customized for different survey configurations,
systematic effects, and analysis requirements.
"""

import itertools as it
from pathlib import Path

import firecrown.likelihood.weak_lensing as wl
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.likelihood.factories import load_sacc_data
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory


def build_likelihood(params: NamedParameters):
    """Build a cosmic shear likelihood with configurable tomographic bins.

    Creates a Gaussian likelihood for cosmic shear analysis with:
    - Configurable number of weak lensing source populations (tomographic bins)
    - Photo-z shift systematics for each bin
    - All auto and cross-correlation power spectra combinations
    - SACC data file integration with validation

    Required parameters:
    - sacc_file: Path to SACC data file
    - n_bins: Number of tomographic redshift bins

    :param params: Named parameters containing configuration
    :return: Configured ConstGaussian likelihood object
    :raises ValueError: If required parameters are missing
    :raises FileNotFoundError: If SACC file does not exist
    """

    # Validate required configuration parameters
    if "sacc_file" not in params:
        raise ValueError("sacc_file must be provided in the configuration")
    if "n_bins" not in params:
        raise ValueError("n_bins must be provided in the configuration")

    sacc_file = Path(params.get_string("sacc_file"))
    n_bins = params.get_int("n_bins")

    # Create weak lensing sources for all tomographic bins
    # Each source maps to a SACC tracer (trc0, trc1, ..., trc{n_bins-1})
    # Photo-z shift systematics model redshift distribution uncertainties
    sources = [
        wl.WeakLensing(
            sacc_tracer=f"trc{i}",
            systematics=[wl.PhotoZShift(sacc_tracer=f"trc{i}")],
        )
        for i in range(n_bins)
    ]

    # Generate all unique source pair combinations (upper triangular)
    # For n_bins=2: (0,0), (0,1), (1,1)
    # For n_bins=3: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    # This includes both auto-correlations and cross-correlations
    stats = [
        TwoPoint("galaxy_shear_cl_ee", sourceX, sourceY)
        for sourceX, sourceY in it.combinations_with_replacement(sources, 2)
    ]

    # Build Gaussian likelihood from two-point statistics
    # Statistics order determines data vector structure in likelihood
    likelihood = ConstGaussian(statistics=stats)

    # Validate and load SACC data file
    if not sacc_file.exists():
        raise FileNotFoundError(f"SACC file not found: {sacc_file}")
    sacc_data = load_sacc_data(sacc_file)

    # Initialize likelihood with SACC data
    # - Two-point functions extract relevant power spectra
    # - Sources receive corresponding redshift distributions (n(z))
    # - Covariance matrix is loaded for parameter estimation
    likelihood.read(sacc_data)

    # Create modeling tools with CCL factory for computing non-linear power spectra
    # - CCLFactory provides cosmological calculations via Core Cosmology Library
    # - require_nonlinear_pk=True enables non-linear corrections
    modeling_tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True))

    # Return likelihood and modeling tools for parameter estimation
    # - likelihood: Configured ConstGaussian likelihood with SACC data
    # - modeling_tools: CCL-based tools for cosmological computations
    # Compatible with CosmoSIS, Cobaya and NumCosmo
    return likelihood, modeling_tools
