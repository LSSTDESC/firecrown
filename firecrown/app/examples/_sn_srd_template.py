"""Supernova SRD likelihood factory for Firecrown analysis.

This module defines the likelihood factory function for supernova SRD analysis.
It demonstrates how to set up supernova sources and create the corresponding
likelihood from SACC data.
"""

from pathlib import Path

import sacc

import firecrown.likelihood.supernova as sn
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory


def build_likelihood(params: NamedParameters):
    """Build a supernova SRD likelihood.

    Creates a Gaussian likelihood for supernova analysis with:
    - Supernova source with systematic effects
    - SACC data file integration with validation

    Required parameters:
    - sacc_file: Path to SACC data file

    :param params: Named parameters containing configuration
    :return: Configured ConstGaussian likelihood object
    :raises ValueError: If required parameters are missing
    :raises FileNotFoundError: If SACC file does not exist
    """
    # Validate required configuration parameters
    if "sacc_file" not in params:
        raise ValueError("sacc_file must be provided in the configuration")

    sacc_file = Path(params.get_string("sacc_file"))

    # Create supernova statistic
    stat = sn.Supernova(sacc_tracer="sn_ddf_sample")

    # Build Gaussian likelihood from supernova statistics
    likelihood = ConstGaussian(statistics=[stat])

    # Validate and load SACC data file
    if not sacc_file.exists():
        raise FileNotFoundError(f"SACC file not found: {sacc_file}")

    sacc_data = sacc.Sacc.load_fits(sacc_file)

    # Initialize likelihood with SACC data
    # - Supernova statistic extracts relevant data
    # - Source receives corresponding supernova measurements
    # - Covariance matrix is loaded for parameter estimation
    likelihood.read(sacc_data)

    # Create modeling tools with CCL factory for cosmological calculations
    # - CCLFactory provides cosmological calculations via Core Cosmology Library
    # - require_nonlinear_pk=False for supernova analysis
    modeling_tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=False))

    # Return likelihood and modeling tools for parameter estimation
    # - likelihood: Configured ConstGaussian likelihood with SACC data
    # - modeling_tools: CCL-based tools for cosmological computations
    # Compatible with CosmoSIS, Cobaya and NumCosmo
    return likelihood, modeling_tools
