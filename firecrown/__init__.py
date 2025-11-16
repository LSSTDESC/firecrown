"""Firecrown: A framework for cosmological likelihood analysis.

Firecrown provides a framework for implementing cosmological likelihoods
and connecting them to external statistical analysis tools.
It is developed by the LSST Dark Energy Science Collaboration (DESC).

Key Features:
    - Flexible likelihood framework for various cosmological observables
    - Integration with CCL (Core Cosmology Library) for theoretical predictions
    - Support for two-point statistics, cluster abundance, weak lensing, and
      supernova distance moduli
    - Connector interfaces for samplers (e.g., CosmoSIS, NumCosmo, Cobaya)
    - Updatable parameter system for efficient likelihood evaluation

Main Submodules:
    - :mod:`likelihood`: Likelihood implementations and infrastructure
    - :mod:`models`: Theoretical models for cosmological observables
    - :mod:`modeling_tools`: Cosmological modeling utilities (CCL integration)
    - :mod:`parameters`: Parameter management system
    - :mod:`connector`: Interfaces to external sampling frameworks
    - :mod:`updatable`: Base classes for parameter updates
"""

from firecrown.version import __version__  # noqa: F401
