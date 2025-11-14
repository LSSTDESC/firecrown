"""Factory functions for creating likelihoods from SACC files.

This module provides factory functions to create likelihood objects by combining a SACC
file and a set of statistic factories. Users can define their own custom statistic
factories for advanced use cases or rely on the generic factory functions provided here
for simpler scenarios.

For straightforward contexts where all data in the SACC file is utilized, the generic
factories simplify the process. The user only needs to supply the SACC file and specify
which statistic factories to use, and the likelihood factory will handle the creation of
the likelihood object, assembling the necessary components automatically.

These functions are particularly useful when the full set of statistics present in a
SACC file is being used without the need for complex customization.
"""

from firecrown.likelihood._two_point import TwoPointFactory

from firecrown.likelihood.factories._sacc_utils import load_sacc_data
from firecrown.likelihood.factories._models import DataSourceSacc, TwoPointExperiment
from firecrown.likelihood.factories._builders import build_two_point_likelihood

__all__ = [
    # Re-exported dependencies (used in non-test code)
    "TwoPointFactory",
    # SACC utilities
    "load_sacc_data",
    # Data models
    "DataSourceSacc",
    "TwoPointExperiment",
    # Likelihood builders
    "build_two_point_likelihood",
]
