"""Supernova statistics.

This subpackage provides the Supernova statistic class for Type Ia supernova
likelihood calculations.
"""

# Re-export the Supernova class from the private module
from firecrown.likelihood._supernova import Supernova

__all__ = [
    "Supernova",
]
