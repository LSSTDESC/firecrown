"""Gaussian Family Statistic Module.

The Statistic class describing objects that implement methods to compute the
data and theory vectors for a :class:`GaussFamily` subclass.
"""

from __future__ import annotations

# Import base classes from _base.py
from firecrown.likelihood._base import (
    GuardedStatistic,
    Statistic,
    StatisticUnreadError,
    TrivialStatistic,
)

# All classes have been moved to _base.py
# This module now just re-exports them for backward compatibility

__all__ = [
    "Statistic",
    "StatisticUnreadError",
    "GuardedStatistic",
    "TrivialStatistic",
]
