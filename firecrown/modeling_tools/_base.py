"""Abstract base classes for modeling tools.

This module contains abstract base classes for power spectrum modifiers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pyccl

from firecrown.updatable import Updatable

if TYPE_CHECKING:
    from firecrown.modeling_tools._modeling_tools import ModelingTools


class PowerspectrumModifier(Updatable, ABC):
    """Abstract base class for power spectrum modifiers."""

    name: str = "base:base"

    @abstractmethod
    def compute_p_of_k_z(self, tools: ModelingTools) -> pyccl.Pk2D:
        """Compute the 3D power spectrum P(k, z)."""
