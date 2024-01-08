"""The integrator module

This module holds the classes that define the interface required to
integrate an assembled cluster abundance.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Callable


class Integrator(ABC):
    """The integrator base class

    This class acts as an adapter around an integration library, and must provides
    a specific set of methods to be used to integrate a cluster abundance integral."""

    def __init__(self) -> None:
        self.integral_bounds: List[Tuple[float, float]] = []
        self.extra_args: List[float] = []

    @abstractmethod
    def integrate(self, func_to_integrate: Callable) -> float:
        """Call this method to integrate the provided integrand argument."""
