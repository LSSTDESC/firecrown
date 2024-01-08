"""The integrator module

This module holds the classes that define the interface required to
integrate a function.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Callable


# pylint: disable=too-few-public-methods
class Integrator(ABC):
    """The integrator base class

    This class acts as an adapter around an integration library."""

    def __init__(self) -> None:
        self.integral_bounds: List[Tuple[float, float]] = []
        self.extra_args: List[float] = []

    @abstractmethod
    def integrate(self, func_to_integrate: Callable) -> float:
        """Call this method to integrate the provided integrand argument."""
