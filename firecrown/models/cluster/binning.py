"""This module contains the classes that define the bins and binning
used for cluster theoretical predictions within Firecrown."""

from typing import Tuple, List, TypeVar, Generic
from abc import ABC, abstractmethod
import sacc

T = TypeVar("T")


class NDimensionalBin(Generic[T], ABC):
    """Class which defines the interface for an N dimensional bin used in
    the cluster likelihood."""

    def __init__(self, coordinate_bins: List[T]):
        """_summary_

        Args:
            coordinate_bins (List[T]): _description_
            dimension (int): _description_
        """
        self.coordinate_bins = coordinate_bins
        self.dimension = len(coordinate_bins)

    @property
    @abstractmethod
    def z_edges(self) -> Tuple[float, float]:
        """Redshift bin edges"""

    @property
    @abstractmethod
    def mass_proxy_edges(self) -> Tuple[float, float]:
        """Mass proxy bin edges"""

    def __str__(self) -> str:
        return f"[{self.z_edges}, {self.mass_proxy_edges}]\n"

    def __repr__(self) -> str:
        return f"[{self.z_edges}, {self.mass_proxy_edges}]\n"


class SaccBin(NDimensionalBin[sacc.BaseTracer]):
    """An implementation of the N dimensional bin using sacc tracers."""

    def __init__(self, bins: List[sacc.BaseTracer]):
        super().__init__(bins)

    @property
    def z_edges(self) -> Tuple[float, float]:
        z_bin = [x for x in self.coordinate_bins if x.tracer_type == "bin_z"]
        if len(z_bin) != 1:
            raise ValueError("SaccBin must have exactly one z bin")
        return z_bin[0].lower, z_bin[0].upper

    @property
    def mass_proxy_edges(self) -> Tuple[float, float]:
        mass_bin = [x for x in self.coordinate_bins if x.tracer_type == "bin_richness"]
        if len(mass_bin) != 1:
            raise ValueError("SaccBin must have exactly one richness bin")
        return mass_bin[0].lower, mass_bin[0].upper

    def __eq__(self, other: object) -> bool:
        """Two bins are equal if they have the same lower/upper bound."""
        if not isinstance(other, SaccBin):
            return False

        if self.dimension != other.dimension:
            return False

        for my_bin in self.coordinate_bins:
            other_bin = [
                x for x in other.coordinate_bins if x.tracer_type == my_bin.tracer_type
            ]

            if my_bin.lower != other_bin[0].lower:
                return False
            if my_bin.upper != other_bin[0].upper:
                return False

        return True

    def __hash__(self) -> int:
        """One bin's hash is determined by the dimension and lower/upper bound."""
        bin_bounds = [(bin.lower, bin.upper) for bin in self.coordinate_bins]
        return hash((self.dimension, tuple(bin_bounds)))


class TupleBin(NDimensionalBin[Tuple]):
    """An implementation of the N dimensional bin using sacc tracers."""

    def __init__(self, bins: List[Tuple]):
        super().__init__(bins)

    @property
    def mass_proxy_edges(self) -> Tuple[float, float]:
        mass_bin = self.coordinate_bins[0]
        return mass_bin[0], mass_bin[1]

    @property
    def z_edges(self) -> Tuple[float, float]:
        z_bin = self.coordinate_bins[1]
        return z_bin[0], z_bin[1]

    def __eq__(self, other: object) -> bool:
        """Two bins are equal if they have the same lower/upper bound."""
        if not isinstance(other, TupleBin):
            return False

        if self.dimension != other.dimension:
            return False

        for i, my_bin in enumerate(self.coordinate_bins):
            other_bin = other.coordinate_bins[i]
            if len(my_bin) != len(other_bin):
                return False
            if my_bin[0] != other_bin[0]:
                return False
            if my_bin[1] != other_bin[1]:
                return False

        return True

    def __hash__(self) -> int:
        """One bin's hash is determined by the dimension and lower/upper bound."""
        bin_bounds = [(bin[0], bin[1]) for bin in self.coordinate_bins]
        return hash((self.dimension, tuple(bin_bounds)))