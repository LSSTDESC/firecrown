"""Classes for defining bins used in the cluster likelihood.

This module contains the classes that define the bins and binning
used for cluster theoretical predictions within Firecrown.
"""

from abc import ABC, abstractmethod

import sacc


class NDimensionalBin(ABC):
    """Class which defines the interface for an N dimensional bin."""

    @property
    @abstractmethod
    def z_edges(self) -> tuple[float, float]:
        """Redshift bin edges."""

    @property
    @abstractmethod
    def mass_proxy_edges(self) -> tuple[float, float]:
        """Mass proxy bin edges."""

    @property
    @abstractmethod
    def radius_edges(self) -> tuple[float, float]:
        """Radius bin edges."""

    @property
    @abstractmethod
    def radius_center(self) -> float:
        """Radius bin edges."""

    def __str__(self) -> str:
        """Returns a string representation of the bin edges."""
        return f"[{self.z_edges}, {self.mass_proxy_edges}, {self.radius_edges}]\n"


class SaccBin(NDimensionalBin):
    """An implementation of the N dimensional bin using sacc tracers."""

    def __init__(self, coordinate_bins: list[sacc.BaseTracer]):
        self.coordinate_bins = coordinate_bins
        self.dimension = len(coordinate_bins)

    @property
    def z_edges(self) -> tuple[float, float]:
        """Redshift bin edges."""
        z_bin = [x for x in self.coordinate_bins if x.tracer_type == "bin_z"]
        if len(z_bin) != 1:
            raise ValueError("SaccBin must have exactly one z bin")
        return z_bin[0].lower, z_bin[0].upper

    @property
    def mass_proxy_edges(self) -> tuple[float, float]:
        """Mass proxy bin edges."""
        mass_bin = [x for x in self.coordinate_bins if x.tracer_type == "bin_richness"]
        if len(mass_bin) != 1:
            raise ValueError("SaccBin must have exactly one richness bin")
        return mass_bin[0].lower, mass_bin[0].upper

    @property
    def radius_edges(self) -> tuple[float, float]:
        """Radius bin edges."""
        radius_bin = [x for x in self.coordinate_bins if x.tracer_type == "bin_radius"]
        if len(radius_bin) != 1:
            raise ValueError("SaccBin must have exactly one radius bin")
        return radius_bin[0].lower, radius_bin[0].upper

    @property
    def radius_center(self) -> float:
        """Radius bin center."""
        radius_bin = [x for x in self.coordinate_bins if x.tracer_type == "bin_radius"]
        if len(radius_bin) != 1:
            raise ValueError("SaccBin must have exactly one radius bin")
        return radius_bin[0].center

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


class TupleBin(NDimensionalBin):
    """An implementation of the N dimensional bin using sacc tracers."""

    def __init__(self, coordinate_bins: list[tuple]):
        self.coordinate_bins = coordinate_bins

    @property
    def dimension(self) -> int:
        """Number of dimensions for this bin."""
        return len(self.coordinate_bins)

    @property
    def mass_proxy_edges(self) -> tuple[float, float]:
        """Mass proxy bin edges."""
        mass_bin = self.coordinate_bins[0]
        return mass_bin[0], mass_bin[1]

    @property
    def z_edges(self) -> tuple[float, float]:
        """Redshift bin edges."""
        z_bin = self.coordinate_bins[1]
        return z_bin[0], z_bin[1]

    @property
    def radius_edges(self) -> tuple[float, float]:
        """Radius bin edges."""
        radius_bin = self.coordinate_bins[2]
        return radius_bin[0], radius_bin[1]

    @property
    def radius_center(self) -> float:
        """Radius bin center."""
        radius_bin = self.coordinate_bins[2]
        return radius_bin[2]

    def __eq__(self, other: object) -> bool:
        """Two bins are equal if they have the same lower/upper bound."""
        if not isinstance(other, TupleBin):
            return False
        return self.coordinate_bins == other.coordinate_bins

    def __hash__(self) -> int:
        """One bin's hash is determined by the dimension and lower/upper bound."""
        bin_bounds = [(bin[0], bin[1]) for bin in self.coordinate_bins]
        return hash((self.dimension, tuple(bin_bounds)))
