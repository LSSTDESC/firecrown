"""Abstract base classes for GaussianFamily statistics.

"""

from __future__ import annotations
from typing import Optional, Sequence, final
from abc import abstractmethod
import sacc

from .....likelihood.likelihood import CosmologyBundle
from .....parameters import ParamsMap
from .....updatable import Updatable


class Systematic(Updatable):
    """An abstract systematic class (e.g., shear biases, photo-z shifts, etc.).

    This class currently has no methods at all, because the argument types for
    the `apply` method of different subclasses are different."""

    def read(self, sacc_data: sacc.Sacc):
        """This method is called to allow the systematic object to read from the
        appropriated sacc data."""


class Source(Updatable):
    """An abstract source class (e.g., a sample of lenses).

    Parameters
    ----------
    scale : 1.0, optional
        The default scale for this source.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.
    """

    systematics: Sequence[Systematic]
    cosmo_hash: Optional[int]
    tracers: Optional[Tracer]

    @final
    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file."""
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.read(sacc_data)
        self._read(sacc_data)

    @abstractmethod
    def _read(self, sacc_data: sacc.Sacc):
        """Abstract method to read the data for this source from the SACC file."""

    @abstractmethod
    def _update_source(self, params: ParamsMap):
        """Abstract method to update the source from the given ParamsMap."""

    @abstractmethod
    def _reset_source(self):
        """Abstract method to reset the source."""

    @final
    def _update(self, params: ParamsMap):
        """Implementation of Updatable interface method `_update`.

        This clears the current hash and tracer, and calls the abstract method
        `_update_source`, which must be implemented in all subclasses."""
        self.cosmo_hash = None
        self.tracers = None
        self._update_source(params)

    @final
    def _reset(self) -> None:
        """Implementation of the Updatable interface method `_reset`.

        This calls the abstract method `_reset_source`, which must be implemented by
        all subclasses."""
        self._reset_source()

    @abstractmethod
    def get_scale(self) -> float:
        """Abstract method to return the scales for this `Source`."""

    @abstractmethod
    def create_tracers(self, cosmo: CosmologyBundle):
        """Abstract method to create tracers for this `Source`, for the given
        cosmology."""

    @final
    def get_tracers(self, cosmo: CosmologyBundle) -> Sequence[Tracer]:
        """Return the tracer for the given cosmology.

        This method caches its result, so if called a second time with the same
        cosmology, no calculation needs to be done."""
        cur_hash = hash(cosmo)
        if hasattr(self, "cosmo_hash") and self.cosmo_hash == cur_hash:
            return self.tracers

        self.tracers, _ = self.create_tracers(cosmo)
        self.cosmo_hash = cur_hash
        return self.tracers


class Tracer:
    """Bundles together a pyccl.Tracer object with optional information about the
    underlying 3D field, a pyccl.nl_pt.PTTracer, and halo profiles."""

    def __init__(
        self, tracer, field=None, pt_tracer=None, halo_profile=None, halo_2pt=None
    ):
        self.ccl_tracer = tracer
        self.field = field
        self.pt_tracer = pt_tracer
        self.halo_profile = halo_profile

    @property
    def has_pt(self):
        return self.pt_tracer is not None

    @property
    def has_hm(self):
        return self.halo_profile is not None
