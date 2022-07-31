"""

Two Point Statistic Source Module
=================================

The class in this file define ...

"""

from __future__ import annotations
from typing import Optional, Sequence, final
from abc import abstractmethod
import pyccl
import sacc
from .....parameters import ParamsMap
from .....updatable import Updatable


class Systematic(Updatable):
    """The systematic (e.g., shear biases, photo-z shifts, etc.).

    This class currently has no methods at all, because the argument types for
    the `apply` method of different subclasses are different."""

    def read(self, sacc_data: sacc.Sacc):
        """This method is called to allow the systematic object to read from the appropriated sacc data."""


class Source(Updatable):
    """The source (e.g., a sample of lenses).

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
    tracer: Optional[pyccl.tracers.Tracer]

    @final
    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file."""
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.read(sacc_data)
        self._read(sacc_data)

    @abstractmethod
    def _read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file."""
        pass

    @abstractmethod
    def _update_source(self, params: ParamsMap):
        pass

    @final
    def _update(self, params: ParamsMap):
        self.cosmo_hash = None
        self.tracer = None
        self._update_source(params)

    @abstractmethod
    def get_scale(self) -> float:
        pass

    @abstractmethod
    def create_tracer(self, cosmo: pyccl.Cosmology):
        pass

    @final
    def get_tracer(self, cosmo: pyccl.Cosmology) -> pyccl.tracers.Tracer:
        cur_hash = hash(cosmo)
        if hasattr(self, "cosmo_hash") and self.cosmo_hash == cur_hash:
            return self.tracer
        else:
            self.tracer, _ = self.create_tracer(cosmo)
            self.cosmo_hash = cur_hash
            return self.tracer
