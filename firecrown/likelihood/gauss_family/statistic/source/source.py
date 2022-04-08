"""The classe in this file define ...

"""

from __future__ import annotations
from typing import Dict, Sequence
from abc import ABC, abstractmethod
from typing import final
import pyccl
import sacc
from firecrown.parameters import ParamsMap


def get_params_hash(params: Dict[str, float]):
    return repr(sorted(params.items()))


class Systematic:
    """The systematic (e.g., shear biases, photo-z shifts, etc.).

    This class currently has no methods at all, because the argument types for
    the `apply` method of different subclasses are different."""

    @abstractmethod
    def update_params(self, params: ParamsMap):
        pass

    def read(self, sacc_data: sacc.Sacc):
        pass


class Source(ABC):
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
    cosmo_hash: int
    tracer: pyccl.tracers.Tracer

    @final
    def read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.read(sacc_data)
        self._read(sacc_data)

    @abstractmethod
    def _read(self, sacc_data: sacc.Sacc):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        pass

    @abstractmethod
    def get_scale(self) -> float:
        pass

    @final
    def update_params(self, params: ParamsMap):
        if hasattr(self, "systematics"):
            for systematic in self.systematics:
                systematic.update_params(params)
        self._update_params(params)

    @abstractmethod
    def _update_params(self, params: ParamsMap):
        pass

    @abstractmethod
    def create_tracer(self, cosmo: pyccl.Cosmology, params: ParamsMap):
        pass

    @final
    def get_tracer(
        self, cosmo: pyccl.Cosmology, params: ParamsMap
    ) -> pyccl.tracers.Tracer:
        cur_hash = hash((cosmo, get_params_hash(params)))
        if hasattr(self, "cosmo_hash") and self.cosmo_hash == cur_hash:
            return self.tracer
        else:
            self.tracer, _ = self.create_tracer(cosmo, params)
            self.cosmo_hash = cur_hash
            return self.tracer
