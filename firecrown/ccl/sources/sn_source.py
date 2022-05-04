from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from ..core import Source
from ..core import Systematic
from ..core import SNSystematic
from ..parameters import get_from_prefix_param

__all__ = ["SNSource"]


@dataclass(frozen=True)
class SNSourceArgs:
    """Class for sn tracer builder argument."""

    scale: float
    z: np.ndarray
    mb: np.ndarray

class SNSourceSystematic(Systematic):
    pass
class PhotoZShift(SNSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.
    """

    params_names = ["delta_z"]

    def __init__(self, sacc_tracer: str):
        self.sacc_tracer = sacc_tracer
        self.delta_z = None

    def update_params(self, params):
        self.delta_z = get_from_prefix_param(self, params, self.sacc_tracer, "delta_z")

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: SNArgs):
        """Apply a shift to the photo-z distribution of a source."""

        mb_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.mb)

        mb = mb_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        mb[np.isnan(mb)] = 0.0
        
        return SNArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            mb=mb,
        )


class SNSource(Source):
    def __init__(
        self,
        *,
        sacc_tracer,
        scale=1.0,
        systematics: Optional[List[SNSystematic]] = None,
    ):
        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.z_orig: Optional[np.ndarray] = None
        self.mb_orig: Optional[np.ndarray] = None
        self.mb_interp = None
        self.tracer_args = None
        self.current_tracer_args = None
        self.scale_ = None
        self.tracer_ = None

        self.systematics = []
        for systematic in systematics:
            self.systematics.append(systematic)

    def _update_params(self, params):
        pass

    def _read(self, sacc_data):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        tracer = sacc_data.get_tracer(self.sacc_tracer)

        z = getattr(tracer, "z").copy().flatten()
        mb = getattr(tracer, "mb").copy().flatten()
        inds = np.argsort(z)
        z = z[inds]
        mb = mb[inds]

        self.tracer_args = SNSourceArgs(scale=self.scale, z=z, mb=mb, ia_bias=None)

    def create_tracer(self, cosmo: pyccl.Cosmology, params):
        """
        Render a source by applying systematics.

        """
        tracer_args = self.tracer_args

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        tracer = pyccl.SNTracer(
            cosmo, mb=(tracer_args.z, tracer_args.mb), ia_bias=tracer_args.ia_bias
        )
        self.current_tracer_args = tracer_args

        return tracer, tracer_args

    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale


