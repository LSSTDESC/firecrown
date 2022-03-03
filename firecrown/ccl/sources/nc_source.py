from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

from ..core import Source
from ..core import Systematic
from ..parameters import get_from_prefix_param

__all__ = ["NumberCountsSource"]

@dataclass(frozen=True)
class NumberCountsSourceArgs:
    """Class for weak lensing tracer builder argument."""

    scale: float
    z: np.ndarray
    dndz: np.ndarray
    bias: np.ndarray
    mag_bias: np.ndarray

class NumberCountsSourceSystematic(Systematic):
    pass

class PhotoZShift(NumberCountsSourceSystematic):
    """A photo-z shift bias.

    This systematic shifts the photo-z distribution by some ammount `delta_z`.
    """

    params_names = ["delta_z"]

    def __init__(self, sacc_tracer: str):
        self.sacc_tracer = sacc_tracer
        self.delta_z = None

    def update_params(self, params):
        self.delta_z = get_from_prefix_param(self, params, self.sacc_tracer, "delta_z")

    def apply(self, cosmo: pyccl.Cosmology, tracer_arg: NumberCountsSourceArgs):
        """Apply a shift to the photo-z distribution of a source."""

        dndz_interp = Akima1DInterpolator(tracer_arg.z, tracer_arg.dndz)

        dndz = dndz_interp(tracer_arg.z - self.delta_z, extrapolate=False)
        dndz[np.isnan(dndz)] = 0.0

        return NumberCountsSourceArgs(
            scale=tracer_arg.scale,
            z=tracer_arg.z,
            dndz=dndz,
            bias=tracer_arg.bias,
            mag_bias=tracer_arg.mag_bias,
        )


class NumberCountsSource(Source):

    params_names = ["bias", "mag_bias"]

    def __init__(
        self,
        *,
        sacc_tracer,
        has_rsd=False,
        has_mag_bias=False,
        scale=1.0,
        systematics: Optional[List[NumberCountsSourceSystematic]] = None,
    ):
        self.sacc_tracer = sacc_tracer
        self.bias = None
        self.mag_bias = None
        self.has_rsd = has_rsd
        self.has_mag_bias = has_mag_bias

        self.systematics = []
        for systematic in systematics:
            self.systematics.append(systematic)

        self.scale = scale
        self.tracer_args = None
        self.current_tracer_args = None
        self.scale_ = None
        self.tracer_ = None

    def _update_params(self, params):
        self.bias = get_from_prefix_param(self, params, self.sacc_tracer, "bias")

        if self.has_mag_bias:
            self.mag_bias = get_from_prefix_param(
                self, params, self.sacc_tracer, "mag_bias"
            )

        for systematic in self.systematics:
            systematic.update_params(params)

    def _read(self, sacc_data):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        tracer = sacc_data.get_tracer(self.sacc_tracer)
        z = getattr(tracer, "z").copy().flatten()
        nz = getattr(tracer, "nz").copy().flatten()
        inds = np.argsort(z)
        z = z[inds]
        nz = nz[inds]

        self.tracer_args = NumberCountsSourceArgs(
            scale=self.scale, z=z, dndz=nz, bias=None, mag_bias=None
        )

    def create_tracer(self, cosmo: pyccl.Cosmology, params):
        tracer_args = self.tracer_args

        bias = np.ones_like(tracer_args.z) * self.bias
        tracer_args = NumberCountsSourceArgs(
            scale=tracer_args.scale,
            z=tracer_args.z,
            dndz=tracer_args.dndz,
            bias=bias,
            mag_bias=tracer_args.mag_bias,
        )

        if self.mag_bias is not None:
            mag_bias = np.ones_like(tracer_args.z) * self.mag_bias
            tracer_args = NumberCountsSourceArgs(
                scale=tracer_args.scale,
                z=tracer_args.z,
                dndz=tracer_args.dndz,
                bias=tracer_args.bias,
                mag_bias=mag_bias,
            )

        for systematic in self.systematics:
            tracer_args = systematic.apply(cosmo, tracer_args)

        if self.has_mag_bias:
            tracer = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=self.has_rsd,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=(tracer_args.z, tracer_args.bias),
                mag_bias=(tracer_args.z, tracer_args.mag_bias),
            )
        else:
            tracer = pyccl.NumberCountsTracer(
                cosmo,
                has_rsd=self.has_rsd,
                dndz=(tracer_args.z, tracer_args.dndz),
                bias=(tracer_args.z, tracer_args.bias),
            )
        self.current_tracer_args = tracer_args

        return tracer, tracer_args
    
    def get_scale(self):
        assert self.current_tracer_args
        return self.current_tracer_args.scale
