from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pyccl
from scipy.interpolate import Akima1DInterpolator

import pyccl as ccl

from ..core import Source
from ..core import Systematic

__all__ = ["NumberCountsSource"]

class NumberCountsSource(Source):
    def __init__(
        self,
        *,
        sacc_tracer,
        bias,
        has_rsd=False,
        mag_bias=None,
        scale=1.0,
        systematics=None
    ):
        self.sacc_tracer = sacc_tracer
        self.bias = bias
        self.has_rsd = has_rsd
        self.mag_bias = mag_bias
        self.systematics = systematics or []
        self.scale = scale

    def read(self, sacc_data):
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
        self.z_orig = z
        self.dndz_orig = nz
        self.dndz_interp = Akima1DInterpolator(self.z_orig, self.dndz_orig)

    def render(self, cosmo, params, systematics=None):
        """
        Render a source by applying systematics.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        systematics : dict
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        systematics = systematics or {}

        self.z_ = self.z_orig.copy()
        self.dndz_ = self.dndz_orig.copy()
        self.scale_ = self.scale
        self.bias_ = np.ones_like(self.z_) * params[self.bias]

        if self.mag_bias is not None:
            self.mag_bias_ = np.ones_like(self.z_) * params[self.mag_bias]

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        if self.mag_bias is not None:
            tracer = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=self.has_rsd,
                dndz=(self.z_, self.dndz_),
                bias=(self.z_, self.bias_),
                mag_bias=(self.z_, self.mag_bias_),
            )
        else:
            tracer = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=self.has_rsd,
                dndz=(self.z_, self.dndz_),
                bias=(self.z_, self.bias_),
            )
        self.tracer_ = tracer
