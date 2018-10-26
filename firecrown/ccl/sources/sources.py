import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

import pyccl as ccl

from ..core import Source


__all__ = ['WLSource']


class WLSource(Source):
    """A CCL weak lensing Source.

    Parameters
    ----------
    nz_data : str
        The path to the photo-z distribution in a CSV. The columns should be
        {'z', 'nz'}.
    has_intrinsic_alignment : bool, optional
        If `True`, the source has intrinsic alignment terms.
    f_red : str, optional
        The parameter for the red fraction. Only used if
        `has_intrinsic_alignment` is `True`.
    bias_ia : str, optional
        The parameter for the intrinsic alignment amplitude. Only used if
        `has_intrinsic_alignment` is `True`.
    scale : float, optional
        The default scale for this source. Usually the default of 1.0 is
        correct.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.

    Attributes
    ----------
    nz_interp : Akima1DInterpolator
        A spline interpolation of the initial photo-z distribution.
    z_ : np.ndarray, shape (n_z,)
        The array of redshifts for the photo-z distribution. Set after a call
        to `render`.
    nz_ : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes.  Set after a call to `render`.
    f_red_ : np.ndarray, shape (n_z,)
        The red fraction as a function of redshift.  Set after a call to
        `render`. Only present in `has_intrinsic_alignment` is `True`.
    bias_ia_ : np.ndarray, shape (n_z,)
        The intrinsic alignment amplitude as a function of redshift. Set after
        a call to `render`. Only present in `has_intrinsic_alignment` is
        `True`.
    scale_ : float
        The overall scale associated with the source. Set after a call to
        `render`.
    tracer_ : `pyccl.CLTracerLensing`
        The CCL tracer associated with this source. Set after a call to
        `render`.

    Methods
    -------
    render : apply systematics to this source and build the
        `pyccl.ClTracerLensing`
    """
    def __init__(
            self, nz_data, has_intrinsic_alignment=False, f_red=None,
            bias_ia=None, scale=1.0, systematics=None):
        self.nz_data = nz_data
        self.has_intrinsic_alignment = has_intrinsic_alignment
        self.f_red = f_red
        self.bias_ia = bias_ia
        self.systematics = systematics or []
        df = pd.read_csv(nz_data)
        _z, _nz = df['z'].values.copy(), df['nz'].values.copy()
        self._z_orig = _z
        self._nz_orig = _nz
        self.nz_interp = Akima1DInterpolator(
            self._z_orig, self._nz_orig)
        self.scale = scale

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

        self.z_ = self._z_orig.copy()
        self.nz_ = self._nz_orig.copy()
        self.scale_ = self.scale
        if self.has_intrinsic_alignment:
            self.f_red_ = np.ones_like(self.z_) * params[self.f_red]
            self.bias_ia_ = np.ones_like(self.z_) * params[self.bias_ia]

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        if self.has_intrinsic_alignment:
            tracer = ccl.ClTracerLensing(
                cosmo,
                has_intrinsic_alignment=True,
                n=(self.z_, self.nz_),
                bias_ia=(self.z_, self.bias_ia_),
                f_red=(self.z_, self.f_red_))
        else:
            tracer = ccl.ClTracerLensing(
                cosmo,
                has_intrinsic_alignment=False,
                n=(self.z_, self.nz_))
        self.tracer_ = tracer
