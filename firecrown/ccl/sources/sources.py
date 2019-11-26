import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

import pyccl as ccl

from ..core import Source


__all__ = ['WLSource', 'NumberCountsSource']


class WLSource(Source):
    """A CCL weak lensing Source.

    Parameters
    ----------
    dndz_data : str
        The path to the photo-z distribution in a CSV. The columns should be
        {'z', 'dndz'}.
    ia_bias : str, optional
        The parameter for the intrinsic alignment amplitude.
    scale : float, optional
        The default scale for this source. Usually the default of 1.0 is
        correct.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.

    Attributes
    ----------
    dndz_interp : Akima1DInterpolator
        A spline interpolation of the initial photo-z distribution.
    z_ : np.ndarray, shape (n_z,)
        The array of redshifts for the photo-z distribution. Set after a call
        to `render`.
    dndz_ : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes.  Set after a call to `render`.
    ia_bias_ : np.ndarray, shape (n_z,)
        The intrinsic alignment amplitude as a function of redshift. Set after
        a call to `render`. Only present in `is_bias` was non-None when the
        object was made.
    scale_ : float
        The overall scale associated with the source. Set after a call to
        `render`.
    tracer_ : `pyccl.WeakLensingTracer`
        The CCL tracer associated with this source. Set after a call to
        `render`.

    Methods
    -------
    render : apply systematics to this source and build the
        `pyccl.WeakLensingTracer`
    """
    def __init__(
            self, dndz_data, ia_bias=None, scale=1.0, systematics=None):
        self.dndz_data = dndz_data
        self.ia_bias = ia_bias
        self.systematics = systematics or []
        df = pd.read_csv(dndz_data)
        _z, _dndz = df['z'].values.copy(), df['dndz'].values.copy()
        self._z_orig = _z
        self._dndz_orig = _dndz
        self.dndz_interp = Akima1DInterpolator(
            self._z_orig, self._dndz_orig)
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
        self.dndz_ = self._dndz_orig.copy()
        self.scale_ = self.scale
        if self.ia_bias is not None:
            self.ia_bias_ = np.ones_like(self.z_) * params[self.ia_bias]

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        if self.ia_bias is not None:
            tracer = ccl.WeakLensingTracer(
                cosmo,
                dndz=(self.z_, self.dndz_),
                ia_bias=(self.z_, self.ia_bias_))
        else:
            tracer = ccl.WeakLensingTracer(
                cosmo,
                dndz=(self.z_, self.dndz_))
        self.tracer_ = tracer


class NumberCountsSource(Source):
    """A CCL number counts source.

    Parameters
    ----------
    dndz_data : str
        The path to the photo-z distribution in a CSV. The columns should be
        {'z', 'dndz'}.
    bias : str
        The parameter for the bias of the source.
    has_rsd : bool, optional
        If `True`, the source has RSD terms.
    mag_bias : str, optional
        The parameter for the magnification bias of the source.
    scale : float, optional
        The default scale for this source. Usually the default of 1.0 is
        correct.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.

    Attributes
    ----------
    dndz_interp : Akima1DInterpolator
        A spline interpolation of the initial photo-z distribution.
    z_ : np.ndarray, shape (n_z,)
        The array of redshifts for the photo-z distribution. Set after a call
        to `render`.
    dndz_ : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes.  Set after a call to `render`.
    bias_ : np.ndarray, shape (n_z,)
        The bias of the source. Set after a call to `render`.
    mag_bias_ : np.ndarray, shape (n_z,)
        The magnification bias of the source. Only used if `has_magnification`
        is `True` and only set after a call to `render`.
    scale_ : float
        The overall scale associated with the source. Set after a call to
        `render`.
    tracer_ : `pyccl.WeakLensingTracer`
        The CCL tracer associated with this source. Set after a call to
        `render`.

    Methods
    -------
    render : apply systematics to this source and build the
        `pyccl.NumberCountsTracer`
    """
    def __init__(
            self, dndz_data, bias, has_rsd=False,
            mag_bias=None, scale=1.0, systematics=None):
        self.dndz_data = dndz_data
        self.bias = bias
        self.has_rsd = has_rsd
        self.mag_bias = mag_bias
        self.systematics = systematics or []
        df = pd.read_csv(dndz_data)
        _z, _dndz = df['z'].values.copy(), df['dndz'].values.copy()
        self._z_orig = _z
        self._dndz_orig = _dndz
        self.dndz_interp = Akima1DInterpolator(
            self._z_orig, self._dndz_orig)
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
        self.dndz_ = self._dndz_orig.copy()
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
                mag_bias=(self.z_, self.mag_bias_))
        else:
            tracer = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=self.has_rsd,
                dndz=(self.z_, self.dndz_),
                bias=(self.z_, self.bias_))
        self.tracer_ = tracer
