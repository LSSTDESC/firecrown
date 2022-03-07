from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.interpolate import Akima1DInterpolator

import pyccl as ccl

from ..core import Source
from ..systematics import IdentityFunctionMOR, TopHatSelectionFunction

__all__ = ["WLSource", "NumberCountsSource", "ClusterSource", "CMBLSource", "SNSource"]


class WLSource(Source):
    """A CCL weak lensing Source.

    Parameters
    ----------
    sacc_tracer : str
        The name of the tracer in the SACC file.
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
    z_orig : np.ndarray, shape (n_z,)
        The original redshifts for the photo-z distribution before any
        systematics are applied. Set after the call to `read`.
    dndz_orig : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes before any systematics are applied.
        Set after the call to `read`.
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

    def __init__(self, *, sacc_tracer, ia_bias=None, scale=1.0, systematics=None):
        self.sacc_tracer = sacc_tracer
        self.ia_bias = ia_bias
        self.systematics = systematics or []
        self.scale = scale
        self.z_orig: Optional[np.ndarray] = None
        self.dndz_orig: Optional[np.ndarray] = None
        self.dndz_interp = None

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
        if self.ia_bias is not None:
            self.ia_bias_ = np.ones_like(self.z_) * params[self.ia_bias]

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        if self.ia_bias is not None:
            tracer = ccl.WeakLensingTracer(
                cosmo, dndz=(self.z_, self.dndz_), ia_bias=(self.z_, self.ia_bias_)
            )
        else:
            tracer = ccl.WeakLensingTracer(cosmo, dndz=(self.z_, self.dndz_))
        self.tracer_ = tracer


class NumberCountsSource(Source):
    """A CCL number counts source.

    Parameters
    ----------
    sacc_tracer : str
        The name of the source in the SACC file.
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
    z_orig : np.ndarray, shape (n_z,)
        The original redshifts for the photo-z distribution before any
        systematics are applied. Set after the call to `read`.
    dndz_orig : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes before any systematics are applied.
        Set after the call to `read`.
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
    tracer_ : `pyccl.NumberCountsTracer`
        The CCL tracer associated with this source. Set after a call to
        `render`.

    Methods
    -------
    render : apply systematics to this source and build the
        `pyccl.NumberCountsTracer`
    """

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


class ClusterSource(Source):
    """A galaxy cluster source.

    Parameters
    ----------
    sacc_tracer : str
        The name of the source in the SACC file.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.

    Attributes
    ----------
    z_orig : np.ndarray, shape (n_z,)
        The original redshifts for the photo-z distribution before any
        systematics are applied. Set after the call to `read`.
    dndz_orig : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes before any systematics are applied.
        Set after the call to `read`.
    dndz_interp : Akima1DInterpolator
        A spline interpolation of the initial photo-z distribution.
    z_ : np.ndarray, shape (n_z,)
        The array of redshifts for the photo-z distribution. Set after a call
        to `render`.
    dndz_ : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes. Set after a call to `render`.
    dndz_interp_ : Akima1DInterpolator
        A spline interpolation of the final photo-z distribution. Set after a
        call to `render`.
    lnlam_min_orig : float
        The minimum lnlambda value read from the SACC file. Set after the call to
        `read`.
    lnlam_max_orig : float
        The maximum lnlambda value read from the SACC file. Set after the call to
        `read`.
    lnlam_min_ : float
        The minimum lnlambda value. Set after a call to `render`.
    lnlam_max_ : float
        The maximum lnlambda value. Set after a call to `render`.
    area_sr_orig : float
        The original area in steradians of the cluster sample.
    area_sr_ : float
        The (effective) area in steradians of the cluster sample.
    scale_ : float
        The overall scale associated with the source. Set after a call to
        `render`. Not currently used for anything.

    Methods
    -------
    render : apply systematics to this source, build the
        `pyccl.NumberCountsTracer`, and compute the linear bias
    """

    def __init__(self, *, sacc_tracer, systematics=None):
        self.sacc_tracer = sacc_tracer
        self.systematics = systematics or []
        self.scale = 1.0

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
        self.lnlam_min_orig = tracer.metadata["lnlam_min"]
        self.lnlam_max_orig = tracer.metadata["lnlam_max"]
        self.area_sr_orig = tracer.metadata["area_sd"] * (np.pi / 180.0) ** 2

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
        self.lnlam_min_ = self.lnlam_min_orig
        self.lnlam_max_ = self.lnlam_max_orig
        self.area_sr_ = self.area_sr_orig

        # set fiducial MOR and selection function systematics
        mor_sys = IdentityFunctionMOR()
        mor_sys.apply(cosmo, params, self)
        sel_sys = TopHatSelectionFunction()
        sel_sys.apply(cosmo, params, self)

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        self.dndz_interp_ = Akima1DInterpolator(self.z_, self.dndz_)


class CMBLSource(Source):
    """A CCL CMB Lensing Source.

    Parameters
    ----------
    sacc_tracer : str
        The name of the tracer in the SACC file.
    scale : float, optional
        The default scale for this source. Usually the default of 1.0 is
        correct.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.

    Attributes
    ----------
    scale_ : float
        The overall scale associated with the source. Set after a call to
        `render`.
    tracer_ : `pyccl.CMBLensingTracer`
        The CCL tracer associated with this source. Set after a call to
        `render`.

    Methods
    -------
    render : apply systematics to this source and build the
        `pyccl.CMBLSource`
    """

    def __init__(self, *, sacc_tracer, scale=1.0, systematics=None):
        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.systematics = systematics or []

    def read(self, sacc_data):
        """
        Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        pass

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

        self.scale_ = self.scale

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        tracer = ccl.CMBLensingTracer(cosmo, 1100.0)
        self.tracer_ = tracer

class SNSource(Source):
    """A galaxy cluster source.

    Parameters
    ----------
    sacc_tracer : str
        The name of the source in the SACC file.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.

    Attributes
    ----------
    z_orig : np.ndarray, shape (n_z,)
        The original redshifts for the photo-z distribution before any
        systematics are applied. Set after the call to `read`.
    dndz_orig : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes before any systematics are applied.
        Set after the call to `read`.
    dndz_interp : Akima1DInterpolator
        A spline interpolation of the initial photo-z distribution.
    z_ : np.ndarray, shape (n_z,)
        The array of redshifts for the photo-z distribution. Set after a call
        to `render`.
    dndz_ : np.ndarray, shape (n_z,)
        The photo-z distribution amplitudes. Set after a call to `render`.
    dndz_interp_ : Akima1DInterpolator
        A spline interpolation of the final photo-z distribution. Set after a
        call to `render`.
    lnlam_min_orig : float
        The minimum lnlambda value read from the SACC file. Set after the call to
        `read`.
    lnlam_max_orig : float
        The maximum lnlambda value read from the SACC file. Set after the call to
        `read`.
    lnlam_min_ : float
        The minimum lnlambda value. Set after a call to `render`.
    lnlam_max_ : float
        The maximum lnlambda value. Set after a call to `render`.
    area_sr_orig : float
        The original area in steradians of the cluster sample.
    area_sr_ : float
        The (effective) area in steradians of the cluster sample.
    scale_ : float
        The overall scale associated with the source. Set after a call to
        `render`. Not currently used for anything.

    Methods
    -------
    render : apply systematics to this source, build the
        `pyccl.NumberCountsTracer`, and compute the linear bias
    """

    def __init__(self, *, sacc_tracer, systematics=None):
        self.sacc_tracer = sacc_tracer
        self.systematics = systematics or []
        self.scale = 1.0

    def read(self, sacc_data):
        """Read the data for this source from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """
        tracer = sacc_data.get_tracer(self.sacc_tracer)
        z = np.array([datapt.get_tag("z") for datapt in sacc_data.get_data_points()])
        mu = np.array([datapt.value for datapt in sacc_data.get_data_points()])
        mb = np.array([datapt.get_tag("mb") for datapt in sacc_data.get_data_points()])
        inds = np.argsort(z)
        z = z[inds]
        mu = mu[inds]
        mb = mb[inds]
        self.mu_orig = mu
        self.z_orig = z
        self.mb_orig = mb
        #print("Yay! Read the SACC data file")

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
        self.mu_ = self.mu_orig.copy()
        self.mb_ = self.mb_orig.copy()
        self.scale_ = self.scale

        for systematic in self.systematics:
            print("Systematics not included")
            systematics[systematic].apply(cosmo, params, self)
