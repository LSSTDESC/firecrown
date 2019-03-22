import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

import pyccl as ccl

from ..core import Source

Z_MIN = 0.0
Z_MAX = 5.0
M_MIN = 1.e+7
M_MAX = 1.e+18

__all__ = ['WLSource', 'NumberCountsSource', 'ClusterSource']


class WLSource(Source):
    """A CCL weak lensing Source.

    Parameters
    ----------
    dndz_data : str
        The path to the photo-z distribution in a CSV. The columns should be
        {'z', 'dndz'}.
    red_frac : str, optional
        The parameter for the red fraction. Only used if
        `has_intrinsic_alignment` is `True`.
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
    red_frac_ : np.ndarray, shape (n_z,)
        The red fraction as a function of redshift.  Set after a call to
        `render`. Only present in `has_intrinsic_alignment` is `True`.
    ia_bias_ : np.ndarray, shape (n_z,)
        The intrinsic alignment amplitude as a function of redshift. Set after
        a call to `render`. Only present in `has_intrinsic_alignment` is
        `True`.
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
            self, dndz_data, red_frac=None,
            ia_bias=None, scale=1.0, systematics=None):
        self.dndz_data = dndz_data
        self.red_frac = red_frac
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
        if self.red_frac is not None or self.ia_bias is not None:
            self.red_frac_ = np.ones_like(self.z_) * params[self.red_frac]
            self.ia_bias_ = np.ones_like(self.z_) * params[self.ia_bias]

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        if self.red_frac is not None or self.ia_bias is not None:
            tracer = ccl.WeakLensingTracer(
                cosmo,
                dndz=(self.z_, self.dndz_),
                ia_bias=(self.z_, self.ia_bias_),
                red_frac=(self.z_, self.red_frac_))
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
    tracer_ : `pyccl.NumberCountsTracer`
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


class ClusterSource(Source):
    """A CCL volume-limited count source binned MOR. Create separate source per mass proxy, redshift bin.

    Parameters
    ----------
    bin_data : str
        A CSV file with columns z_min, z_max, proxy_min, proxy_max, area_eff
    has_rsd : bool, optional
        If `True`, the source has RSD terms.
    scale : float, optional
        The default scale for this source. Usually the default of 1.0 is
        correct.
    systematics : list of str, optional
        A list of the source-level systematics to apply to the source. The
        default of `None` implies no systematics.

    Attributes
    ----------
    bin_data : str
        A CSV file with columns z_min, z_max, proxy_min, proxy_max, area_eff
    z_ : np.ndarray, shape (n_z,)
        An array of redshifts, on which PZ systematics operate. Set after a call
        to `render`.
    dndz_ : np.ndarray, shape (n_z,)
        The comoving volume element, which for a volume limited sample
        is the equivalent to the dndzhistogram.  Set after a call to `render`.
    dndz_interp : Akima1DInterpolator
        A spline interpolation of the comoving volume element
    bias_ : np.ndarray, shape (n_z,)
        The bias of the source. Set after a call to `render`.
    scale_ : float
        The overall scale associated with the source. Set after a call to
        `render`.
    tracer_ : `pyccl.NumberCountsTracer`
        The CCL tracer associated with this source. Set after a call to
        `render`.

    Methods
    -------
    integrate_pmor_dz_dm_dproxy : evaluate
        \int dz n(z) \int dM n(M,z) weight(M,z) \int dproxy P(proxy|M,z).
    render : apply systematics to this source and build the
        `pyccl.NumberCountsTracer`.
    """
    def __init__(
            self, bin_data, has_rsd=False,
            scale=1.0, systematics=None):
        df = pd.read_csv(bin_data)
        self._z_min = f['z_min'].values.copy()[0]
        self._z_max = f['z_max'].values.copy()[0]
        self._proxy_min = f['proxy_min'].values.copy()[0]
        self._proxy_max = f['proxy_max'].values.copy()[0]
        self._a_eff = f['area_eff'].values.copy()[0]
        self.bin_data = bin_data
        self.has_rsd = has_rsd
        self.systematics = systematics or []
        self.scale = scale

    def integrate_pmor_dz_dm_dproxy(self, cosmo, params, mor, weight=None):
        """Evaluate \int dz n(z) \int dM n(M,z) weight(M,z) \int dproxy P(proxy|M,z)
        if weight = None, this amounts to number count integral

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        mor: firecrown.ccl.systematic
            A Mass-Observable relation systematic.
        weight : function
            An optional weight function with signature (cosmo, halo mass, scale factor).
        """

        if weight is None:
            weight = lambda cosmo, M, a : 1.0

        if weight is not None:
            norm = self.int_dz_dM_dproxy(cosmo, params, mor)
        else:
            norm = 1

        def _integrand_pmor_dz_dm_dproxy(z, ln_m, params):
            return (np.exp(ln_m) * self.dndz_interp(z) *
                    ccl.massfunc(cosmo,np.exp(ln_m),1/(1+z)) *
                    mor.integrate_p_dproxy(params,ln_m,z,self._proxy_min,self._proxy_max) *
                    weight(cosmo,np.exp(ln_m),1/(1+z)))

        result = scipy.integrate.dblquad(
            _integrand_dz_dm_dproxy,
            np.log(M_MIN),np.log(M_MAX),
            self._z_min,self._z_max,args=(params,))[0]
        return result / norm


    def render(self, cosmo, params, systematics=None):
        """Render a source by applying systematics.

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

        def _compute_comoving_volume_elements_zbin(self,cosmo):
            """Compute comoving volume elements for self.z_ in the range [self._z_min, self._z_max].
            Return zero outside range [self._z_min, self._z_max].
            """
            _dndz_masked = np.zeros_like(self.z_)
            _z_in_range = np.where((self.z_ >= self._z_min) & (self.z_ <= self._z_max))
            _a = 1./(1+self.z_[_z_in_range])
            dndz_masked[_z_in_range] = (
                ccl.h_over_h0(cosmo,_a) *
                ccl.comoving_radial_distance(cosmo, _a)**2)
            return dndz_masked

        systematics = systematics or {}

        self.z_ = np.linspace(Z_MIN, Z_MAX, num=500)
        self.dndz_ = _compute_comoving_volume_elements_zbin(cosmo)
        self.dndz_interp = Akima1DInterpolator(
            self.z_ , self.dndz_)
        self.scale_ = self.scale
        self.bias_ = np.ones_like(self.z_)

        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        #TODO: check that self.systematics includes some type of MOR

        tracer = ccl.NumberCountsTracer(
            cosmo,
            has_rsd=self.has_rsd,
            dndz=(self.z_, self.dndz_),
            bias=(self.z_, self.bias_))
        self.tracer_ = tracer
