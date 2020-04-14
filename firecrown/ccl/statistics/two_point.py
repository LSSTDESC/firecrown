import copy
import functools
import numpy as np
import pyccl as ccl

from ..core import Statistic

# only supported types are here, any thing else will throw
# a value error
SACC_DATA_TYPE_TO_CCL_KIND = {
    "galaxy_density_cl": 'cl',
    "galaxy_density_xi": 'gg',
    "galaxy_shearDensity_cl_e": 'cl',
    "galaxy_shearDensity_xi_t": 'gl',
    "galaxy_shear_cl_ee": 'cl',
    "galaxy_shear_xi_minus": 'l-',
    "galaxy_shear_xi_plus": 'l+',
}


ELL_FOR_XI_DEFAULTS = dict(min=2, mid=50, max=6e4, n_log=200)


def _ell_for_xi(*, ell_min, ell_mid, ell_max, n_log):
    """Build an array of ells to sample the power spectrum for real-space
    predictions.
    """
    return np.concatenate((
        np.linspace(ell_min, ell_mid-1, ell_mid-ell_min),
        np.logspace(np.log10(ell_mid), np.log10(ell_max), n_log)))


@functools.lru_cache(maxsize=128)
def _cached_angular_cl(cosmo, tracers, ells):
    return ccl.angular_cl(cosmo, *tracers, np.array(ells))


class TwoPointStatistic(Statistic):
    """A two-point statistic (e.g., shear correlation function, galaxy-shear
    correlation function, etc.).

    Parameters
    ----------
    sacc_data_type : str
        The kind of two-point statistic. This must be a valid SACC data type that
        maps to one of the CCL correlation function kinds or a power spectra.
        Possible options are

          - galaxy_density_cl : maps to 'cl' (a CCL angular power spectrum)
          - galaxy_density_xi : maps to 'gg' (a CCL angular position corr. function)
          - galaxy_shearDensity_cl_e : maps to 'cl' (a CCL angular power spectrum)
          - galaxy_shearDensity_xi_t : maps to 'gl' (a CCL angular cross-correlation
            between position and shear)
          - galaxy_shear_cl_ee : maps to 'cl' (a CCL angular power spectrum)
          - galaxy_shear_xi_minus : maps to 'l-' (a CCL angular shear corr.
            function xi-)
          - galaxy_shear_xi_plus : maps to 'l+' (a CCL angular shear corr.
            function xi-)

    sources : list of str
        A list of the sources needed to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.
    ell_for_xi : dict, optional
        A dictionary of options for making the ell values at which to compute
        Cls for use in real-space integrations. The possible keys are:

         - min : int, optional - The minimum angulare wavenumber to use for
           real-space integrations. Default is 2.
         - mid : int, optional - The midpoint angular wavenumber to use for
           real-space integrations. The angular wavenumber samples are linearly
           spaced at integers between `min` and `mid`. Default is 50.
         - max : float, optional - The maximum angular wavenumber to use for
           real-space integrations. The angular wavenumber samples are
           logarithmically spaced between `mid` and `max`. Default is 6e4.
         - n_log : int, optional - The number of logarithmically spaced angular
           wavenumber samples between `mid` and `max`. Default is 200.

    Attributes
    ----------
    ccl_kind : str
        The CCL correlation function kind or 'cl' for power spectra corresponding
        to the SACC data type.
    sacc_tracers : 2-tuple of str
        A tuple of the SACC tracer names for this 2pt statistic. Set after a
        call to read.
    ell_or_theta_ : np.ndarray
        The final array of ell/theta values for the statistic. Set after
        `compute` is called.
    measured_statistic_ : np.ndarray
        The measured value for the statistic.
    predicted_statistic_ : np.ndarray
        The final prediction for the statistic. Set after `compute` is called.
    scale_ : float
        The final scale factor applied to the statistic. Set after `compute`
        is called. Note that this scale factor is already applied.
    """
    def __init__(self, sacc_data_type, sources, systematics=None,
                 ell_for_xi=None):
        self.sacc_data_type = sacc_data_type
        self.sources = sources
        self.systematics = systematics or []
        self.ell_for_xi = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        if ell_for_xi is not None:
            self.ell_for_xi.update(ell_for_xi)

        if self.sacc_data_type in SACC_DATA_TYPE_TO_CCL_KIND:
            self.ccl_kind = SACC_DATA_TYPE_TO_CCL_KIND[self.sacc_data_type]
        else:
            raise ValueError(
                "The SACC data type '%s' is not supported!" % sacc_data_type)

        if len(sources) != 2:
            raise ValueError(
                "A firecrown 2pt statistic should only have two "
                "sources, you sent '%s'!" % self.sources)

    def read(self, sacc_data, sources):
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        sources : dict
            A dictionary mapping sources to their objects. These sources do
            not have to have been rendered.
        """

        tracers = [sources[src].sacc_tracer for src in self.sources]
        if len(tracers) != 2:
            raise RuntimeError(
                "A firecrown 2pt statistic should only have two "
                "tracers, you sent '%s'!" % self.sources)

        if self.ccl_kind == 'cl':
            _ell_or_theta, _stat = sacc_data.get_ell_cl(
                self.sacc_data_type, *tracers, return_cov=False)
        else:
            _ell_or_theta, _stat = sacc_data.get_theta_xi(
                self.sacc_data_type, *tracers, return_cov=False)

        # sacc is tracer order sensitive
        # so we try again if we didn't find anything
        if len(_ell_or_theta) == 0 or len(_stat) == 0:
            tracers = tracers[::-1]

            if self.ccl_kind == 'cl':
                _ell_or_theta, _stat = sacc_data.get_ell_cl(
                    self.sacc_data_type, *tracers, return_cov=False)
            else:
                _ell_or_theta, _stat = sacc_data.get_theta_xi(
                    self.sacc_data_type, *tracers, return_cov=False)

            if len(_ell_or_theta) == 0 or len(_stat) == 0:
                raise RuntimeError(
                    "Tracers '%s' have no 2pt data in the SACC file!" % tracers)

        self.sacc_tracers = tuple(tracers)

        # I don't think we need these copies, but being safe here.
        self._ell_or_theta = _ell_or_theta.copy()
        self._stat = _stat.copy()

    def compute(self, cosmo, params, sources, systematics=None):
        """Compute a two-point statistic from sources.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        sources : dict
            A dictionary mapping sources to their objects. The sources must
            already have been rendered by calling `render` on them.
        systematics : dict, optional
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        self.ell_or_theta_ = self._ell_or_theta.copy()

        tracers = [sources[k].tracer_ for k in self.sources]
        self.scale_ = np.prod([sources[k].scale_ for k in self.sources])

        if self.ccl_kind == 'cl':
            self.predicted_statistic_ = _cached_angular_cl(
                cosmo, tuple(tracers), tuple(self.ell_or_theta_.tolist())
            ) * self.scale_
        else:
            ells = _ell_for_xi(**self.ell_for_xi)
            cells = _cached_angular_cl(
                cosmo, tuple(tracers), tuple(ells.tolist()))
            self.predicted_statistic_ = ccl.correlation(
                cosmo, ells, cells, self.ell_or_theta_ / 60,
                corr_type=self.ccl_kind) * self.scale_

        systematics = systematics or {}
        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        if not hasattr(self, '_stat'):
            self.measured_statistic_ = self.predicted_statistic_
        else:
            self.measured_statistic_ = self._stat.copy()
