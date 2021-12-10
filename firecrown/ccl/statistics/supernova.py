import copy
import functools
import warnings

import numpy as np
import pyccl as ccl

from ..core import Statistic

# only supported types are here, any thing else will throw
# a value error
SACC_DATA_TYPE_TO_CCL_KIND = {
    "supernova": 'sn'
}


Z_FOR_MU_DEFAULTS = dict(min=0, max=2, n=100)


def _z_for_mu(*, min, max, n):
    """Build an array of z to sample the distance modulus
    predictions.
    """
    return  np.linspace(min, max,n)


@functools.lru_cache(maxsize=128)
def _cached_distmod(cosmo, tracers, z):
    a = 1./(1+z)
    return ccl.background.distance_modulus(cosmo, *tracers, np.array(a))


class TwoPointStatistic(Statistic):
    """A two-point statistic (e.g., shear correlation function, galaxy-shear
    correlation function, etc.).

    Parameters
    ----------
    sacc_data_type : str
    sources : list of str
        A list of the sources needed to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.
    z_for_mu : dict, optional
        A dictionary of options for making the z values at which to compute
        mus for use in real-space integrations. The possible keys are:

         - min : int, optional - The minimum redshift. Default is 0
         - max : float, optional - The maximum redshift. The default is 2
         - n: int, optional - The number of linearly spaced redshift bins. The default is 50

    Attributes
    ----------
    ccl_kind : str
        The CCL correlation function kind or 'cl' for power spectra corresponding
        to the SACC data type.
    sacc_tracers : 2-tuple of str
        A tuple of the SACC tracer names for this 2pt statistic. Set after a
        call to read.
    measured_statistic_ : np.ndarray
        The measured value for the statistic.
    predicted_statistic_ : np.ndarray
        The final prediction for the statistic. Set after `compute` is called.
    scale_ : float
        The final scale factor applied to the statistic. Set after `compute`
        is called. Note that this scale factor is already applied.
    """
# RH still needs to edit here
    def __init__(self, sacc_data_type, sources, systematics=None,
                 ell_for_xi=None, ell_or_theta=None, ell_or_theta_min=None,
                 ell_or_theta_max=None):
        self.sacc_data_type = sacc_data_type
        self.sources = sources
        self.systematics = systematics or []
        self.ell_for_xi = copy.deepcopy(ELL_FOR_XI_DEFAULTS)
        if ell_for_xi is not None:
            self.ell_for_xi.update(ell_for_xi)
        self.ell_or_theta = ell_or_theta
        self.ell_or_theta_min = ell_or_theta_min
        self.ell_or_theta_max = ell_or_theta_max

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

        # sacc is tracer order sensitive
        # so we try again if we didn't find anything
        for order in [1, -1]:
            tracers = tracers[::order]

            if self.ccl_kind == 'cl':
                _ell_or_theta, _stat = sacc_data.get_ell_cl(
                    self.sacc_data_type, *tracers, return_cov=False)
            else:
                _ell_or_theta, _stat = sacc_data.get_theta_xi(
                    self.sacc_data_type, *tracers, return_cov=False)

            if len(_ell_or_theta) > 0 and len(_stat) > 0:
                break

        if self.ell_or_theta is None and (len(_ell_or_theta) == 0 or len(_stat) == 0):
            raise RuntimeError(
                "Tracers '%s' have no 2pt data in the SACC file "
                "and no input ell or theta values were given!" % tracers)
        elif (
            self.ell_or_theta is not None
            and len(_ell_or_theta) > 0
            and len(_stat) > 0
        ):
            warnings.warn(
                "Tracers '%s' have 2pt data and you have specified `ell_or_theta` "
                "in the configuration. `ell_or_theta` is being ignored!" % tracers,
                warnings.UserWarning,
                stacklevel=2,
            )

        self.sacc_tracers = tuple(tracers)

        # at this point we default to the values in the sacc file
        if len(_ell_or_theta) == 0 or len(_stat) == 0:
            _ell_or_theta = _generate_ell_or_theta(**self.ell_or_theta)
            _stat = np.zeros_like(_ell_or_theta)
            self.sacc_inds = None
        else:
            self.sacc_inds = np.atleast_1d(sacc_data.indices(
                self.sacc_data_type,
                self.sacc_tracers))

        if self.ell_or_theta_min is not None:
            q = np.where(_ell_or_theta >= self.ell_or_theta_min)
            _ell_or_theta = _ell_or_theta[q]
            _stat = _stat[q]
            if self.sacc_inds is not None:
                self.sacc_inds = self.sacc_inds[q]

        if self.ell_or_theta_max is not None:
            q = np.where(_ell_or_theta <= self.ell_or_theta_max)
            _ell_or_theta = _ell_or_theta[q]
            _stat = _stat[q]
            if self.sacc_inds is not None:
                self.sacc_inds = self.sacc_inds[q]

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
