import numpy as np
import pyccl as ccl

from ..core import Statistic


class ClusterCountStatistic(Statistic):
    """A cluster count statistic.

    Parameters
    ----------
    sources : list of str
        The source used to compute the counts. Should always be length one.
    mass_def : list of args
        The mass definition to use. A valid entry might be `[200, 'matter']`.
    mass_func : str
        The mass function to use. Should be in a form such that "MassFunc"
        + mass_func results in a valid class in `pyccl.halos`.
    halo_bias : str
        The halo bias function to use. Should be in a form such that "HalosBias"
        + halo_bias results in a valid class in `pyccl.halos`.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.
    na : int, optional
        The number of points in scale factor used for integrations.
    nlog10M : int, optional
        The number of points used in log10M for integrations.

    Attributes
    ----------
    sacc_tracers : 1-tuple of str
        A tuple of the SACC tracer name for this count statistic. Set after a
        call to read.
    measured_statistic_ : np.ndarray
        The measured value for the statistic.
    predicted_statistic_ : np.ndarray
        The final prediction for the statistic. Set after `compute` is called.
    """
    def __init__(
        self, sources, mass_def, mass_func, halo_bias,
        systematics=None, na=256, nlog10M=256
    ):
        self.sources = sources
        self.mass_def = mass_def
        self.mass_func = mass_func
        self.halo_bias = halo_bias
        self.systematics = systematics or []
        self.na = na
        self.nlog10M = nlog10M

        if len(sources) != 1:
            raise ValueError(
                "A firecrown count statistic should only have one "
                "source, you sent '%s'!" % self.sources)

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
        if len(tracers) != 1:
            raise RuntimeError(
                "A firecrown count statistic should only have one "
                "tracer, you sent '%s'!" % self.sources)

        counts = sacc_data.get_data_points(
            data_type='count',
            tracers=tracers
        )

        self.sacc_tracers = tuple(tracers)

        if len(counts) > 1:
            raise ValueError(
                "Only one data point per tracer can be included "
                "for counts!"
            )

        if len(counts) == 0:
            _stat = np.zeros(1)
            self.sacc_inds = None
        else:
            self.sacc_inds = np.atleast_1d(sacc_data.indices(
                "count",
                self.sacc_tracers))
            _stat = counts[0].value

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

        mdef = ccl.halos.MassDef(*self.mass_def)
        hmf = getattr(ccl.halos, "MassFunc" + self.mass_func)
        hmf = hmf(cosmo, mdef, mass_def_strict=False)
        hbf = getattr(ccl.halos, "HaloBias" + self.halo_bias)
        hbf = hbf(cosmo, mdef, mass_def_strict=False)

        hmc = ccl.halos.HMCalculator(
            cosmo, hmf, hbf, mdef,
            nlog10M=self.nlog10M,
            # not using splint here will result in very poor integrations and
            # inaccurate stats
            integration_method_M='spline',
        )

        _src = sources[self.sources[0]]

        def _sel(m, a):
            sel = np.atleast_2d(_src.selfunc_(np.log(m), a))
            msk = ~np.isfinite(sel)
            sel[msk] = 0.0
            return sel

        cnts = hmc.number_counts(
            cosmo,
            _sel,
            amin=1.0 / (1.0 + np.max(_src.z_)),
            amax=1.0 / (1.0 + np.min(_src.z_)),
            na=self.na
        )
        cnts *= _src.area_sr_

        self.predicted_statistic_ = cnts

        systematics = systematics or {}
        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)

        if not hasattr(self, '_stat'):
            self.measured_statistic_ = self.predicted_statistic_
        else:
            self.measured_statistic_ = self._stat.copy()
