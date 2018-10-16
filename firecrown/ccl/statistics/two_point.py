import numpy as np
import pandas as pd
import pyccl as ccl

from ..core import Statistic


class TwoPointStatistic(Statistic):
    """A two-point statistic (e.g., shear correlation function, galaxy-shear
    correlation function, etc.).

    Parameters
    ----------
    data : str
        The path to a CSV file with the measured statistic. The columns should
        either be {'t', 'xi'} or {'l', 'cl'}.
    kind : str
        The kind of two-point statistic. One of
            - 'cl' : angular power spectrum
            - 'gg' : angular position auto-correlation function
            - 'gl' : angular cross-correlation between position and shear
            - 'l+' : angular shear auto-correlation function (xi+)
            - 'l-' : angular shear auto-correlation function (xi-)
    sources : list of str
        A list of the sources needed to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.

    Attributes
    ----------
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
    def __init__(self, data, kind, sources, systematics=None):
        self.data = data
        self.kind = kind
        df = pd.read_csv(self.data)
        if self.kind == 'cl':
            self._ell_or_theta = df['l'].values.copy()
            self._stat = df['cl'].values.copy()
        else:
            self._ell_or_theta = df['t'].values.copy()
            self._stat = df['xi'].values.copy()
        self.sources = sources
        self.systematics = systematics or []

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
        self.measured_statistic_ = self._stat.copy()

        tracers = [sources[k].tracer_ for k in self.sources]
        self.scale_ = np.prod([sources[k].scale_ for k in self.sources])

        if self.kind == 'cl':
            self.predicted_statistic_ = ccl.angular_cl(
                cosmo, *tracers, self.ell_or_theta_) * self.scale_

        systematics = systematics or {}
        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)
