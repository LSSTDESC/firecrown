import functools
import numpy as np
import pandas as pd
import pyccl as ccl

from ..core import Statistic



class CountsStatistic(Statistic):
    """A number count statistic (e.g., cluster counts, etc.).

    Parameters
    ----------
    data : str
        The path to a CSV file with the measured statistic. The columns should
        be {'z_min', 'z_max', 'massproxy_min','massproxy_min','measured_statistic'}.
    source : list of str
        A source for which to compute this statistic.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.

    Attributes
    ----------
    measured_statistic_ : np.ndarray
        The measured value for the statistic.
    predicted_statistic_ : np.ndarray
        The final prediction for the statistic. Set after `compute` is called.
    scale_ : float
        The final scale factor applied to the statistic. Set after `compute`
        is called. Note that this scale factor is already applied.
    """
    def __init__(self, data, kind, source, systematics=None):
        self.data = data
        self.kind = kind

        df = pd.read_csv(self.data)
        if 'measured_statistic' not in df:
            self._stat = None
        else:
            self._stat = df['measured_statistic'].values.copy()

        self.source = source
        self.systematics = systematics or []

    def compute(self, cosmo, params, source, systematics=None):
        """Compute a number counts statistic from source.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            A pyccl.Cosmology object.
        params : dict
            A dictionary mapping parameter names to their current values.
        source : dict
            A dictionary mapping source to its object. The source must
            already have been rendered by calling `render` on it.
        systematics : dict, optional
            A dictionary mapping systematic names to their objects. The
            default of `None` corresponds to no systematics.
        """
        systematics = systematics or {}
        for systematic in self.systematics:
            systematics[systematic].apply(cosmo, params, self)


        mor = #need to filter the mor systematic from self.systematics
        self.predicted_statistic_ = source.integrate_pmor_dz_dm_dproxy(
        	cosmo, params, mor)


        if self._stat is None:
            self.measured_statistic_ = self.predicted_statistic_
        else:
            self.measured_statistic_ = self._stat.copy()
