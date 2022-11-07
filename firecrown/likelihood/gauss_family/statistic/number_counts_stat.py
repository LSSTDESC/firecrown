"""Number Count statistic support.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, final
import copy
import functools
import warnings

import numpy as np
import scipy.interpolate

import pyccl

from .statistic import Statistic
from .source.source import Systematic
from ....parameters import ParamsMap, RequiredParameters, DerivedParameterCollection
from ....updatable import UpdatableCollection

# only supported types are here, any thing else will throw
# a value error
SACC_DATA_TYPE_TO_CCL_KIND = {
    "cluster_mass_count_wl": "wl",
    "cluster_mass_count_xray": "xr",
    "counts": "nz",
}

class NumberCountStatsArgs:
    """Class for number counts tracer builder argument."""

    def __init__(self, scale=None, z=None, dndz=None, bias=None, mag_bias=None):

        self.scale = scale
        self.z =  z  # pylint: disable-msg=invalid-name
        self.dndz = dndz
        self.bias = bias
        self.mag_bias = mag_bias

class NumberCountStat(Statistic):
    """A Number Count statistic (e.g., halo mass function, multiplicity functions,
     volume element,  etc.).

    Parameters
    ----------
    sacc_data_type : str
        The kind of number count statistic. This must be a valid SACC data type that
        maps to one of the CCL correlation function kinds or a power spectra.
        Possible options are


    Attributes
    ----------
    measured_statistic_ : np.ndarray
        The measured value for the statistic.
    predicted_statistic_ : np.ndarray
        The final prediction for the statistic. Set after `compute` is called.
    """


    def __init__(
        self,
        sacc_data_type,
        sacc_tracer: str,
        systematics: Optional[List[Systematic]] = None,
    ):

        super().__init__()

        self.systematics = systematics or []
        self.sacc_data_type = sacc_data_type
        self.data_vector = None
        self.theory_vector = None
        self.sacc_tracer = sacc_tracer
        self.scale = None
        self.systematics = UpdatableCollection([])
        if self.sacc_data_type in SACC_DATA_TYPE_TO_CCL_KIND:
            self.ccl_kind = SACC_DATA_TYPE_TO_CCL_KIND[self.sacc_data_type]
        else:
            raise ValueError(
                f"The SACC data type {sacc_data_type}'%s' is not " f"supported!"
            )
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    def _update(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    def required_parameters(self) -> RequiredParameters:
        return self.systematics.required_parameters()

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])
        derived_parameters = self.systematics.get_derived_parameters()

        return derived_parameters

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.

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
        self.tracer_args = NumberCountStatsArgs(
            scale=self.scale, z=z, dndz=nz, bias=None, mag_bias=None
        )
        self.sacc_inds = np.atleast_1d(
        sacc_data.indices(self.sacc_data_type, tuple(self.sacc_tracer))
        )




    def compute(self, cosmo: pyccl.Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a Number Count statistic."""


        return np.array(self.data_vector), np.array(theory_vector)
