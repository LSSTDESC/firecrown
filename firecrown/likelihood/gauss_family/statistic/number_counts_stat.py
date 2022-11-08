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

    def __init__(self, scale=None, tracers=None, z_min=None, z_max=None, nz=None, metadata=None):

        self.scale = scale
        self.z_min =  z_min
        self.z_max =  z_max
        self.nz = nz
        self.metadata = metadata

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
        systematics: Optional[List[Systematic]] = None,
    ):

        super().__init__()

        self.systematics = systematics or []
        self.sacc_data_type = sacc_data_type
        self.data_vector = None
        self.theory_vector = None
        self.sacc_tracer = None
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

    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

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

        tracer = []
        tracer2 = []
        z_min = []
        z_max = []
        nz = []
        metadata = []
        inds = []
        for i in range (0, len(sacc_data.data)):
            #IM assuming that there is only one tracer. We have to change if there is more. Also Im assuming the entire file is Number Counts
            tracer.append(sacc_data.data[i].tracers[0])
            metadata.append(sacc_data.tracers[tracer[i]].metadata)
            if sacc_data.data[i].data_type == 'cluster_mass_count_wl':
                z_min.append(metadata[i]["z_min"])
                z_max.append(metadata[i]["z_max"])
                nz.append(sacc_data.data[i].value)
       
        self.tracer_args = NumberCountStatsArgs(
            scale=self.scale, tracers=tracer, z_min=z_min, z_max=z_max, nz=nz, metadata=metadata)

        self.sacc_tracer = tracer
        
        self.sacc_inds = np.atleast_1d(
            sacc_data.indices("cluster_mass_count_wl")
        )


    def compute(self, cosmo: pyccl.Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a Number Count statistic."""


        return np.array(self.data_vector), np.array(theory_vector)
