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

    def __init__(self, scale=None, tracers=None, z_min=None, z_max=None, logm_min=None, logm_max=None, nz=None, metadata=None):

        self.scale = scale
        self.z_min =  z_min
        self.z_max =  z_max
        self.logm_min =  logm_min
        self.logm_max =  logm_max
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

    def _dmdz_dV (self, logm, z, cosmo):
        """
        parameters
        __________
        logm: float
            Cluster mass
        z : float
            Cluster Redshift
        cosmo : pyccl Cosmology
        
        reuturn
        _______
        
        integrand : float
                    Number Counts pdf at z and logm.
        """
        a = 1./(1. + z)
        hmd_200c = pyccl.halos.MassDef(200, 'critical')
        mass = 10**(logm)
        hmf_200c = pyccl.halos.MassFuncTinker08(cosmo, mass_def=hmd_200c)
        nm = hmf_200c.get_mass_function(cosmo, mass, a)
        da = pyccl.background.angular_diameter_distance(cosmo, a)
        E = pyccl.background.h_over_h0(cosmo, a)
        dV = ((1.+z)**2)*(da**2)*pyccl.physical_constants.CLIGHT_HMPC/cosmo['h']/E
        integrand = nm * dV
        
        return integrand
    
        
    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """

        tracer = []
        z_min = []
        z_max = []
        logm_min = []
        logm_max = []
        nz = []
        data = []
        metadata = []
        inds = []
        for i in range (0, len(sacc_data.data)):
            #IM assuming that there is only one tracer. We have to change if there is more. Also Im assuming the entire file is Number Counts
            tracer.append(sacc_data.data[i].tracers[0])
            metadata.append(sacc_data.tracers[tracer[i]].metadata)
            data.append(sacc_data.data[i].value)
            if sacc_data.data[i].data_type == 'cluster_mass_count_wl':
                z_min.append(metadata[i]["z_min"])
                z_max.append(metadata[i]["z_max"])
                logm_min.append(np.log10(metadata[i]["Mproxy_min"]))
                logm_max.append(np.log10(metadata[i]["Mproxy_max"]))
                nz.append(sacc_data.data[i].value)
        self.tracer_args = NumberCountStatsArgs(
            scale=self.scale, tracers=tracer, z_min=z_min, z_max=z_max, 
             logm_min=logm_min, logm_max=logm_max, nz=nz, metadata=metadata)
        self.sacc_tracer = tracer
        self.sacc_inds = np.atleast_1d(
        sacc_data.indices("cluster_mass_count_wl")
        )
        self.data_vector = data


    def compute(self, cosmo: pyccl.Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a Number Count statistic."""
        z_bins = self.tracer_args.z_min
        logm_bins = self.tracer_args.logm_min
        z_bins.append(self.tracer_args.z_max[-1])
        logm_bins.append(self.tracer_args.logm_max[-1])
        z_bins = list(set(z_bins))
        logm_bins = list(set(logm_bins))
        theory_vector = []
        def integrand(logm, z):
            return self._dmdz_dV(logm, z, cosmo)
        norm = norm = 1./scipy.integrate.dblquad(integrand, z_bins[0], z_bins[-1],
        lambda x:logm_bins[0], lambda x:logm_bins[-1], epsrel=1.e-4)[0] 
        for i in range(len(z_bins) -1):
            for j in range(len(logm_bins)-1):
                bin_count = scipy.integrate.dblquad(integrand, z_bins[i], z_bins[i+1],
                lambda x:logm_bins[j], lambda x:logm_bins[j+1], epsrel=1.e-4)[0] 
                theory_vector.append(bin_count / norm)

        return np.array(self.data_vector), np.array(theory_vector)
