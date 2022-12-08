"""Number Count statistic support.
This module reads the necessary data from a SACC file to compute the
theoretical prediction of cluster number counts inside bins of redshift
 and a mass proxy. For further information, check README.md.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, final

import numpy as np
import scipy.interpolate

import pyccl

from .statistic import Statistic
from .source.source import Systematic
from ....parameters import ParamsMap, RequiredParameters, DerivedParameterCollection

# only supported types are here, any thing else will throw
# a value error
SACC_DATA_TYPE_TO_CCL_KIND = {
    "cluster_mass_count_wl": "wl",
    "cluster_mass_count_xray": "xr",
    "counts": "nz",
}

SACC_TRACER_NAMES = {
    "cluster_counts_true_mass": "tm",
}


class NumberCountStatsArgs:
    """Class for number counts tracer builder argument."""

    def __init__(
        self, tracers=None, z_bins=None, Mproxy_bins=None, nz=None, metadata=None
    ):

        self.z_bins = z_bins
        self.Mproxy_bins = Mproxy_bins
        self.nz = nz
        self.metadata = metadata


class NumberCountStat(Statistic):
    """A Number Count statistic (e.g., halo mass function, multiplicity functions,
     volume element,  etc.). This subclass implements the read and computes method for
     the Statistic class. It is used to compute the theoretical prediction of
     cluster number counts given a SACC file and a cosmology. For further information
     on how the SACC file shall be created, check README.md.

    Parameters
    ----------
    sacc_tracer : str
        The SACC tracer. There must be only one tracer for all the number Counts
        data points. Following the SACC file documentation README.md, this string
        should be 'cluster_counts_true_mass'.
    sacc_data_type : str
        The kind of number count statistic. This must be a valid SACC data type that
        maps to one of the CCL correlation function kinds or a power spectra.
        So far, the only possible option is "cluster_mass_count_wl", which is a standard
        type in the SACC library.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to the statistic.
        The default of `None` implies no systematics.

    Attributes
    ----------
    data_vector : list of float
        A list with the number of clusters in each bin of redshift and Mproxy.
        Set after `read` is called.
    theory_vector
        A list with the theoretical prediction of the number of clusters in each bin
        of redshift and Mproxy. Set after `compute` is called.
    sacc_inds : list of str
        A list with the indices for the data vector that corresponds to the type of data
        provided in the SACC file. Set after `read` is called.
    """

    def __init__(
        self,
        sacc_tracer,
        sacc_data_type,
        number_density_func,
        systematics: Optional[List[Systematic]] = None,
    ):

        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.sacc_data_type = sacc_data_type
        self.systematics = systematics or []
        self.data_vector = None
        self.theory_vector = None
        self.number_density_func = number_density_func

        if self.sacc_data_type in SACC_DATA_TYPE_TO_CCL_KIND:
            self.ccl_kind = SACC_DATA_TYPE_TO_CCL_KIND[self.sacc_data_type]
        else:
            raise ValueError(
                f"The SACC data type {sacc_data_type}'%s' is not " f"supported!"
            )

        if self.sacc_tracer in SACC_TRACER_NAMES:
            self.tracer_name = SACC_TRACER_NAMES[self.sacc_tracer]
        else:
            raise ValueError(
                f"The SACC tracer name {sacc_tracer}'%s' is not " f"supported!"
            )

    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    def _update(self, params: ParamsMap) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    def _required_parameters(self) -> RequiredParameters:
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        derived_parameters = DerivedParameterCollection([])

        return derived_parameters

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.
        This function takes the SACC file and extract the necessary
        parameters needed to compute the number counts likelihood.
        Check README.MD for a complete description of the method.

        Parameters
        ----------
        sacc_data : sacc.Sacc
            The data in the sacc format.
        """

        tracer = sacc_data.get_tracer(self.sacc_tracer)
        metadata = tracer.metadata

        nz = sacc_data.get_mean(
            data_type="cluster_mass_count_wl", tracers=(self.sacc_tracer,)
        )
        self.tracer_args = NumberCountStatsArgs(
            tracers=tracer,
            z_bins=metadata["z_edges"],
            Mproxy_bins=metadata["Mproxy_edges"],
            nz=nz,
            metadata=metadata,
        )

        self.data_vector = nz
        self.sacc_inds = sacc_data.indices(
            data_type="cluster_mass_count_wl", tracers=(self.sacc_tracer,)
        )

    def compute(self, cosmo: pyccl.Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a Number Count statistic using the data from the
        Read method, the cosmology object, and the Bocquet16 halo mass function.
                Check README.MD for a complete description of the method.

        Parameters
        ----------
        cosmo : pyccl.Cosmology object
            The cosmology object that corresponds to the data in the SACC file.

        return
        --------
        data_vector : Numpy Array of floats
            An array with the number of clusters in each bin of redshift and proxy.
            Set after the read method is called.
        theory_vector : Numpy Array of floats
            An array with the theoretical prediction of the number of clusters
            in each bin of redsfhit and mass.
        """
        skyarea = self.tracer_args.metadata["sky_area"]
        DeltaOmega = skyarea * np.pi**2 / 180**2
        z_bins = self.tracer_args.z_bins
        logm_bins = self.tracer_args.Mproxy_bins
        theory_vector = []

        def integrand(logm, z):
            s = 1.0
            q = 0.0
            nuisance = (logm - 13.8) * s + q
            nm = self.number_density_func.compute_number_density(cosmo, logm, z)
            dv = self.number_density_func.compute_volume_density(cosmo, z)
            return nm * dv * nuisance

        for i in range(len(z_bins) - 1):
            for j in range(len(logm_bins) - 1):
                bin_count = scipy.integrate.dblquad(
                    integrand,
                    z_bins[i],
                    z_bins[i + 1],
                    lambda x: logm_bins[j],
                    lambda x: logm_bins[j + 1],
                    epsabs=1.0e-4,
                    epsrel=1.0e-4,
                )[0]
                theory_vector.append(bin_count * DeltaOmega)
        return np.array(self.data_vector), np.array(theory_vector)
