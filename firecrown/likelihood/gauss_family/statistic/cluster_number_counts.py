"""Cluster Number Count statistic support.
This module reads the necessary data from a SACC file to compute the
theoretical prediction of cluster number counts inside bins of redshift
 and a mass proxy. For further information, check README.md.
"""

from __future__ import annotations
from typing import List, Optional, final

import numpy as np
from scipy.integrate import simps

from .statistic import Statistic, DataVector, TheoryVector
from .source.source import SourceSystematic
from .... import parameters
from ....parameters import (
    ParamsMap,
    RequiredParameters,
    DerivedParameterCollection,
)
from ....models.cluster_abundance_binned import ClusterAbundanceBinned
from ....models.cluster_mean_mass_bin import ClusterMeanMass
from ....models.cluster_mass import ClusterMass
from ....models.cluster_mass_rich_proxy import ClusterMassRich
from ....modeling_tools import ModelingTools
from .cluster_number_counts_enum import (
    SupportedTracerNames,
    SupportedDataTypes,
)


class ClusterNumberCountsArgs:
    """Class for number counts tracer builder argument."""

    def __init__(
        self,
        tracers=None,
        z_bins=None,
        Mproxy_bins=None,  # pylint: disable-msg=invalid-name
        nz=None,  # pylint: disable-msg=invalid-name
        metadata=None,
    ):
        self.z_bins = z_bins
        self.Mproxy_bins = Mproxy_bins  # pylint: disable-msg=invalid-name
        self.nz = nz  # pylint: disable-msg=invalid-name
        self.metadata = metadata
        self.tracers = tracers


class ClusterNumberCounts(Statistic):
    """A Cluster Number Count statistic (e.g., halo mass function,
     multiplicity functions, volume element,  etc.).
     This subclass implements the read and computes method for
     the Statistic class. It is used to compute the theoretical prediction of
     cluster number counts given a SACC file and a cosmology. For
     further information on how the SACC file shall be created,
     check README.md.

    Parameters
    ----------
    sacc_tracer : str
        The SACC tracer. There must be only one tracer for all
        the number Counts data points. Following the SACC file
        documentation README.md, this string should be
        'cluster_counts_true_mass'.
    sacc_data_type : str
        The kind of number count statistic. This must be a valid
        SACC data type that maps to one of the CCL correlation
        function kinds or a power spectra. So far, the only
        possible option is "cluster_mass_count_wl", which is a standard
        type in the SACC library.
    systematics : list of str, optional
        A list of the statistics-level systematics to apply to
        the statistic. The default of `None` implies no systematics.

    Attributes
    ----------
    data_vector : list of float
        A list with the number of clusters in each bin of redshift and Mproxy.
        Set after `read` is called.
    theory_vector
        A list with the theoretical prediction of the number of
        clusters in each bin of redshift and Mproxy.
        Set after `compute` is called.
    sacc_indices : list of str
        A list with the indices for the data vector that
        corresponds to the type of data provided in
        the SACC file. Set after `read` is called.
    """

    def __init__(
        self,
        sacc_tracer,
        sacc_data_type,
        cluster_mass,
        cluster_redshift,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.sacc_data_type = sacc_data_type
        self.systematics = systematics or []
        self.data_vector: Optional[DataVector] = None
        self.theory_vector: Optional[TheoryVector] = None
        self.cluster_mass = cluster_mass
        self.cluster_z = cluster_redshift
        self.cluster_abundance_binned = None
        if (
            type(self.cluster_mass).__name__
            == "ClusterMassRich"
        ):
            self.mu_p0 = parameters.create()
            self.mu_p1 = parameters.create()
            self.mu_p2 = parameters.create()
            self.sigma_p0 = parameters.create()
            self.sigma_p1 = parameters.create()
            self.sigma_p2 = parameters.create()
        try:
            self.ccl_kind = SupportedDataTypes[sacc_data_type.upper()].name
        except KeyError:
            supported_data = [data.name for data in SupportedDataTypes]
            raise KeyError(
                f"The SACC data type '{sacc_data_type}' is not "
                f"supported!"
                f"Supported names are: {supported_data}"
            ) from None
        try:
            self.tracer_name = SupportedTracerNames[sacc_tracer.upper()].name
        except KeyError:
            supported_tracers = [tracer.name for tracer in SupportedTracerNames]
            raise KeyError(
                f"The SACC tracer name '{sacc_tracer}' is not "
                f"supported!"
                f"Supported names are: {supported_tracers}"
            ) from None
        self.tracer_args: ClusterNumberCountsArgs

    @final
    def _reset(self) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def _update(self, params: ParamsMap) -> None:
        """Reset this systematic.

        This implementation has nothing to do."""

    @final
    def _required_parameters(self) -> RequiredParameters:
        """Return an empty RequiredParameters."""
        return RequiredParameters([])

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return an empty DerivedParameterCollection."""
        derived_parameters = DerivedParameterCollection([])

        return derived_parameters

    # pylint: disable-next=invalid-name
    def _compute_grids(self, cosmo, logN_tuple, logm_tuple, z_tuple, n_intervals=20):
        mu_p0 = self.mu_p0
        mu_p1 = self.mu_p1
        mu_p2 = self.mu_p2
        sigma_p0 = self.sigma_p0
        sigma_p1 = self.sigma_p1
        sigma_p2 = self.sigma_p2
        richess_mass_proxy = RMProxy()
        # pylint: disable-next=invalid-name
        logN_grid = np.linspace(logN_tuple[0], logN_tuple[1], n_intervals)

        logm_grid = np.linspace(logm_tuple[0], logm_tuple[1], n_intervals)
        z_grid = np.linspace(z_tuple[0], z_tuple[1], n_intervals)
        # pylint: disable-next=invalid-name
        Nmz_grid = np.zeros([len(z_grid), len(logN_grid), len(logm_grid)])
        # pylint: disable-next=invalid-name
        dlnN_dlog10N = np.log(10.0) / np.log10(10.0)
        for i, z in enumerate(z_grid):
            # pylint: disable-next=invalid-name
            dv = self.number_density_func.compute_differential_comoving_volume(cosmo, z)
            # pylint: disable-next=invalid-name
            for k, logm in enumerate(logm_grid):
                # pylint: disable-next=invalid-name
                nm = self.number_density_func.compute_number_density(cosmo, logm, z)
                # pylint: disable-next=invalid-name
                for j, logN in enumerate(logN_grid):
                    lk_rm = richess_mass_proxy.mass_proxy_likelihood(
                        logN, logm, z, mu_p0, mu_p1, mu_p2, sigma_p0, sigma_p1, sigma_p2
                    )
                    pdf = nm * dv * lk_rm
                    Nmz_grid[i, j, k] = pdf * dlnN_dlog10N
        return Nmz_grid, z_grid, logN_grid, logm_grid

    # pylint: disable-next=invalid-name
    def _richness_proxy_integral(self, cosmo, logN_bins, logm_interval, z_bins):
        logm_tuple = logm_interval
        bin_counts = []
        for i in range(0, len(z_bins) - 1):
            for j in range(0, len(logN_bins) - 1):
                z_tuple = (z_bins[i], z_bins[i + 1])
                # pylint: disable-next=invalid-name
                logN_tuple = (logN_bins[j], logN_bins[j + 1])
                # pylint: disable-next=invalid-name
                Nmz_grid, z_grid, logN_grid, logm_grid = self._compute_grids(
                    cosmo, logN_tuple, logm_tuple, z_tuple
                )
                integral = simps(
                    simps(simps(Nmz_grid, z_grid, axis=0), logN_grid, axis=0),
                    logm_grid,
                    axis=0,
                )
                bin_counts.append(integral)
        return bin_counts

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
        tracer = sacc_data.get_tracer(self.sacc_tracer.lower())
        metadata = tracer.metadata
#         proxy_type = metadata["Mproxy_type"].upper()
#         if (
#             SupportedTracerNames[self.sacc_tracer.upper()].value
#             != SupportedProxyTypes[proxy_type.upper()].value
#         ):
#             raise TypeError(
#                 f"The proxy {proxy_type} is not supported"
#                 f"by the tracer {self.sacc_tracer}"
#             )

        # pylint: disable-next=invalid-name
        nz = sacc_data.get_mean(
            data_type="cluster_mass_count_wl", tracers=(self.sacc_tracer,)
        )
        self.tracer_args = ClusterNumberCountsArgs(
            tracers=tracer,
            z_bins=metadata["z_edges"],
            Mproxy_bins=metadata["Mproxy_edges"],
            nz=nz,
            metadata=metadata,
        )
        self.data_vector = DataVector.from_list(nz)

        self.sacc_indices = sacc_data.indices(
            data_type="cluster_mass_count_wl", tracers=(self.sacc_tracer,)
        )
        self.cluster_abundance_binned = ClusterAbundanceBinned(
            self.cluster_mass, self.cluster_z, metadata["sky_area"]
        )

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a Number Count statistic using the data from the
        Read method, the cosmology object, and the Bocquet16 halo mass function.
                Check README.MD for a complete description of the method.

        Parameters
        ----------
        tools : ModelingTools firecrown object
            Firecrown object used to load the required cosmology.

        return
        --------
        theory_vector : Numpy Array of floats
            An array with the theoretical prediction of the number of clusters
            in each bin of redsfhit and mass.
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        z_bins = self.tracer_args.z_bins
        proxy_bins = self.tracer_args.Mproxy_bins
        theory_vector = []
        if self.sacc_tracer == "cluster_counts_true_mass":
            for i in range(len(z_bins) - 1):
                for j in range(len(proxy_bins) - 1):
                    bin_count = self.cluster_abundance_binned.compute_bin_N(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )

                    theory_vector.append(bin_count)
        elif self.sacc_tracer == "cluster_counts_richness_proxy":
            self.cluster_abundance_binned.cluster_m.proxy_params = [
                    self.mu_p0,
                    self.mu_p1,
                    self.mu_p2,
                    self.sigma_p0,
                    self.sigma_p1,
                    self.sigma_p2,
                ]
            for i in range(0, len(z_bins) - 1):
                for j in range(0, len(proxy_bins) - 1):
                    bin_count = self.cluster_abundance_binned.compute_bin_N(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    theory_vector.append(bin_count)


        elif self.sacc_tracer == "cluster_counts_richness_proxy_plusmean":
            self.cluster_abundance_binned.cluster_m.proxy_params = [
                    self.mu_p0,
                    self.mu_p1,
                    self.mu_p2,
                    self.sigma_p0,
                    self.sigma_p1,
                    self.sigma_p2,
                ]
            mean_mass_obj = ClusterMeanMass(self.cluster_mass, self.cluster_z,
                                            self.tracer_args.metadata["sky_area"], [True, False])
            mean_mass = []
            for i in range(0, len(z_bins) - 1):
                for j in range(0, len(proxy_bins) - 1):
                    bin_count = self.cluster_abundance_binned.compute_bin_N(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    theory_vector.append(bin_count)
                    mass_count = mean_mass_obj.compute_bin_logM(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    mean_mass.append(mass_count)

            theory_vector = theory_vector + mean_mass

        elif self.sacc_tracer == "cluster_counts_richness_meanonly_proxy":
            self.cluster_abundance_binned.cluster_m.proxy_params = [
                    self.mu_p0,
                    self.mu_p1,
                    self.mu_p2,
                    self.sigma_p0,
                    self.sigma_p1,
                    self.sigma_p2,
                ]
            mean_mass_obj = ClusterMeanMass(self.cluster_mass, self.cluster_z,
                                            self.tracer_args.metadata["sky_area"], [True, False])

            for i in range(0, len(z_bins) - 1):
                for j in range(0, len(proxy_bins) - 1):
                    mass_count = mean_mass_obj.compute_bin_logM(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    theory_vector.append(mass_count)

        return TheoryVector.from_list(theory_vector)
