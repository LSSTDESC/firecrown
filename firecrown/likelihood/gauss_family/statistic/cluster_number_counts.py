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
from ....models.cluster_abundance import ClusterAbundance
from ....models.cluster_mean_mass import ClusterMeanMass
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
    """

    def __init__(
        self,
        sacc_tracer: str,
        sacc_data_type,
        cluster_abundance: ClusterAbundance,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        """Initialize the ClusterNumberCounts object.
        Parameters

        :param sacc_tracer: The SACC tracer. There must be only one tracer for all
            the number Counts data points. Following the SACC file
            documentation README.md, this string should be
            'cluster_counts_true_mass'.
        :param sacc_data_type: The kind of number count statistic. This must be a valid
            SACC data type that maps to one of the CCL correlation
            function kinds or a power spectra. So far, the only
            possible option is "cluster_mass_count_wl", which is a standard
            type in the SACC library.
        :param cluster_abundance: The cluster abundance model to use.
        :param systematics: A list of the statistics-level systematics to apply to
            the statistic. The default of `None` implies no systematics.

        """
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.sacc_data_type = sacc_data_type
        self.systematics = systematics or []
        self.data_vector: Optional[DataVector] = None
        self.theory_vector: Optional[TheoryVector] = None
        self.cluster_abundance: ClusterAbundance = cluster_abundance
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

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.
        This function takes the SACC file and extract the necessary
        parameters needed to compute the number counts likelihood.
        Check README.MD for a complete description of the method.

        :param sacc_data: The data in the sacc format.
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

        self.cluster_abundance.read(sacc_data)
        self.data_vector = DataVector.from_list(nz)

        self.sacc_indices = sacc_data.indices(
            data_type="cluster_mass_count_wl", tracers=(self.sacc_tracer,)
        )
        self.cluster_abundance.set_sky_area(metadata["sky_area"])

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a Number Count statistic using the data from the
        Read method, the cosmology object, and the Bocquet16 halo mass function.
                Check README.MD for a complete description of the method.

        :param tools: ModelingTools firecrown object
            Firecrown object used to load the required cosmology.

        :return: Numpy Array of floats
            An array with the theoretical prediction of the number of clusters
            in each bin of redsfhit and mass.
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        z_bins = self.tracer_args.z_bins
        proxy_bins = self.tracer_args.Mproxy_bins
        theory_vector = []


        return TheoryVector.from_list(self.cluster_abundance.compute(ccl_cosmo))

        self.cluster_abundance.cluster_m.set_point()




        if self.sacc_tracer == "cluster_counts_true_mass":
            for i in range(len(z_bins) - 1):
                for j in range(len(proxy_bins) - 1):
                    bin_count = self.cluster_abundance.compute_N(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )

                    theory_vector.append(bin_count)
        elif self.sacc_tracer == "cluster_counts_richness_proxy":
            for i in range(0, len(z_bins) - 1):
                for j in range(0, len(proxy_bins) - 1):
                    bin_count = self.cluster_abundance.compute_intp_N(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    theory_vector.append(bin_count)


        elif self.sacc_tracer == "cluster_counts_richness_proxy_plusmean":
            mean_mass_obj = ClusterMeanMass(self.cluster_mass, self.cluster_z,
                                            self.tracer_args.metadata["sky_area"], [True, False])
            mean_mass = []
            for i in range(0, len(z_bins) - 1):
                for j in range(0, len(proxy_bins) - 1):
                    bin_count = self.cluster_abundance.compute_intp_N(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    theory_vector.append(bin_count)
                    mass_count = mean_mass_obj.compute_intp_logM(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    mean_mass.append(mass_count)

            theory_vector = theory_vector + mean_mass

        elif self.sacc_tracer == "cluster_counts_richness_meanonly_proxy":
            mean_mass_obj = ClusterMeanMass(self.cluster_mass, self.cluster_z,
                                            self.tracer_args.metadata["sky_area"], [True, False])

            for i in range(0, len(z_bins) - 1):
                for j in range(0, len(proxy_bins) - 1):
                    mass_count = mean_mass_obj.compute_intp_logM(
                        ccl_cosmo,
                        proxy_bins[j],
                        proxy_bins[j + 1],
                        z_bins[i],
                        z_bins[i + 1],
                    )
                    theory_vector.append(mass_count)

        return TheoryVector.from_list(theory_vector)
