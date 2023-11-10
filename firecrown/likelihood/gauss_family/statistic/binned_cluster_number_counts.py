"""This module holds classes needed to predict the binned cluster number counts

The binned cluster number counts statistic predicts the number of galaxy
clusters within a single redshift and mass bin.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import sacc
import numpy as np
from firecrown.models.cluster.integrator.integrator import Integrator
from firecrown.models.cluster.abundance_data import AbundanceData
from firecrown.likelihood.gauss_family.statistic.statistic import (
    Statistic,
    DataVector,
    TheoryVector,
)
from firecrown.likelihood.gauss_family.statistic.source.source import SourceSystematic
from firecrown.modeling_tools import ModelingTools


class BinnedClusterNumberCounts(Statistic):
    """The Binned Cluster Number Counts statistic

    This class will make a prediction for the number of clusters in a z, mass bin
    and compare that prediction to the data provided in the sacc file.
    """

    def __init__(
        self,
        cluster_counts: bool,
        mean_log_mass: bool,
        survey_name: str,
        integrator: Integrator,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        super().__init__()
        self.systematics = systematics or []
        self.theory_vector: Optional[TheoryVector] = None
        self.use_cluster_counts = cluster_counts
        self.use_mean_log_mass = mean_log_mass
        self.survey_name = survey_name
        self.integrator = integrator
        self.data_vector = DataVector.from_list([])
        self.sky_area = 0.0
        self.bin_limits: List[List[Tuple[float, float]]] = []

    def read(self, sacc_data: sacc.Sacc) -> None:
        # Build the data vector and indices needed for the likelihood

        data_vector = []
        sacc_indices = []

        sacc_types = sacc.data_types.standard_types
        sacc_adapter = AbundanceData(
            sacc_data, self.survey_name, self.use_cluster_counts, self.use_mean_log_mass
        )

        if self.use_cluster_counts:
            # pylint: disable=no-member
            data, indices = sacc_adapter.get_data_and_indices(sacc_types.cluster_counts)
            data_vector += data
            sacc_indices += indices

        if self.use_mean_log_mass:
            # pylint: disable=no-member
            data, indices = sacc_adapter.get_data_and_indices(
                sacc_types.cluster_mean_log_mass
            )
            data_vector += data
            sacc_indices += indices

        self.sky_area = sacc_adapter.survey_tracer.sky_area
        # Note - this is the same for both cl mass and cl counts... Why do we need to
        # specify a data type?

        # pylint: disable=no-member
        self.bin_limits = sacc_adapter.get_bin_limits(sacc_types.cluster_mean_log_mass)
        self.data_vector = DataVector.from_list(data_vector)

        self.sacc_indices = np.array(sacc_indices)
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        assert tools.cluster_abundance is not None

        theory_vector_list: List[float] = []
        cluster_counts = []

        if not self.use_cluster_counts and not self.use_mean_log_mass:
            return TheoryVector.from_list(theory_vector_list)

        cluster_counts = self.get_binned_cluster_counts(tools)

        if self.use_cluster_counts:
            theory_vector_list += cluster_counts

        if self.use_mean_log_mass:
            theory_vector_list += self.get_binned_cluster_masses(tools, cluster_counts)

        return TheoryVector.from_list(theory_vector_list)

    def get_binned_cluster_masses(
        self, tools: ModelingTools, cluster_counts: List[float]
    ) -> List[float]:
        """Computes the mean mass of clusters in each bin

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted mean mass of
        the clusters in each bin."""
        assert tools.cluster_abundance is not None

        cluster_masses = []
        for (z_proxy_limits, mass_proxy_limits), counts in zip(
            self.bin_limits, cluster_counts
        ):
            integrand = tools.cluster_abundance.get_integrand(avg_mass=True)
            self.integrator.set_integration_bounds(
                tools.cluster_abundance, z_proxy_limits, mass_proxy_limits
            )

            total_mass = self.integrator.integrate(integrand)
            mean_mass = total_mass / counts
            cluster_masses.append(mean_mass)

        return cluster_masses

    def get_binned_cluster_counts(self, tools: ModelingTools) -> List[float]:
        """Computes the number of clusters in each bin

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted number of
        clusters in each bin."""
        assert tools.cluster_abundance is not None

        cluster_counts = []
        for z_proxy_limits, mass_proxy_limits in self.bin_limits:
            self.integrator.set_integration_bounds(
                tools.cluster_abundance, z_proxy_limits, mass_proxy_limits
            )

            integrand = tools.cluster_abundance.get_integrand()
            counts = self.integrator.integrate(integrand)
            cluster_counts.append(counts)

        return cluster_counts
