"""Binned cluster number counts statistic support."""

from __future__ import annotations

import sacc
from firecrown.data_types import TheoryVector
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster import AbundanceData, ClusterProperty

from .binned_cluster import BinnedCluster
from .updatable_wrapper import UpdatableClusterObjects


class BinnedClusterNumberCounts(BinnedCluster):
    """A statistic representing the number of clusters in a z, mass bin."""

    def _create_updatable_parameters(self):
        self.updatable_parameters = UpdatableClusterObjects(
            (
                {
                    "attribute_name": "mass_distribution",
                    "parameters": [
                        "mu_p0",
                        "mu_p1",
                        "mu_p2",
                        "sigma_p0",
                        "sigma_p1",
                        "sigma_p2",
                    ],
                },
                {
                    "attribute_name": "cluster_theory",
                    "parameters": [],
                    "has_cosmo": True,
                },
            )
        )

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic and mark it as ready for use.

        :param sacc_data: The data in the sacc format.
        """
        # Build the data vector and indices needed for the likelihood
        if self.cluster_properties == ClusterProperty.NONE:
            raise ValueError("You must specify at least one cluster property.")
        cluster_data = AbundanceData(sacc_data)
        self._read(cluster_data)

        super().read(sacc_data)

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, concrete implementation.

        :param tools: The modeling tools used to compute the statistic.
        :return: The computed statistic.
        """
        self.updatable_parameters.export_all_parameters(
            self.cluster_recipe, tools.get_ccl_cosmology()
        )

        theory_vector_list: list[float] = []
        cluster_counts = []

        cluster_counts = self.get_binned_cluster_counts(tools)

        for cl_property in ClusterProperty:
            include_prop = cl_property & self.cluster_properties
            if not include_prop:
                continue

            if cl_property == ClusterProperty.COUNTS:
                theory_vector_list += cluster_counts
                continue
            if cl_property == ClusterProperty.DELTASIGMA:
                continue
            theory_vector_list += self.get_binned_cluster_property(
                tools, cluster_counts, cl_property
            )
        return TheoryVector.from_list(theory_vector_list)

    def get_binned_cluster_property(
        self,
        tools: ModelingTools,
        cluster_counts: list[float],
        cluster_properties: ClusterProperty,
    ) -> list[float]:
        """Computes the mean mass of clusters in each bin.

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted mean mass of
        the clusters in each bin.

        :param cluster_counts: The number of clusters in each bin.
        :param cluster_properties: The cluster observables to use.
        """

        mean_values = []
        for this_bin, counts in zip(self.bins, cluster_counts):
            total_observable = self.cluster_recipe.evaluate_theory_prediction_counts(
                this_bin.z_edges,
                this_bin.mass_proxy_edges,
                self.sky_area,
                cluster_properties,
            )
            mean_observable = total_observable / counts
            mean_values.append(mean_observable)

        return mean_values

    def get_binned_cluster_counts(self, tools: ModelingTools) -> list[float]:
        """Computes the number of clusters in each bin.

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted number of
        clusters in each bin.

        :param tools: The modeling tools used to compute the statistic.
        :return: The number of clusters in each bin.
        """

        cluster_counts = []
        for this_bin in self.bins:
            counts = self.cluster_recipe.evaluate_theory_prediction_counts(
                this_bin.z_edges, this_bin.mass_proxy_edges, self.sky_area
            )
            cluster_counts.append(counts)

        return cluster_counts
