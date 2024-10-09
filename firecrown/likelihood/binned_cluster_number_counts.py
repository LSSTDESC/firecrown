"""Binned cluster number counts statistic support."""

from __future__ import annotations

import numpy as np
import sacc

# firecrown is needed for backward compatibility; remove support for deprecated
# directory structure is removed.
import firecrown  # pylint: disable=unused-import # noqa: F401
from firecrown.likelihood.source import SourceSystematic
from firecrown.likelihood.statistic import (
    DataVector,
    Statistic,
    TheoryVector,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster.abundance_data import AbundanceData
from firecrown.models.cluster.binning import SaccBin
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe


class BinnedClusterNumberCounts(Statistic):
    """A statistic representing the number of clusters in a z, mass bin."""

    def __init__(
        self,
        cluster_properties: ClusterProperty,
        survey_name: str,
        cluster_recipe: ClusterRecipe,
        systematics: None | list[SourceSystematic] = None,
    ):
        """Initialize this statistic.

        :param cluster_properties: The cluster observables to use.
        :param survey_name: The name of the survey to use.
        :param cluster_recipe: The cluster recipe to use.
        :param systematics: The systematics to apply to this statistic.
        """
        super().__init__()
        self.systematics = systematics or []
        self.theory_vector: None | TheoryVector = None
        self.cluster_properties = cluster_properties
        self.survey_name = survey_name
        self.cluster_recipe = cluster_recipe
        self.data_vector = DataVector.from_list([])
        self.sky_area = 0.0
        self.bins: list[SaccBin] = []

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic and mark it as ready for use.

        :param sacc_data: The data in the sacc format.
        """
        # Build the data vector and indices needed for the likelihood
        if self.cluster_properties == ClusterProperty.NONE:
            raise ValueError("You must specify at least one cluster property.")

        sacc_adapter = AbundanceData(sacc_data)
        self.sky_area = sacc_adapter.get_survey_tracer(self.survey_name).sky_area

        data, indices = sacc_adapter.get_observed_data_and_indices_by_survey(
            self.survey_name, self.cluster_properties
        )
        self.data_vector = DataVector.from_list(data)
        self.sacc_indices = np.array(indices)

        self.bins = sacc_adapter.get_bin_edges(
            self.survey_name, self.cluster_properties
        )
        for bin_edge in self.bins:
            if bin_edge.dimension != self.bins[0].dimension:
                raise ValueError(
                    "The cluster number counts statistic requires all bins to be the "
                    "same dimension."
                )

        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        """Gets the statistic data vector.

        :return: The statistic data vector.
        """
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, concrete implementation.

        :param tools: The modeling tools used to compute the statistic.
        :return: The computed statistic.
        """
        assert tools.cluster_abundance is not None

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
        assert tools.cluster_abundance is not None

        mean_values = []
        for this_bin, counts in zip(self.bins, cluster_counts):
            total_observable = self.cluster_recipe.evaluate_theory_prediction(
                tools.cluster_abundance, this_bin, self.sky_area, cluster_properties
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
        assert tools.cluster_abundance is not None

        cluster_counts = []
        for this_bin in self.bins:
            counts = self.cluster_recipe.evaluate_theory_prediction(
                tools.cluster_abundance, this_bin, self.sky_area
            )
            cluster_counts.append(counts)

        return cluster_counts
