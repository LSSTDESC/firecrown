"""This module holds classes needed to predict the binned cluster number counts

The binned cluster number counts statistic predicts the number of galaxy
clusters within a single redshift and mass bin.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import sacc

from firecrown.likelihood.gauss_family.statistic.source.source import SourceSystematic
from firecrown.likelihood.gauss_family.statistic.statistic import (
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
    """The Binned Cluster Number Counts statistic

    This class will make a prediction for the number of clusters in a z, mass bin
    and compare that prediction to the data provided in the sacc file.
    """

    def __init__(
        self,
        cluster_properties: ClusterProperty,
        survey_name: str,
        cluster_recipe: ClusterRecipe,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        super().__init__()
        self.systematics = systematics or []
        self.theory_vector: Optional[TheoryVector] = None
        self.cluster_properties = cluster_properties
        self.survey_name = survey_name
        self.cluster_recipe = cluster_recipe
        self.data_vector = DataVector.from_list([])
        self.sky_area = 0.0
        self.bins: List[SaccBin] = []

    def read(self, sacc_data: sacc.Sacc) -> None:
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
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        assert tools.cluster_abundance is not None

        theory_vector_list: List[float] = []
        cluster_counts = []

        cluster_counts = self.get_binned_cluster_counts(tools)

        for cl_property in ClusterProperty:
            include_prop = cl_property & self.cluster_properties
            if not include_prop:
                continue

            if cl_property == ClusterProperty.COUNTS:
                theory_vector_list += cluster_counts
                continue

            theory_vector_list += self.get_binned_cluster_property(
                tools, cluster_counts, cl_property
            )

        return TheoryVector.from_list(theory_vector_list)

    def get_binned_cluster_property(
        self,
        tools: ModelingTools,
        cluster_counts: List[float],
        cluster_properties: ClusterProperty,
    ) -> List[float]:
        """Computes the mean mass of clusters in each bin

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted mean mass of
        the clusters in each bin."""
        assert tools.cluster_abundance is not None

        mean_values = []
        for this_bin, counts in zip(self.bins, cluster_counts):
            total_observable = self.cluster_recipe.evaluate_theory_prediction(
                tools.cluster_abundance, this_bin, self.sky_area, cluster_properties
            )
            cluster_counts.append(counts)

            mean_observable = total_observable / counts
            mean_values.append(mean_observable)

        return mean_values

    def get_binned_cluster_counts(self, tools: ModelingTools) -> List[float]:
        """Computes the number of clusters in each bin

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted number of
        clusters in each bin."""
        assert tools.cluster_abundance is not None

        cluster_counts = []
        for this_bin in self.bins:
            counts = self.cluster_recipe.evaluate_theory_prediction(
                tools.cluster_abundance, this_bin, self.sky_area
            )
            cluster_counts.append(counts)

        return cluster_counts
