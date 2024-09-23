"""This module holds classes needed to predict the binned cluster number counts.

The binned cluster number counts statistic predicts the number of galaxy
clusters within a single redshift and mass bin.
"""

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
from firecrown.models.cluster.deltasigma_data import DeltaSigmaData
from firecrown.models.cluster.binning import SaccBin
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.recipes.cluster_recipe import ClusterRecipe


class BinnedClusterDeltaSigma(Statistic):
    """The Binned Cluster Delta Sigma statistic.

    This class will make a prediction for the deltasigma of clusters in a z, mass, radial bin
    and compare that prediction to the data provided in the sacc file.
    """

    def __init__(
        self,
        cluster_properties: ClusterProperty,
        survey_name: str,
        cluster_recipe: ClusterRecipe,
        systematics: None | list[SourceSystematic] = None,
    ):
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
        """Read the data for this statistic and mark it as ready for use."""
        # Build the data vector and indices needed for the likelihood
        if self.cluster_properties == ClusterProperty.NONE:
            raise ValueError("You must specify at least one cluster property.")

        sacc_adapter = DeltaSigmaData(sacc_data)
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
                    "The cluster deltasigma statistic requrires all bins to be the "
                    "same dimension."
                )
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        """Gets the statistic data vector."""
        assert self.data_vector is not None
        return self.data_vector

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, concrete implementation."""
        assert tools.cluster_abundance is not None
        assert tools.cluster_deltasigma is not None
        theory_vector_list: list[float] = []
        cluster_counts = []
        # cluster_counts = self.get_binned_cluster_counts(tools)

        for cl_property in ClusterProperty:
            include_prop = cl_property & self.cluster_properties
            if not include_prop:
                continue
            if cl_property == ClusterProperty.DELTASIGMA:
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
        """Computes the mean deltasigma of clusters in each bin.

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted mean deltasigma of
        the clusters in each bin.
        """
        assert tools.cluster_abundance is not None
        mean_values = []
        mass_edges = None
        z_edges = None
        for this_bin in self.bins:
            if mass_edges != this_bin.mass_proxy_edges or z_edges != this_bin.z_edges:
                mass_edges = this_bin.mass_proxy_edges
                z_edges = this_bin.z_edges
                counts = self.cluster_recipe.evaluate_theory_prediction_counts(
                    tools.cluster_abundance, this_bin, self.sky_area, cluster_properties
                )
            total_observable = self.cluster_recipe.evaluate_theory_prediction(
                tools.cluster_abundance,
                tools.cluster_deltasigma,
                this_bin,
                self.sky_area,
                cluster_properties,
            )
            mean_observable = total_observable / counts
            mean_values.append(mean_observable)
        return mean_values
