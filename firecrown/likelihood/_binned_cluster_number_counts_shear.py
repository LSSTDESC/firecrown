"""This module holds classes needed to predict the binned cluster shear profile.

The binned cluster shear profile statistic predicts the excess density
surface mass of clusters within a single redshift and mass bin.
"""

from __future__ import annotations
from typing import Any

import sacc
import numpy as np
from crow.properties import ClusterProperty

from firecrown.data_types import TheoryVector
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster import ShearData
from firecrown.likelihood._binned_cluster import BinnedCluster


class BinnedClusterShearProfile(BinnedCluster):
    """The Binned Cluster Delta Sigma statistic.

    This class will make a prediction for the shear of clusters in a z, mass,
    radial bin and compare that prediction to the data provided in the sacc file.
    """

    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this statistic and mark it as ready for use.

        :param sacc_data: The data in the sacc format.
        """
        # Build the data vector and indices needed for the likelihood
        if self.cluster_properties == ClusterProperty.NONE:
            raise ValueError("You must specify at least one cluster property.")
        cluster_data = ShearData(sacc_data)
        self._read(cluster_data)

        super().read(sacc_data)

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, concrete implementation."""
        theory_vector_list: list[float] = []
        self.updatable_parameters.export_all_parameters(
            self.cluster_recipe, tools.get_ccl_cosmology()
        )
        self.cluster_recipe.setup()
        for cl_property in ClusterProperty:
            include_prop = cl_property & self.cluster_properties
            if not include_prop:
                continue
            if cl_property == ClusterProperty.DELTASIGMA:
                theory_vector_list += self.get_binned_cluster_property(cl_property)
            elif cl_property == ClusterProperty.SHEAR:
                theory_vector_list += self.get_binned_cluster_property(cl_property)
        return TheoryVector.from_list(theory_vector_list)

    def get_binned_cluster_property(
        self,
        cluster_properties: ClusterProperty,
    ) -> list[float]:
        """Computes the mean deltasigma of clusters in each bin.

        Using the data from the sacc file, this function evaluates the likelihood for
        a single point of the parameter space, and returns the predicted
        mean deltasigma of the clusters in each bin.
        """

        grouped = self._group_bins_by_edges()

        results = {}
        # loop trough all the redshift proxy bins, but now we know all the radiuses
        for (z_edges, proxy_edges), entries in grouped.items():
            radius_list = [radius for (_, radius) in entries]
            radius_array = np.array(radius_list)

            counts = self.cluster_recipe.evaluate_theory_prediction_counts(
                z_edges, proxy_edges, self.sky_area
            )

            total_observable = (
                self.cluster_recipe.evaluate_theory_prediction_lensing_profile(
                    z_edges,
                    proxy_edges,
                    radius_array,
                    self.sky_area,
                    cluster_properties,
                )
            )
            mean_obs = (
                total_observable / counts
                if counts != 0.0
                else np.zeros_like(total_observable)
            )
            # store results indexed by radius bin index
            for (i_bin, _), value in zip(entries, mean_obs):
                results[i_bin] = value

        # 3. return in the original order of self.bins
        return [results[i] for i in range(len(self.bins))]

    def _group_bins_by_edges(self) -> dict[tuple[Any, Any], list[tuple[int, float]]]:
        """Group bins by (z_edges, mass_proxy_edges)."""
        grouped = {}  # type: dict[tuple, list[tuple[int, float]]]
        for i, b in enumerate(self.bins):
            key = (b.z_edges, b.mass_proxy_edges)
            grouped.setdefault(key, []).append((i, b.radius_center))
        return grouped
