"""This module holds classes needed to predict the binned cluster shear profile.

The binned cluster shear profile statistic predicts the excess density
surface mass of clusters within a single redshift and mass bin.
"""

from __future__ import annotations

import os
import sys

# firecrown is needed for backward compatibility; remove support for deprecated
# directory structure is removed.
import firecrown  # pylint: disable=unused-import # noqa: F401
import sacc
from firecrown.likelihood.source import SourceSystematic
from firecrown.likelihood.statistic import TheoryVector
from firecrown.modeling_tools import ModelingTools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firecrown.models.cluster import (
    ClusterProperty,
    DeltaSigmaData,
)

from crow.recipes.murata_binned_spec_z import (
    MurataBinnedSpecZRecipe,
)

from .binned_cluster import BinnedCluster
from .updatable_wrapper import UpdatableClusterObjects


class BinnedClusterShearProfile(BinnedCluster):
    """The Binned Cluster Delta Sigma statistic.

    This class will make a prediction for the deltasigma of clusters in a z, mass,
    radial bin and compare that prediction to the data provided in the sacc file.
    """

    def __init__(
        self,
        cluster_properties: ClusterProperty,
        survey_name: str,
        cluster_recipe: MurataBinnedSpecZRecipe,
        systematics: None | list[SourceSystematic] = None,
    ):
        """Initialize this statistic.

        :param cluster_properties: The cluster observables to use.
        :param survey_name: The name of the survey to use.
        :param cluster_recipe: The cluster recipe to use.
        :param systematics: The systematics to apply to this statistic.
        """
        super().__init__(cluster_properties, survey_name, cluster_recipe, systematics)

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
                    "parameters": ["cluster_concentration"],
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
        cluster_data = DeltaSigmaData(sacc_data)
        self._read(cluster_data)

        super().read(sacc_data)

    def _compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a statistic from sources, concrete implementation."""
        theory_vector_list: list[float] = []
        self.updatable_parameters.export_all_parameters(
            self.cluster_recipe, tools.get_ccl_cosmology()
        )

        for cl_property in ClusterProperty:
            include_prop = cl_property & self.cluster_properties
            if not include_prop:
                continue
            if cl_property == ClusterProperty.DELTASIGMA:
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
        mean_values = []
        mass_edges = None
        z_edges = None
        counts = 1.0
        for this_bin in self.bins:
            if mass_edges != this_bin.mass_proxy_edges or z_edges != this_bin.z_edges:
                mass_edges = this_bin.mass_proxy_edges
                z_edges = this_bin.z_edges
                counts = self.cluster_recipe.evaluate_theory_prediction_counts(
                    this_bin.z_edges,
                    this_bin.mass_proxy_edges,
                    self.sky_area,
                )
            total_observable = (
                self.cluster_recipe.evaluate_theory_prediction_shear_profile(
                    this_bin.z_edges,
                    this_bin.mass_proxy_edges,
                    this_bin.radius_center,
                    self.sky_area,
                    cluster_properties,
                )
            )
            mean_observable = total_observable / counts
            mean_values.append(mean_observable)
        return mean_values
