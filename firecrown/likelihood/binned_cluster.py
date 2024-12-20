"""Binned cluster statistic support."""

from __future__ import annotations

import numpy as np

# firecrown is needed for backward compatibility; remove support for deprecated
# directory structure is removed.
import firecrown  # pylint: disable=unused-import # noqa: F401
from firecrown.likelihood.source import SourceSystematic
from firecrown.likelihood.statistic import (
    Statistic,
)
from firecrown.data_types import TheoryVector, DataVector
from firecrown.models.cluster.cluster_data import ClusterData
from firecrown.models.cluster.binning import SaccBin
from firecrown.models.cluster.properties import ClusterProperty


class BinnedCluster(Statistic):
    """A statistic representing clusters in a z, mass bin."""

    def __init__(
        self,
        cluster_properties: ClusterProperty,
        survey_name: str,
        cluster_recipe,
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

    def _read(self, cluster_data: ClusterData) -> None:
        sacc_adapter = cluster_data
        self.sky_area = sacc_adapter.get_survey_tracer(self.survey_name).sky_area

        data, indices = sacc_adapter.get_observed_data_and_indices_by_survey(
            self.survey_name, self.cluster_properties
        )
        self.data_vector = DataVector.from_list(data)
        self.sacc_indices = np.array(indices)

        self.bins = sacc_adapter.get_bin_edges(
            self.survey_name, self.cluster_properties
        )

    def get_data_vector(self) -> DataVector:
        """Gets the statistic data vector.

        :return: The statistic data vector.
        """
        assert self.data_vector is not None
        return self.data_vector
