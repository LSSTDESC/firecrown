from __future__ import annotations
from typing import List, Optional
import sacc

from firecrown.models.cluster.abundance_data import AbundanceData
from .statistic import Statistic, DataVector, TheoryVector
from .source.source import SourceSystematic
from ....modeling_tools import ModelingTools
import numpy as np

import cProfile

# from pstats import SortKey


class ClusterNumberCounts(Statistic):
    def __init__(
        self,
        cluster_counts: bool,
        mean_log_mass: bool,
        survey_name: str,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        self.pr = cProfile.Profile()
        super().__init__()
        self.systematics = systematics or []
        self.theory_vector: Optional[TheoryVector] = None
        self.use_cluster_counts = cluster_counts
        self.use_mean_log_mass = mean_log_mass
        self.survey_name = survey_name
        self.data_vector = DataVector.from_list([])

    def read(self, sacc_data: sacc.Sacc):
        # Build the data vector and indices needed for the likelihood

        data_vector = []
        sacc_indices = []

        sacc_types = sacc.data_types.standard_types
        sacc_adapter = AbundanceData(
            sacc_data, self.survey_name, self.use_cluster_counts, self.use_mean_log_mass
        )

        if self.use_cluster_counts:
            data, indices = sacc_adapter.get_data_and_indices(sacc_types.cluster_counts)
            data_vector += data
            sacc_indices += indices

        if self.use_mean_log_mass:
            data, indices = sacc_adapter.get_data_and_indices(
                sacc_types.cluster_mean_log_mass
            )
            data_vector += data
            sacc_indices += indices

        self.sky_area = sacc_adapter.survey_tracer.sky_area
        # Note - this is the same for both cl mass and cl counts... Why do we need to
        # specify a data type?
        self.bin_limits = sacc_adapter.get_bin_limits(sacc_types.cluster_mean_log_mass)
        self.data_vector = DataVector.from_list(data_vector)
        print(len(data_vector))
        self.sacc_indices = np.array(sacc_indices)
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        theory_vector_list = []
        cluster_counts = []
        cluster_masses = []

        if self.use_cluster_counts or self.use_mean_log_mass:
            for z_proxy_limits, mass_proxy_limits in self.bin_limits:
                counts = tools.cluster_abundance.compute_counts(
                    z_proxy_limits, mass_proxy_limits
                )
                cluster_counts.append(counts)
            theory_vector_list += cluster_counts

        if self.use_mean_log_mass:
            for (z_proxy_limits, mass_proxy_limits), counts in zip(
                self.bin_limits, cluster_counts
            ):
                cluster_mass = (
                    tools.cluster_abundance.compute_mass(
                        z_proxy_limits, mass_proxy_limits
                    )
                    / counts
                )
                cluster_masses.append(cluster_mass)
            theory_vector_list += cluster_masses

        print(len(theory_vector_list))
        return TheoryVector.from_list(theory_vector_list)
