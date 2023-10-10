from __future__ import annotations
from typing import List, Optional
import sacc
from .statistic import Statistic, DataVector, TheoryVector
from .source.source import SourceSystematic
from ....modeling_tools import ModelingTools


class ClusterNumberCounts(Statistic):
    def __init__(
        self,
        cluster_counts: bool,
        mean_log_mass: bool,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        super().__init__()
        self.systematics = systematics or []
        self.theory_vector: Optional[TheoryVector] = None
        self.use_cluster_counts = cluster_counts
        self.use_mean_log_mass = mean_log_mass
        self.sky_area = 0.0
        self.data_vector = DataVector.from_list([])

    def read(self, sacc_data: sacc.Sacc):
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        tools.cluster_abundance.sky_area = self.sky_area
        ccl_cosmo = tools.get_ccl_cosmology()

        theory_vector_list = []
        cluster_counts_list = []

        if self.use_cluster_counts:
            cluster_counts_list = tools.cluster_abundance.compute()
            theory_vector_list += cluster_counts_list

        # if self.use_mean_log_mass:
        #     mean_log_mass_list = [
        #         self.cluster_abundance.compute_unormalized_mean_logM(
        #             ccl_cosmo, logM_tracer_arg, z_tracer_arg
        #         )
        #         / counts
        #         for (z_tracer_arg, logM_tracer_arg), counts in zip(
        #             self.tracer_args, cluster_counts_list
        #         )
        #     ]
        #     theory_vector_list += mean_log_mass_list
        return TheoryVector.from_list(theory_vector_list)
