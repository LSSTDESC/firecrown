from __future__ import annotations
from typing import List, Optional
import sacc

from firecrown.models.sacc_adapter import SaccAdapter
from .statistic import Statistic, DataVector, TheoryVector
from .source.source import SourceSystematic
from ....modeling_tools import ModelingTools
import pdb
import numpy as np


class ClusterNumberCounts(Statistic):
    def __init__(
        self,
        cluster_counts: bool,
        mean_log_mass: bool,
        survey_name: str,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
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
        bin_limits = []
        sacc_types = sacc.data_types.standard_types
        sacc_adapter = SaccAdapter(
            sacc_data, self.survey_name, self.use_cluster_counts, self.use_mean_log_mass
        )

        if self.use_cluster_counts:
            data, indices = sacc_adapter.get_data_and_indices(sacc_types.cluster_counts)
            bin_limits += sacc_adapter.get_bin_limits(sacc_types.cluster_counts)
            data_vector += data
            sacc_indices += indices

        if self.use_mean_log_mass:
            data, indices = sacc_adapter.get_data_and_indices(
                sacc_types.cluster_mean_log_mass
            )
            bin_limits += sacc_adapter.get_bin_limits(sacc_types.cluster_mean_log_mass)
            data_vector += data
            sacc_indices += indices
        self.sky_area = sacc_adapter.survey_tracer.sky_area
        self.bin_limits = bin_limits
        self.data_vector = DataVector.from_list(data_vector)
        self.sacc_indices = np.array(sacc_indices)
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        tools.cluster_abundance.sky_area = self.sky_area

        theory_vector_list = []
        cluster_counts_list = []

        if self.use_cluster_counts:
            for z_proxy_limits, mass_proxy_limits in self.bin_limits:
                cluster_counts = tools.cluster_abundance.compute(
                    z_proxy_limits, mass_proxy_limits
                )
                cluster_counts_list.append(cluster_counts)
            theory_vector_list += cluster_counts_list
            print(theory_vector_list)

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
