from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy as np

import sacc
from sacc.tracers import SurveyTracer

from .statistic import Statistic, DataVector, TheoryVector
from .source.source import SourceSystematic
from ....models.cluster_abundance import ClusterAbundance
from ....modeling_tools import ModelingTools
from ....models.sacc_adapter import SaccAdapter


class ClusterNumberCounts(Statistic):
    def __init__(
        self,
        survey_name: str,
        cluster_counts: bool,
        mean_log_mass: bool,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        super().__init__()
        self.systematics = systematics or []
        self.theory_vector: Optional[TheoryVector] = None
        self.survey_nm = survey_name
        self.use_cluster_counts = cluster_counts
        self.use_mean_log_mass = mean_log_mass

    def read(self, sacc_data: sacc.Sacc):
        sacc_types = sacc.data_types.standard_types
        sa = SaccAdapter(
            sacc_data, self.survey_nm, self.use_cluster_counts, self.use_mean_log_mass
        )

        data_vector_list = []
        sacc_indices_list = []
        tracer_bounds = []

        if self.use_cluster_counts:
            dtype = sacc_types.cluster_counts
            data, indices = sa.get_data_and_indices(dtype)
            data_vector_list += data
            sacc_indices_list += indices
            tracer_bounds += sa.get_tracer_bounds(dtype)

        if self.use_mean_log_mass:
            dtype = sacc_types.cluster_mean_log_mass
            data, indices = sa.get_data_and_indices(dtype, survey_tracer)
            data_vector_list += data
            sacc_indices_list += indices
            tracer_bounds += sa.get_tracer_bounds(dtype, survey_tracer)

        self.sky_area = sa.survey_tracer.sky_area
        self.data_vector = DataVector.from_list(data_vector_list)
        self.sacc_indices = np.array(sacc_indices_list)
        self.tracer_bounds = np.array(tracer_bounds)
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
            for (z_min, z_max), (mproxy_min, mproxy_max) in self.tracer_bounds:
                integrand = tools.cluster_abundance.get_abundance_integrand(
                    mass_min, z_min, mass_max, z_max
                )

            theory_vector_list += cluster_counts_list

        if self.use_mean_log_mass:
            mean_log_mass_list = [
                self.cluster_abundance.compute_unormalized_mean_logM(
                    ccl_cosmo, logM_tracer_arg, z_tracer_arg
                )
                / counts
                for (z_tracer_arg, logM_tracer_arg), counts in zip(
                    self.tracer_args, cluster_counts_list
                )
            ]
            theory_vector_list += mean_log_mass_list
        return TheoryVector.from_list(theory_vector_list)
