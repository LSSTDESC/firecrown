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
    @property
    def use_cluster_counts(self) -> bool:
        return self._use_cluster_counts

    @property
    def use_mean_log_mass(self) -> bool:
        return self._use_mean_log_mass

    @use_cluster_counts.setter
    def use_cluster_counts(self, value: bool):
        self._use_cluster_counts = value

    @use_mean_log_mass.setter
    def use_mean_log_mass(self, value: bool):
        self._use_mean_log_mass = value

    def __init__(
        self,
        systematics: Optional[List[SourceSystematic]] = None,
        survey_nm: str = "numcosmo_simulated_redshift_richness",
    ):
        super().__init__()
        self.systematics = systematics or []
        self.theory_vector: Optional[TheoryVector] = None
        self.survey_nm = survey_nm

    def read(self, sacc_data: sacc.Sacc):
        try:
            survey_tracer: SurveyTracer = sacc_data.get_tracer(self.survey_nm)
        except KeyError as exc:
            raise ValueError(
                f"The SACC file does not contain the SurveyTracer " f"{self.survey_nm}."
            ) from exc
        if not isinstance(survey_tracer, SurveyTracer):
            raise ValueError(f"The SACC tracer {self.survey_nm} is not a SurveyTracer.")

        sacc_types = sacc.data_types.standard_types
        sa = SaccAdapter(sacc_data)

        data_vector_list = []
        sacc_indices_list = []

        if self.use_cluster_counts:
            dtype = sacc_types.cluster_counts
            data, indices = sa.get_data_and_indices(dtype, survey_tracer)
            data_vector_list += data
            sacc_indices_list += indices

        if self.use_mean_log_mass:
            dtype = sacc_types.cluster_mean_log_mass
            data, indices = sa.get_data_and_indices(dtype, survey_tracer)
            data_vector_list += data
            sacc_indices_list += indices

        self.sky_area = sa.survey_tracer.sky_area
        self.data_vector = DataVector.from_list(data_vector_list)
        self.sacc_indices = np.array(sacc_indices_list)
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        tools.cluster_abundance.sky_area = self.sky_area
        ccl_cosmo = tools.get_ccl_cosmology()
        # tools.cluster_abundance.etc()
        theory_vector_list = []
        cluster_counts_list = []

        if self.use_cluster_counts or self.use_mean_log_mass:
            cluster_counts_list = [
                self.cluster_abundance.compute(ccl_cosmo, logM_tracer_arg, z_tracer_arg)
                for z_tracer_arg, logM_tracer_arg in self.tracer_args
            ]
            if self.use_cluster_counts:
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
