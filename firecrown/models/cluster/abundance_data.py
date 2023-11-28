"""The module responsible for extracting cluster data from a sacc file.
"""
from typing import Tuple, List
import numpy as np
import numpy.typing as npt
import sacc
from sacc.tracers import SurveyTracer
from firecrown.models.cluster.properties import ClusterProperty


class AbundanceData:
    """The class used to wrap a sacc file and return the cluster abundance data.

    The sacc file is a complicated set of tracers and surveys.  This class
    manipulates that data and returns only the data relevant for the cluster
    number count statistic.  The data in this class is specific to a single
    survey name."""

    # Hard coded in SACC, how do we want to handle this?
    _survey_index = 0
    _redshift_index = 1
    _mass_index = 2

    def __init__(
        self,
        sacc_data: sacc.Sacc,
        survey_nm: str,
        properties: ClusterProperty,
    ):
        self.sacc_data = sacc_data
        self.properties = properties
        try:
            self.survey_tracer: SurveyTracer = sacc_data.get_tracer(survey_nm)
            self.survey_nm = survey_nm
        except KeyError as exc:
            raise ValueError(
                f"The SACC file does not contain the SurveyTracer " f"{survey_nm}."
            ) from exc
        if not isinstance(self.survey_tracer, SurveyTracer):
            raise ValueError(f"The SACC tracer {survey_nm} is not a SurveyTracer.")

    def get_filtered_tracers(self, data_type: str) -> Tuple[npt.NDArray, npt.NDArray]:
        """Returns only tracers that match the data type requested."""
        all_tracers = np.array(
            self.sacc_data.get_tracer_combinations(data_type=data_type)
        )
        self.validate_tracers(all_tracers, data_type)
        survey_mask = all_tracers[:, self._survey_index] == self.survey_nm
        filtered_tracers = all_tracers[survey_mask]
        return filtered_tracers, survey_mask

    def get_data_and_indices(self, data_type: str) -> Tuple[List[float], List[int]]:
        """Returns the data vector and indices for the requested data type."""
        _, survey_mask = self.get_filtered_tracers(data_type)
        data_vector_list = list(
            self.sacc_data.get_mean(data_type=data_type)[survey_mask]
        )
        sacc_indices_list = list(
            self.sacc_data.indices(data_type=data_type)[survey_mask]
        )
        return data_vector_list, sacc_indices_list

    def validate_tracers(
        self, tracers_combinations: npt.NDArray, data_type: str
    ) -> None:
        """Validates that the tracers requested exist and are valid."""
        if len(tracers_combinations) == 0:
            raise ValueError(
                f"The SACC file does not contain any tracers for the "
                f"{data_type} data type."
            )

        if tracers_combinations.shape[1] != 3:
            raise ValueError(
                "The SACC file must contain 3 tracers for the "
                "cluster_counts data type: cluster_survey, "
                "redshift argument and mass argument tracers."
            )

    def get_bin_limits(self, data_type: str) -> List[List[Tuple[float, float]]]:
        """Returns the limits for all z, mass bins for the requested data type."""
        filtered_tracers, _ = self.get_filtered_tracers(data_type)

        tracers = []
        for _, z_tracer, mass_tracer in filtered_tracers:
            z_data: sacc.BaseTracer = self.sacc_data.get_tracer(z_tracer)
            mass_data: sacc.BaseTracer = self.sacc_data.get_tracer(mass_tracer)
            tracers.append(
                [(z_data.lower, z_data.upper), (mass_data.lower, mass_data.upper)]
            )

        return tracers
