import sacc
import numpy as np
from sacc.tracers import SurveyTracer
from typing import Tuple, List
import numpy.typing as npt


class AbundanceData:
    # Hard coded in SACC, how do we want to handle this?
    _survey_index = 0
    _redshift_index = 1
    _mass_index = 2

    def __init__(
        self,
        sacc_data: sacc.Sacc,
        survey_nm: str,
        cluster_counts: bool,
        mean_log_mass: bool,
    ):
        self.sacc_data = sacc_data
        self.cluster_counts = cluster_counts
        self.mean_log_mass = mean_log_mass
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
        all_tracers = np.array(
            self.sacc_data.get_tracer_combinations(data_type=data_type)
        )
        self.validate_tracers(all_tracers, data_type)
        survey_mask = all_tracers[:, self._survey_index] == self.survey_nm
        filtered_tracers = all_tracers[survey_mask]
        return filtered_tracers, survey_mask

    def get_data_and_indices(self, data_type: str) -> Tuple[List[float], List[int]]:
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
        filtered_tracers, _ = self.get_filtered_tracers(data_type)

        tracers = []
        for _, z_tracer, mass_tracer in filtered_tracers:
            z_data: sacc.BaseTracer = self.sacc_data.get_tracer(z_tracer)
            mass_data: sacc.BaseTracer = self.sacc_data.get_tracer(mass_tracer)
            tracers.append(
                [(z_data.lower, z_data.upper), (mass_data.lower, mass_data.upper)]
            )

        return tracers
