import sacc
import numpy as np
from sacc.tracers import SurveyTracer


class SaccAdapter:
    # Hard coded in SACC, how do we want to handle this?
    _survey_index = 0
    _mass_index = 1
    _redshift_index = 2

    def __init__(
        self, sacc_data: sacc.Sacc, survey_nm: str, cluster_counts, mean_log_mass
    ):
        self.sacc_data = sacc_data
        self.cluster_counts = cluster_counts
        self.mean_log_mass = mean_log_mass
        try:
            self.survey_tracer: SurveyTracer = sacc_data.get_tracer(survey_nm)
        except KeyError as exc:
            raise ValueError(
                f"The SACC file does not contain the SurveyTracer " f"{survey_nm}."
            ) from exc
        if not isinstance(self.survey_tracer, SurveyTracer):
            raise ValueError(f"The SACC tracer {survey_nm} is not a SurveyTracer.")

    def filter_tracers(self, data_type):
        tracers_combinations = np.array(
            self.sacc_data.get_tracer_combinations(data_type=data_type)
        )
        self.validate_tracers(tracers_combinations, data_type)

        self.z_tracers = np.unique(tracers_combinations[:, self._redshift_index])
        self.mass_tracers = np.unique(tracers_combinations[:, self._mass_index])
        self.survey_z_mass_tracers = tracers_combinations[self.survey_tracer]

    def get_data_and_indices(self, data_type):
        self.filter_tracers(data_type)
        data_vector_list = list(
            self.sacc_data.get_mean(data_type=data_type)[self.survey_tracer]
        )
        sacc_indices_list = list(
            self.sacc_data.indices(data_type=data_type)[self.survey_tracer]
        )
        return data_vector_list, sacc_indices_list

    def validate_tracers(self, tracers_combinations, data_type):
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

    def get_tracer_bounds(self, data_type):
        self.filter_tracers(data_type)

        z_bounds = {}
        for z_tracer_nm in self.z_tracers:
            tracer_data = self.sacc_data.get_tracer(z_tracer_nm)
            z_bounds[z_tracer_nm] = (tracer_data.lower, tracer_data.upper)

        mass_bounds = {}
        for mass_tracer_nm in self.mass_tracers:
            tracer_data = self.sacc_data.get_tracer(mass_tracer_nm)
            mass_bounds[mass_tracer_nm] = (tracer_data.lower, tracer_data.upper)

        tracer_bounds = [
            (z_bounds[z_tracer], mass_bounds[mass_tracer])
            for _, z_tracer, mass_tracer in self.survey_z_mass_tracers
        ]
        return tracer_bounds
