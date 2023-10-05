import sacc
import numpy as np
from sacc.tracers import SurveyTracer


class SaccAdapter:
    # Hard coded in SACC, how do we want to handle this?
    _survey_index = 0
    _mass_index = 1
    _redshift_index = 2

    def __init__(self, sacc_data: sacc.Sacc):
        self.sacc_data = sacc_data

    def get_data_and_indices(self, data_type, survey_tracer):
        tracers_combinations = np.array(
            self.sacc_data.get_tracer_combinations(data_type=data_type)
        )
        self.validate_tracers(tracers_combinations, survey_tracer, data_type)

        cluster_survey_tracers = tracers_combinations[:, self._survey_index]

        self.z_tracers = tracers_combinations[:, self._redshift_index]
        self.mass_tracers = tracers_combinations[:, self._mass_index]
        self.survey_tracer = cluster_survey_tracers == survey_tracer

        data_vector_list = list(
            self.sacc_data.get_mean(data_type=data_type)[self.survey_tracer]
        )
        sacc_indices_list = list(
            self.sacc_data.indices(data_type=data_type)[self.survey_tracer]
        )
        return data_vector_list, sacc_indices_list

    def validate_tracers(self, tracers_combinations, survey_tracer, data_type):
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
        cluster_survey_tracers = tracers_combinations[:, self._survey_index]
        if survey_tracer not in cluster_survey_tracers:
            raise ValueError(
                f"The SACC tracer {self.survey_tracer} is not "
                f"present in the SACC file."
            )

    def get_z_bins(self):
        z_bins = {}
        for z_tracer_nm in self.z_tracers:
            tracer_data = self.sacc_data.get_tracer(z_tracer_nm)
            z_bins[z_tracer_nm] = (tracer_data.lower, tracer_data.upper)

        return z_bins

    def get_mass_bins(self):
        mass_bins = {}
        for mass_tracer_nm in self.mass_tracers:
            tracer_data = self.sacc_data.get_tracer(mass_tracer_nm)
            mass_bins[mass_tracer_nm] = (tracer_data.lower, tracer_data.upper)

        return mass_bins
