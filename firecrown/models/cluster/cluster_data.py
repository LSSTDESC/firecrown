"""The module responsible for extracting cluster data from a sacc file."""

from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import sacc
from sacc.tracers import SurveyTracer
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.binning import SaccBin


class ClusterData:
    """The class used to wrap a sacc file and return the cluster abundance data.

    The sacc file is a complicated set of tracers (bins) and surveys.  This class
    manipulates that data and returns only the data relevant for the cluster
    number count statistic.  The data in this class is specific to a single
    survey name.
    """

    def __init__(self, sacc_data: sacc.Sacc):
        self.sacc_data = sacc_data
        self._survey_index = 0
        self._redshift_index = 1
        self._mass_index = 2

    def get_survey_tracer(self, survey_nm: str) -> SurveyTracer:
        """Returns the SurveyTracer for the specified survey name."""
        try:
            survey_tracer: SurveyTracer = self.sacc_data.get_tracer(survey_nm)
        except KeyError as exc:
            raise ValueError(
                f"The SACC file does not contain the SurveyTracer " f"{survey_nm}."
            ) from exc

        if not isinstance(survey_tracer, SurveyTracer):
            raise ValueError(f"The SACC tracer {survey_nm} is not a SurveyTracer.")

        return survey_tracer

    def _get_observed_data_and_indices_by_survey(
        self,
        survey_nm: str,
        data_types: list,
        tracers_n: int,
    ) -> tuple[list[float], list[int]]:
        """Returns the observed data for the specified survey and data types.

        For example if the caller has enabled COUNTS then the observed cluster counts
        within each N dimensional bin will be returned.
        """
        data_vectors = []
        sacc_indices = []

        for data_type in data_types:
            bin_combinations = self._all_bin_combinations_for_data_type(
                data_type, tracers_n
            )

            my_survey_mask = bin_combinations[:, self._survey_index] == survey_nm

            data_vectors += list(
                self.sacc_data.get_mean(data_type=data_type)[my_survey_mask]
            )

            sacc_indices += list(
                self.sacc_data.indices(data_type=data_type)[my_survey_mask]
            )

        return data_vectors, sacc_indices

    def _all_bin_combinations_for_data_type(
        self, data_type: str, tracers_n: int
    ) -> npt.NDArray:
        bins_combos_for_type = np.array(
            self.sacc_data.get_tracer_combinations(data_type=data_type)
        )

        if len(bins_combos_for_type) == 0:
            raise ValueError(
                f"The SACC file does not contain any tracers for the "
                f"{data_type} data type."
            )

        if bins_combos_for_type.shape[1] != tracers_n:
            raise ValueError(
                f"The SACC file must contain {tracers_n} tracers for the "
                f"{data_type} data type."
            )

        return bins_combos_for_type

    def _all_bin_combinations_for_data_type_and_survey(
        self, survey_nm: str, data_type: str, tracers_n: int
    ) -> npt.NDArray:

        bin_combinations = self._all_bin_combinations_for_data_type(
            data_type, tracers_n
        )
        my_survey_mask = bin_combinations[:, self._survey_index] == survey_nm
        bin_combinations_for_survey = bin_combinations[my_survey_mask]

        return bin_combinations_for_survey

    @abstractmethod
    def get_observed_data_and_indices_by_survey(
        self,
        survey_nm: str,
        properties: ClusterProperty,
    ) -> tuple[list[float], list[int]]:
        """Returns the observed data for the specified survey and properties.

        For example if the caller has enabled COUNTS then the observed cluster counts
        within each N dimensional bin will be returned.
        """

    @abstractmethod
    def get_bin_edges(
        self, survey_nm: str, properties: ClusterProperty
    ) -> list[SaccBin]:
        """Returns the limits for all z, mass bins for the requested data type."""
