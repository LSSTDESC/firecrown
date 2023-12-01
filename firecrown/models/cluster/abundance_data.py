"""The module responsible for extracting cluster data from a sacc file.
"""
from typing import Tuple, List
import numpy as np
import numpy.typing as npt
import sacc
from sacc.tracers import SurveyTracer
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.binning import SaccBin


class AbundanceData:
    """The class used to wrap a sacc file and return the cluster abundance data.

    The sacc file is a complicated set of tracers (bins) and surveys.  This class
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
        bin_dimensions: int,
    ):
        self.sacc_data = sacc_data
        self.bin_dimensions = bin_dimensions

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

    def get_observed_data_and_indices_by_survey(
        self,
        survey_nm: str,
        properties: ClusterProperty,
    ) -> Tuple[List[float], List[int]]:
        """Will return observed data (and sacc indices) for a specified survey based on
        the properties requested by the caller.

        For example if the caller has enabled COUNTS then the observed cluster counts
        within each N dimensional bin will be returned."""

        data_vectors = []
        sacc_indices = []

        for cluster_property in ClusterProperty:
            if not cluster_property & properties:
                continue

            if cluster_property == ClusterProperty.COUNTS:
                # pylint: disable=no-member
                data_type = sacc.standard_types.cluster_counts
            elif cluster_property == ClusterProperty.MASS:
                # pylint: disable=no-member
                data_type = sacc.standard_types.cluster_mean_log_mass
            else:
                raise NotImplementedError(cluster_property)

            bin_combinations = self._all_bin_combinations_for_data_type(data_type)

            my_survey_mask = bin_combinations[:, self._survey_index] == survey_nm

            data_vectors += list(
                self.sacc_data.get_mean(data_type=data_type)[my_survey_mask]
            )

            sacc_indices += list(
                self.sacc_data.indices(data_type=data_type)[my_survey_mask]
            )

        return data_vectors, sacc_indices

    def _all_bin_combinations_for_data_type(self, data_type: str) -> npt.NDArray:
        bins_combos_for_type = np.array(
            self.sacc_data.get_tracer_combinations(data_type=data_type)
        )

        if len(bins_combos_for_type) == 0:
            raise ValueError(
                f"The SACC file does not contain any tracers for the "
                f"{data_type} data type."
            )

        # Add one because we don't include survey in the bin dimensions (for now)
        if bins_combos_for_type.shape[1] != self.bin_dimensions + 1:
            raise ValueError(
                "The SACC file must contain 3 tracers for the "
                "cluster_counts data type: cluster_survey, "
                "redshift argument and mass argument tracers."
            )

        return bins_combos_for_type

    def get_bin_edges(
        self, survey_nm: str, properties: ClusterProperty
    ) -> List[SaccBin]:
        """Returns the limits for all z, mass bins for the requested data type."""
        bins = []

        for cluster_property in ClusterProperty:
            if not cluster_property & properties:
                continue

            if cluster_property == ClusterProperty.COUNTS:
                # pylint: disable=no-member
                data_type = sacc.standard_types.cluster_counts
            elif cluster_property == ClusterProperty.MASS:
                # pylint: disable=no-member
                data_type = sacc.standard_types.cluster_mean_log_mass
            else:
                raise NotImplementedError(cluster_property)

            bin_combinations = self._all_bin_combinations_for_data_type(data_type)
            my_survey_mask = bin_combinations[:, self._survey_index] == survey_nm
            bin_combinations_for_survey = bin_combinations[my_survey_mask]

            for _, z_tracer, mass_tracer in bin_combinations_for_survey:
                z_data: sacc.BaseTracer = self.sacc_data.get_tracer(z_tracer)
                mass_data: sacc.BaseTracer = self.sacc_data.get_tracer(mass_tracer)
                sacc_bin = SaccBin([z_data, mass_data], self.bin_dimensions)
                bins.append(sacc_bin)

        # Remove duplicates while preserving order (i.e. dont use set())
        unique_bins = list(dict.fromkeys(bins))
        return unique_bins
