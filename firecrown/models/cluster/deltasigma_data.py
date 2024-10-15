"""The module responsible for extracting cluster data from a sacc file."""

import numpy as np
import numpy.typing as npt
import sacc
from sacc.tracers import SurveyTracer
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.binning import SaccBin


class DeltaSigmaData:
    """The class used to wrap a sacc file and return the cluster deltasigma data.

    The sacc file is a complicated set of tracers (bins) and surveys.  This class
    manipulates that data and returns only the data relevant for the cluster
    number count statistic.  The data in this class is specific to a single
    survey name.
    """

    _survey_index = 0
    _redshift_index = 1
    _mass_index = 2
    _radius_index = 3

    def __init__(self, sacc_data: sacc.Sacc):
        self.sacc_data = sacc_data

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
    ) -> tuple[list[float], list[int]]:
        """Returns the observed data for the specified survey and properties.

        For example if the caller has enabled DELTASIGMA then the observed
        cluster profile within each N dimensional bin will be returned.
        """
        data_vectors = []
        sacc_indices = []
        data_type = None
        for cluster_property in ClusterProperty:
            include_prop = cluster_property & properties
            if include_prop == ClusterProperty.DELTASIGMA:
                # pylint: disable=no-member
                data_type = sacc.standard_types.cluster_shear
        if data_type is None:
            # pylint: disable=no-member
            raise ValueError(
                f"The SACC file does not contain the"
                f"{sacc.standard_types.cluster_shear} data type"
            )

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

        if bins_combos_for_type.shape[1] != 4:
            raise ValueError(
                "The SACC file must contain 4 tracers for the "
                "cluster_deltasigma data type: cluster_survey, "
                "redshift argument, mass argument and radius argument tracers."
            )

        return bins_combos_for_type

    def get_bin_edges(
        self, survey_nm: str, properties: ClusterProperty
    ) -> list[SaccBin]:
        """Returns the limits for all z, mass bins for the requested data type."""
        bins = []
        data_type = None
        for cluster_property in ClusterProperty:
            if not cluster_property & properties:
                continue
            if cluster_property == ClusterProperty.DELTASIGMA:
                # pylint: disable=no-member
                data_type = sacc.standard_types.cluster_shear
        if data_type is None:
            # pylint: disable=no-member
            raise ValueError(
                f"The SACC file does not contain the"
                f" {sacc.standard_types.cluster_shear} data type"
            )

        bin_combinations = self._all_bin_combinations_for_data_type(data_type)
        my_survey_mask = bin_combinations[:, self._survey_index] == survey_nm
        bin_combinations = bin_combinations[my_survey_mask]

        for _, z_tracer, mass_tracer, radius_tracer in bin_combinations:
            z_data: sacc.tracers.BinZTracer = self.sacc_data.get_tracer(z_tracer)
            mass_data: sacc.tracers.BinRichnessTracer = self.sacc_data.get_tracer(
                mass_tracer
            )
            radius_data: sacc.tracers.BinRadiusTracer = self.sacc_data.get_tracer(
                radius_tracer
            )

            sacc_bin = SaccBin([z_data, mass_data, radius_data])
            bins.append(sacc_bin)

        # Remove duplicates while preserving order (i.e. dont use set())
        return list(dict.fromkeys(bins))
