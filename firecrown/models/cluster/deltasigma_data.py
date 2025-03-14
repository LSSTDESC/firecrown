"""The module responsible for extracting cluster data from a sacc file."""

import sacc
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.binning import SaccBin
from firecrown.models.cluster.cluster_data import ClusterData


class DeltaSigmaData(ClusterData):
    """The class used to wrap a sacc file and return the cluster deltasigma data.

    The sacc file is a complicated set of tracers (bins) and surveys.  This class
    manipulates that data and returns only the data relevant for the cluster
    number count statistic.  The data in this class is specific to a single
    survey name.
    """

    def __init__(self, sacc_data: sacc.Sacc):
        super().__init__(sacc_data)
        self._radius_index = 3

    def get_observed_data_and_indices_by_survey(
        self,
        survey_nm: str,
        properties: ClusterProperty,
    ) -> tuple[list[float], list[int]]:
        """Returns the observed data for the specified survey and properties.

        For example if the caller has enabled DELTASIGMA then the observed
        cluster profile within each N dimensional bin will be returned.
        """
        data_types = []
        for cluster_property in ClusterProperty:
            include_prop = cluster_property & properties
            if include_prop == ClusterProperty.DELTASIGMA:
                # pylint: disable=no-member
                data_types.append(sacc.standard_types.cluster_shear)
        if not data_types:
            # pylint: disable=no-member
            raise ValueError(
                f"The property should be related to the "
                f"{sacc.standard_types.cluster_shear} data type."
            )

        data_vectors, sacc_indices = self._get_observed_data_and_indices_by_survey(
            survey_nm, data_types, 4
        )

        return data_vectors, sacc_indices

    def get_bin_edges(
        self, survey_nm: str, properties: ClusterProperty
    ) -> list[SaccBin]:
        """Returns the limits for all z, mass bins for the shear data type."""
        bins = []
        data_type = None
        if ClusterProperty.DELTASIGMA not in properties:
            raise ValueError(f"The property must be {ClusterProperty.DELTASIGMA}.")
        # pylint: disable=no-member
        data_type = sacc.standard_types.cluster_shear
        bin_combinations_for_survey = (
            self._all_bin_combinations_for_data_type_and_survey(survey_nm, data_type, 4)
        )

        for _, z_tracer, mass_tracer, radius_tracer in bin_combinations_for_survey:
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
