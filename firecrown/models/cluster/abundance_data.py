"""The module responsible for extracting cluster data from a sacc file."""

import sacc
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.models.cluster.binning import SaccBin
from firecrown.models.cluster.cluster_data import ClusterData


class AbundanceData(ClusterData):
    """The class used to wrap a sacc file and return the cluster abundance data.

    The sacc file is a complicated set of tracers (bins) and surveys.  This class
    manipulates that data and returns only the data relevant for the cluster
    number count statistic.  The data in this class is specific to a single
    survey name.
    """

    def get_observed_data_and_indices_by_survey(
        self,
        survey_nm: str,
        properties: ClusterProperty,
    ) -> tuple[list[float], list[int]]:
        """Returns the observed data for the specified survey and properties.

        For example if the caller has enabled COUNTS then the observed cluster counts
        within each N dimensional bin will be returned.
        """
        data_types = []
        for cluster_property in ClusterProperty:
            include_prop = cluster_property & properties
            if not include_prop:
                continue

            if cluster_property == ClusterProperty.COUNTS:
                # pylint: disable=no-member
                data_types.append(sacc.standard_types.cluster_counts)
            elif cluster_property == ClusterProperty.MASS:
                # pylint: disable=no-member
                data_types.append(sacc.standard_types.cluster_mean_log_mass)

        data_vectors, sacc_indices = self._get_observed_data_and_indices_by_survey(
            survey_nm, data_types, 3
        )
        return data_vectors, sacc_indices

    def get_bin_edges(
        self, survey_nm: str, properties: ClusterProperty
    ) -> list[SaccBin]:
        """Returns the limits for all z, mass bins for the requested data type."""
        bins = []
        data_type = None
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
                continue

            bin_combinations_for_survey = (
                self._all_bin_combinations_for_data_type_and_survey(
                    survey_nm, data_type, 3
                )
            )
            for _, z_tracer, mass_tracer in bin_combinations_for_survey:
                z_data: sacc.tracers.BinZTracer = self.sacc_data.get_tracer(z_tracer)
                mass_data: sacc.tracers.BinRichnessTracer = self.sacc_data.get_tracer(
                    mass_tracer
                )
                sacc_bin = SaccBin([z_data, mass_data])
                bins.append(sacc_bin)

        # Remove duplicates while preserving order (i.e. dont use set())
        unique_bins = list(dict.fromkeys(bins))
        return unique_bins
