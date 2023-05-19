"""Cluster Number Count statistic support.
This module reads the necessary data from a SACC file to compute the
theoretical prediction of cluster number counts inside bins of redshift
 and a mass proxy. For further information, check README.md.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, final

import numpy as np

from ....sacc_support import sacc, ClusterSurveyTracer

from .statistic import Statistic, DataVector, TheoryVector
from .source.source import SourceSystematic
from ....parameters import (
    RequiredParameters,
    DerivedParameterCollection,
)
from ....models.cluster_abundance import ClusterAbundance
from ....models.cluster_mass import ClusterMass, ClusterMassArgument
from ....models.cluster_redshift import ClusterRedshift, ClusterRedshiftArgument
from ....modeling_tools import ModelingTools


class ClusterNumberCounts(Statistic):
    """A Cluster Number Count statistic (e.g., halo mass function,
    multiplicity functions, volume element,  etc.).
    This subclass implements the read and computes method for
    the Statistic class. It is used to compute the theoretical prediction of
    cluster number counts given a SACC file and a cosmology. For
    further information on how the SACC file shall be created,
    check README.md.
    """

    def __init__(
        self,
        sacc_tracer: str,
        cluster_abundance: ClusterAbundance,
        cluster_mass: ClusterMass,
        cluster_redshift: ClusterRedshift,
        systematics: Optional[List[SourceSystematic]] = None,
        use_cluster_counts: bool = True,
        use_mean_mass: bool = False,
    ):
        """Initialize the ClusterNumberCounts object.
        Parameters

        :param sacc_tracer: The SACC tracer. There must be only one tracer for all
            the number Counts data points. Following the SACC file
            documentation README.md, this string should be
            'cluster_counts_true_mass'.
        :param cluster_abundance: The cluster abundance model to use.
        :param systematics: A list of the statistics-level systematics to apply to
            the statistic. The default of `None` implies no systematics.

        """
        super().__init__()

        self.sacc_tracer = sacc_tracer
        self.systematics = systematics or []
        self.data_vector: Optional[DataVector] = None
        self.theory_vector: Optional[TheoryVector] = None
        self.cluster_abundance: ClusterAbundance = cluster_abundance
        self.cluster_mass: ClusterMass = cluster_mass
        self.cluster_redshift: ClusterRedshift = cluster_redshift
        self.tracer_args: List[Tuple[ClusterRedshiftArgument, ClusterMassArgument]] = []
        self.use_cluster_counts: bool = use_cluster_counts
        self.use_mean_mass: bool = use_mean_mass

        if not self.use_cluster_counts and not self.use_mean_mass:
            raise ValueError(
                "At least one of use_cluster_counts and use_mean_mass must be True."
            )

    @final
    def _reset(self) -> None:
        """Reset all contained Updatable objects."""
        self.cluster_abundance.reset()
        self.cluster_mass.reset()
        self.cluster_redshift.reset()

    @final
    def _required_parameters(self) -> RequiredParameters:
        """Return an empty RequiredParameters."""
        return (
            self.cluster_abundance.required_parameters()
            + self.cluster_mass.required_parameters()
            + self.cluster_redshift.required_parameters()
        )

    @final
    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Return an empty DerivedParameterCollection."""
        derived_parameters = DerivedParameterCollection([])
        derived_parameters += self.cluster_abundance.get_derived_parameters()
        derived_parameters += self.cluster_mass.get_derived_parameters()
        derived_parameters += self.cluster_redshift.get_derived_parameters()

        return derived_parameters

    def _read_data_type(self, sacc_data, data_type):
        """Internal function to read the data from the SACC file."""
        tracers_combinations = np.array(
            sacc_data.get_tracer_combinations(data_type=data_type)
        )

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

        cluster_survey_tracers = tracers_combinations[:, 0]

        if self.sacc_tracer not in cluster_survey_tracers:
            raise ValueError(
                f"The SACC tracer {self.sacc_tracer} is not "
                f"present in the SACC file."
            )

        survey_selection = cluster_survey_tracers == self.sacc_tracer

        z_tracers = np.unique(tracers_combinations[survey_selection, 1])
        logM_tracers = np.unique(tracers_combinations[survey_selection, 2])

        z_tracer_bins: Dict[str, ClusterRedshiftArgument] = {
            z_tracer: self.cluster_redshift.gen_bin_from_tracer(
                sacc_data.get_tracer(z_tracer)
            )
            for z_tracer in z_tracers
        }
        logM_tracer_bins: Dict[str, ClusterMassArgument] = {
            logM_tracer: self.cluster_mass.gen_bin_from_tracer(
                sacc_data.get_tracer(logM_tracer)
            )
            for logM_tracer in logM_tracers
        }

        self.tracer_args = [
            (z_tracer_bins[z_tracer], logM_tracer_bins[logM_tracer])
            for _, z_tracer, logM_tracer in tracers_combinations[survey_selection]
        ]

        self.cluster_abundance.read(sacc_data)
        self.cluster_mass.read(sacc_data)
        self.cluster_redshift.read(sacc_data)

        data_vector_list = list(
            sacc_data.get_mean(data_type=data_type)[survey_selection]
        )
        sacc_indices_list = list(
            sacc_data.indices(data_type=data_type)[survey_selection]
        )

        return data_vector_list, sacc_indices_list

    def read(self, sacc_data):
        """Read the data for this statistic from the SACC file.
        This function takes the SACC file and extract the necessary
        parameters needed to compute the number counts likelihood.
        Check README.MD for a complete description of the method.

        :param sacc_data: The data in the sacc format.
        """

        survey_tracer: ClusterSurveyTracer = sacc_data.get_tracer(self.sacc_tracer)
        if survey_tracer is None:
            raise ValueError(
                f"The SACC file does not contain the ClusterSurveyTracer "
                f"{self.sacc_tracer}."
            )
        if not isinstance(survey_tracer, ClusterSurveyTracer):
            raise ValueError(
                f"The SACC tracer {self.sacc_tracer} is not a ClusterSurveyTracer."
            )

        self.cluster_abundance.sky_area = survey_tracer.sky_area

        data_vector_list = []
        sacc_indices_list = []

        if self.use_cluster_counts:
            # pylint: disable-next=no-member
            cluster_counts = sacc.standard_types.cluster_counts
            (
                cluster_counts_data_vector_list,
                cluster_counts_sacc_indices_list,
            ) = self._read_data_type(sacc_data, cluster_counts)
            data_vector_list += cluster_counts_data_vector_list
            sacc_indices_list += cluster_counts_sacc_indices_list

        if self.use_mean_mass:
            # pylint: disable-next=no-member
            cluster_mean_mass = sacc.standard_types.cluster_mean_mass
            (
                mean_mass_data_vector_list,
                mean_mass_sacc_indices_list,
            ) = self._read_data_type(sacc_data, cluster_mean_mass)

            data_vector_list += mean_mass_data_vector_list
            sacc_indices_list += mean_mass_sacc_indices_list

        self.data_vector = DataVector.from_list(data_vector_list)
        self.sacc_indices = np.array(sacc_indices_list)

    def get_data_vector(self) -> DataVector:
        """Return the data vector; raise exception if there is none."""
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        """Compute a Number Count statistic using the data from the
        Read method, the cosmology object, and the Bocquet16 halo mass function.
                Check README.MD for a complete description of the method.

        :param tools: ModelingTools firecrown object
            Firecrown object used to load the required cosmology.

        :return: Numpy Array of floats
            An array with the theoretical prediction of the number of clusters
            in each bin of redsfhit and mass.
        """
        ccl_cosmo = tools.get_ccl_cosmology()
        theory_vector_list = []
        cluster_counts_list = []

        if self.use_cluster_counts or self.use_mean_mass:
            cluster_counts_list = [
                self.cluster_abundance.compute(ccl_cosmo, logM_tracer_arg, z_tracer_arg)
                for z_tracer_arg, logM_tracer_arg in self.tracer_args
            ]
            if self.use_cluster_counts:
                theory_vector_list += cluster_counts_list

        if self.use_mean_mass:
            mean_mass_list = [
                self.cluster_abundance.compute_unormalized_mean_logM(
                    ccl_cosmo, logM_tracer_arg, z_tracer_arg
                )
                / counts
                for (z_tracer_arg, logM_tracer_arg), counts in zip(
                    self.tracer_args, cluster_counts_list
                )
            ]
            theory_vector_list += mean_mass_list
        return TheoryVector.from_list(theory_vector_list)
