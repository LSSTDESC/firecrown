from __future__ import annotations
from typing import List, Optional
import sacc

from firecrown.integrator import Integrator
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.abundance_data import AbundanceData
from firecrown.models.cluster.kernel import ArgsMapping, KernelType
from .statistic import Statistic, DataVector, TheoryVector
from .source.source import SourceSystematic
from ....modeling_tools import ModelingTools
import numpy as np

import cProfile


class BinnedClusterNumberCounts(Statistic):
    def __init__(
        self,
        cluster_counts: bool,
        mean_log_mass: bool,
        survey_name: str,
        integrator: Integrator,
        systematics: Optional[List[SourceSystematic]] = None,
    ):
        super().__init__()
        self.pr = cProfile.Profile()
        self.systematics = systematics or []
        self.theory_vector: Optional[TheoryVector] = None
        self.use_cluster_counts = cluster_counts
        self.use_mean_log_mass = mean_log_mass
        self.survey_name = survey_name
        self.integrator = integrator
        self.data_vector = DataVector.from_list([])

    def read(self, sacc_data: sacc.Sacc):
        # Build the data vector and indices needed for the likelihood

        data_vector = []
        sacc_indices = []

        sacc_types = sacc.data_types.standard_types
        sacc_adapter = AbundanceData(
            sacc_data, self.survey_name, self.use_cluster_counts, self.use_mean_log_mass
        )

        if self.use_cluster_counts:
            data, indices = sacc_adapter.get_data_and_indices(sacc_types.cluster_counts)
            data_vector += data
            sacc_indices += indices

        if self.use_mean_log_mass:
            data, indices = sacc_adapter.get_data_and_indices(
                sacc_types.cluster_mean_log_mass
            )
            data_vector += data
            sacc_indices += indices

        self.sky_area = sacc_adapter.survey_tracer.sky_area
        # Note - this is the same for both cl mass and cl counts... Why do we need to
        # specify a data type?
        self.bin_limits = sacc_adapter.get_bin_limits(sacc_types.cluster_mean_log_mass)
        self.data_vector = DataVector.from_list(data_vector)
        print(len(data_vector))
        self.sacc_indices = np.array(sacc_indices)
        super().read(sacc_data)

    def get_data_vector(self) -> DataVector:
        assert self.data_vector is not None
        return self.data_vector

    def compute_theory_vector(self, tools: ModelingTools) -> TheoryVector:
        theory_vector_list = []
        cluster_counts = []
        cluster_masses = []

        if self.use_cluster_counts or self.use_mean_log_mass:
            # self.pr.enable()
            for z_proxy_limits, mass_proxy_limits in self.bin_limits:
                bounds, extra_args, args_mapping = self.get_integration_bounds(
                    tools.cluster_abundance, z_proxy_limits, mass_proxy_limits
                )

                integrand = tools.cluster_abundance.get_integrand()
                counts = self.integrator.integrate(
                    integrand, bounds, args_mapping, extra_args
                )

                cluster_counts.append(counts)
            # self.pr.disable()
            # self.pr.dump_stats("profile.prof")
            theory_vector_list += cluster_counts

        if self.use_mean_log_mass:
            for (z_proxy_limits, mass_proxy_limits), counts in zip(
                self.bin_limits, cluster_counts
            ):
                bounds, extra_args, args_mapping = self.get_integration_bounds(
                    z_proxy_limits, mass_proxy_limits
                )

                integrand = tools.cluster_abundance.get_integrand()
                unnormalized_mass = self.integrator.integrate(
                    integrand, bounds, args_mapping, extra_args
                )

                cluster_mass = unnormalized_mass / counts
                cluster_masses.append(cluster_mass)

            theory_vector_list += cluster_masses

        return TheoryVector.from_list(theory_vector_list)

    def get_integration_bounds(
        self, cl_abundance: ClusterAbundance, z_proxy_limits, mass_proxy_limits
    ):
        args_mapping = ArgsMapping()
        args_mapping.integral_bounds = {KernelType.mass.name: 0, KernelType.z.name: 1}

        integral_bounds = [
            (cl_abundance.min_mass, cl_abundance.max_mass),
            (cl_abundance.min_z, cl_abundance.max_z),
        ]

        # If any kernel is a dirac delta for z or M, just replace the
        # true limits with the proxy limits
        for kernel in cl_abundance.dirac_delta_kernels:
            if kernel.kernel_type == KernelType.z_proxy:
                integral_bounds[1] = z_proxy_limits
            elif kernel.kernel_type == KernelType.mass_proxy:
                integral_bounds[0] = mass_proxy_limits

        # If any kernel is not a dirac delta, integrate over the relevant limits
        mapping_idx = len(args_mapping.integral_bounds.keys())
        for kernel in cl_abundance.integrable_kernels:
            args_mapping.integral_bounds[kernel.kernel_type.name] = mapping_idx
            mapping_idx += 1

            if kernel.kernel_type == KernelType.z_proxy:
                integral_bounds.append(z_proxy_limits)
            elif kernel.kernel_type == KernelType.mass_proxy:
                integral_bounds.append(mass_proxy_limits)

            if kernel.integral_bounds is not None:
                integral_bounds.append(kernel.integral_bounds)

        # Lastly, don't integrate any kernels with an analytic solution
        # This means we pass in their limits as extra arguments to the integrator
        extra_args = []
        for i, kernel in enumerate(cl_abundance.analytic_kernels):
            args_mapping.extra_args[kernel.kernel_type.name] = i

            if kernel.kernel_type == KernelType.z_proxy:
                extra_args.append(z_proxy_limits)
            elif kernel.kernel_type == KernelType.mass_proxy:
                extra_args.append(mass_proxy_limits)

            if kernel.integral_bounds is not None:
                extra_args.append(kernel.integral_bounds)

        return integral_bounds, extra_args, args_mapping
