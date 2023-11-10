"""Tests for the binned cluster number counts module."""
from typing import Tuple
import sacc
import pytest
import pyccl
from firecrown.models.cluster.integrator.integrator import Integrator
from firecrown.likelihood.gauss_family.statistic.source.source import SourceSystematic
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.models.cluster.abundance import ClusterAbundance, AbundanceIntegrand
from firecrown.likelihood.gauss_family.statistic.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)


class MockIntegrator(Integrator):
    """A mock integrator used by the cluster number counts statistic for testing"""

    def integrate(
        self,
        integrand: AbundanceIntegrand,
    ) -> float:
        """Integrate the integrand over the bounds and include extra_args to integral"""
        return 1.0

    def set_integration_bounds(
        self,
        cl_abundance: ClusterAbundance,
        z_proxy_limits: Tuple[float, float],
        mass_proxy_limits: Tuple[float, float],
    ) -> None:
        """Set the limits of integration and extra arguments for the integral"""


def test_create_binned_number_counts():
    integrator = MockIntegrator()
    bnc = BinnedClusterNumberCounts(False, False, "Test", integrator)
    assert bnc is not None
    assert bnc.use_cluster_counts is False
    assert bnc.use_mean_log_mass is False
    assert bnc.survey_name == "Test"
    assert bnc.systematics == []
    assert bnc.theory_vector is None
    assert len(bnc.data_vector) == 0

    bnc = BinnedClusterNumberCounts(True, True, "Test", integrator)
    assert bnc.use_cluster_counts is True
    assert bnc.use_mean_log_mass is True

    systematics = [SourceSystematic("mock_systematic")]
    bnc = BinnedClusterNumberCounts(False, False, "Test", integrator, systematics)
    assert bnc.systematics == systematics


def test_get_data_vector():
    integrator = MockIntegrator()
    bnc = BinnedClusterNumberCounts(False, False, "Test", integrator)
    dv = bnc.get_data_vector()
    assert dv is not None
    assert len(dv) == 0


def test_read(cluster_sacc_data: sacc.Sacc):
    integrator = MockIntegrator()
    bnc = BinnedClusterNumberCounts(False, False, "my_survey", integrator)

    with pytest.raises(
        RuntimeError,
        match="has read a data vector of length 0; the length must be positive",
    ):
        bnc.read(cluster_sacc_data)

    bnc = BinnedClusterNumberCounts(True, False, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bin_limits) == 4
    assert len(bnc.data_vector) == 2
    assert len(bnc.sacc_indices) == 2

    bnc = BinnedClusterNumberCounts(False, True, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bin_limits) == 4
    assert len(bnc.data_vector) == 4
    assert len(bnc.sacc_indices) == 4

    bnc = BinnedClusterNumberCounts(True, True, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    assert bnc.sky_area == 4000
    assert len(bnc.bin_limits) == 4
    assert len(bnc.data_vector) == 6
    assert len(bnc.sacc_indices) == 6


def test_compute_theory_vector(cluster_sacc_data: sacc.Sacc):
    integrator = MockIntegrator()
    tools = ModelingTools()

    hmf = pyccl.halos.MassFuncBocquet16()
    cosmo = pyccl.cosmology.CosmologyVanillaLCDM()
    params = ParamsMap()

    tools.cluster_abundance = ClusterAbundance(13, 17, 0, 2, hmf, 4000)
    tools.update(params)
    tools.prepare(cosmo, params)

    bnc = BinnedClusterNumberCounts(True, False, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 4

    bnc = BinnedClusterNumberCounts(False, True, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 8

    bnc = BinnedClusterNumberCounts(True, True, "my_survey", integrator)
    bnc.read(cluster_sacc_data)
    tv = bnc.compute_theory_vector(tools)
    assert tv is not None
    assert len(tv) == 8
