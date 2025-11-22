"""Tests for the sacc_utils module (determine_ccl_kind function)."""

import pytest
import pyccl
import firecrown.models.two_point as models
import firecrown.likelihood._source as source
import firecrown.modeling_tools as mt


def test_determine_ccl_kind():
    assert models.determine_ccl_kind("galaxy_density_cl") == "cl"
    assert models.determine_ccl_kind("galaxy_shearDensity_cl_e") == "cl"
    assert models.determine_ccl_kind("galaxy_shear_cl_ee") == "cl"
    assert models.determine_ccl_kind("galaxy_density_xi") == "NN"
    assert models.determine_ccl_kind("galaxy_shearDensity_xi_t") == "NG"
    assert models.determine_ccl_kind("galaxy_shear_xi_minus") == "GG-"
    assert models.determine_ccl_kind("galaxy_shear_xi_plus") == "GG+"
    assert models.determine_ccl_kind("cmbGalaxy_convergenceDensity_xi") == "NN"
    assert models.determine_ccl_kind("cmbGalaxy_convergenceShear_xi_t") == "NG"
    assert models.determine_ccl_kind("cmbGalaxy_convergenceDensity_cl") == "cl"
    assert models.determine_ccl_kind("cmbGalaxy_convergenceShear_cl_e") == "cl"
    assert models.determine_ccl_kind("cmb_convergence_cl") == "cl"
    with pytest.raises(ValueError):
        _ = models.determine_ccl_kind("bad_sacc_data_type")


def test_calculate_pk_lacking_pk(
    tools_with_vanilla_cosmology: mt.ModelingTools, empty_pyccl_tracer: pyccl.Tracer
):
    dummy = source.Tracer(empty_pyccl_tracer)
    with pytest.raises(ValueError):
        _ = models.calculate_pk(
            "no such entry",
            tools_with_vanilla_cosmology,
            dummy,
            dummy,
        )
