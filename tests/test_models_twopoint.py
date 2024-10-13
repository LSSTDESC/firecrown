"""
Tests for the TwoPointTheory class and related functions.
"""

import pytest
import firecrown.models.two_point as models


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
    with pytest.raises(ValueError):
        _ = models.determine_ccl_kind("bad_sacc_data_type")
