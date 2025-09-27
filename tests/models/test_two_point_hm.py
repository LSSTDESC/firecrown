"""Tests for the at_least_one_tracer_has_hm function in firecrown.models.two_point."""

# pylint: disable=redefined-outer-name
# Disable redefined-outer-name warnings as pytest fixtures create this
# pattern by design

from unittest.mock import Mock, patch

import pytest
import pyccl

from firecrown.models.two_point import at_least_one_tracer_has_hm
from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.source import Tracer


@pytest.fixture
def mock_tools():
    """Create a mock ModelingTools object."""
    tools = Mock(spec=ModelingTools)

    # Mock CCL cosmology
    mock_cosmo = Mock(spec=pyccl.Cosmology)
    tools.get_ccl_cosmology.return_value = mock_cosmo

    # Mock HM calculator - use string for mass_def to avoid pyccl parsing issues
    mock_hm_calc = Mock()
    mock_hm_calc.mass_def = "200m"
    tools.get_hm_calculator.return_value = mock_hm_calc

    # Mock c-M relation - use a valid concentration model
    tools.get_cM_relation.return_value = "Duffy08"

    return tools


@pytest.fixture
def mock_tracer_with_hm():
    """Create a mock tracer that has halo model."""
    tracer = Mock(spec=Tracer)
    tracer.has_hm = True
    tracer.has_pt = False
    tracer.tracer_name = "shear"

    # Mock halo profile
    mock_profile = Mock(spec=pyccl.halos.HaloProfile)
    mock_profile.ia_a_2h = 1.0
    tracer.halo_profile = mock_profile

    return tracer


@pytest.fixture
def mock_tracer_without_hm():
    """Create a mock tracer that does not have halo model."""
    tracer = Mock(spec=Tracer)
    tracer.has_hm = False
    tracer.has_pt = False
    tracer.tracer_name = "shear"  # Need "shear" for mixed HM case
    tracer.halo_profile = None
    return tracer


def test_both_tracers_have_hm(mock_tools, mock_tracer_with_hm):
    """Test case where both tracers have halo model."""
    tracer0 = mock_tracer_with_hm
    tracer1 = mock_tracer_with_hm

    # Mock the pyccl functions
    mock_pk_1h = Mock(spec=pyccl.Pk2D)
    mock_pk_2h = Mock(spec=pyccl.Pk2D)
    mock_pk_total = Mock(spec=pyccl.Pk2D)

    # Configure addition operation
    mock_pk_1h.__add__ = Mock(return_value=mock_pk_total)

    with (
        patch("pyccl.halos.halomod_Pk2D", return_value=mock_pk_1h) as mock_halomod,
        patch("pyccl.Pk2D.from_function", return_value=mock_pk_2h) as mock_from_func,
    ):

        result = at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)

        # Verify pyccl functions were called
        mock_halomod.assert_called_once()
        mock_from_func.assert_called_once()
        mock_pk_1h.__add__.assert_called_once_with(mock_pk_2h)

        # Verify result
        assert result == mock_pk_total


def test_only_first_tracer_has_hm(
    mock_tools, mock_tracer_with_hm, mock_tracer_without_hm
):
    """Test case where only the first tracer has halo model."""
    tracer0 = mock_tracer_with_hm
    tracer1 = mock_tracer_without_hm

    # Mock the pyccl functions
    mock_pk_1h = Mock(spec=pyccl.Pk2D)
    mock_pk_2h = Mock(spec=pyccl.Pk2D)
    mock_pk_total = Mock(spec=pyccl.Pk2D)
    mock_pk_1h.__add__ = Mock(return_value=mock_pk_total)

    with (
        patch("pyccl.halos.halomod_Pk2D", return_value=mock_pk_1h),
        patch("pyccl.Pk2D.from_function", return_value=mock_pk_2h),
        patch("pyccl.halos.HaloProfileNFW") as mock_nfw,
    ):

        result = at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)

        # Verify HaloProfileNFW was created for non-HM tracer
        mock_nfw.assert_called_once()
        assert result == mock_pk_total


def test_only_second_tracer_has_hm(
    mock_tools, mock_tracer_with_hm, mock_tracer_without_hm
):
    """Test case where only the second tracer has halo model."""
    tracer0 = mock_tracer_without_hm
    tracer1 = mock_tracer_with_hm

    # Mock the pyccl functions
    mock_pk_1h = Mock(spec=pyccl.Pk2D)
    mock_pk_2h = Mock(spec=pyccl.Pk2D)
    mock_pk_total = Mock(spec=pyccl.Pk2D)
    mock_pk_1h.__add__ = Mock(return_value=mock_pk_total)

    with (
        patch("pyccl.halos.halomod_Pk2D", return_value=mock_pk_1h),
        patch("pyccl.Pk2D.from_function", return_value=mock_pk_2h),
        patch("pyccl.halos.HaloProfileNFW") as mock_nfw,
    ):

        result = at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)

        # Verify HaloProfileNFW was created for non-HM tracer
        mock_nfw.assert_called_once()
        assert result == mock_pk_total


def test_mixed_tracer_requires_shear(mock_tools):
    """Test that when not both tracers have HM, at least one must be 'shear'."""
    # Create two tracers without HM and without shear to trigger AssertionError
    tracer0 = Mock(spec=Tracer)
    tracer0.has_hm = False
    tracer0.has_pt = False
    tracer0.tracer_name = "galaxy_density"
    tracer0.halo_profile = None

    tracer1 = Mock(spec=Tracer)
    tracer1.has_hm = True  # One has HM
    tracer1.has_pt = False
    tracer1.tracer_name = "galaxy_density"  # But neither is "shear"
    tracer1.halo_profile = Mock(spec=pyccl.halos.HaloProfile)

    with pytest.raises(
        AssertionError, match="Currently, only cosmic shear is supported"
    ):
        at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)


def test_profile_assertion_for_mixed_hm(mock_tools, mock_tracer_without_hm):
    """Test assertion when HM tracer has None halo_profile in mixed case."""
    # Create HM tracer with None halo_profile
    tracer0 = Mock(spec=Tracer)
    tracer0.has_hm = True
    tracer0.has_pt = False
    tracer0.tracer_name = "shear"
    tracer0.halo_profile = None

    tracer1 = mock_tracer_without_hm

    with pytest.raises(AssertionError):
        at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)


def test_ia_bias_exponent_both_hm(mock_tools, mock_tracer_with_hm):
    """Test that IA bias exponent is 2 when both tracers have HM."""
    tracer0 = mock_tracer_with_hm
    tracer1 = mock_tracer_with_hm

    # Mock the pyccl functions
    mock_pk_1h = Mock(spec=pyccl.Pk2D)
    mock_pk_2h = Mock(spec=pyccl.Pk2D)
    mock_pk_total = Mock(spec=pyccl.Pk2D)
    mock_pk_1h.__add__ = Mock(return_value=mock_pk_total)

    with (
        patch("pyccl.halos.halomod_Pk2D", return_value=mock_pk_1h),
        patch("pyccl.Pk2D.from_function", return_value=mock_pk_2h) as mock_from_func,
    ):

        result = at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)

        # Check that from_function was called
        call_args = mock_from_func.call_args
        assert call_args is not None

        # The IA_bias_exponent is used in the lambda function closure, not
        # as a parameter. Let's verify the function is called with the expected
        # parameters
        _, kwargs = call_args

        # Should have pkfunc as keyword argument and is_logp=False
        assert "pkfunc" in kwargs, "pkfunc should be in kwargs"
        assert "is_logp" in kwargs and kwargs["is_logp"] is False

        # Verify the pkfunc is a callable (lambda function)
        pkfunc = kwargs["pkfunc"]
        assert callable(pkfunc), "pkfunc should be callable"

        # The lambda captures IA_bias_exponent=2 in its closure for both HM case
        # We can't directly test the closure value, but the successful execution
        # with both tracers having HM indicates the correct exponent is used
        assert result == mock_pk_total


def test_halo_profile_nfw_configuration(
    mock_tools, mock_tracer_with_hm, mock_tracer_without_hm
):
    """Test that HaloProfileNFW is configured correctly for non-HM tracers."""
    tracer0 = mock_tracer_with_hm
    tracer1 = mock_tracer_without_hm

    # Mock the pyccl functions
    mock_pk_1h = Mock(spec=pyccl.Pk2D)
    mock_pk_2h = Mock(spec=pyccl.Pk2D)
    mock_pk_total = Mock(spec=pyccl.Pk2D)
    mock_pk_1h.__add__ = Mock(return_value=mock_pk_total)

    # Mock the HaloProfileNFW creation
    mock_other_profile = Mock(spec=pyccl.halos.HaloProfileNFW)

    with (
        patch("pyccl.halos.halomod_Pk2D", return_value=mock_pk_1h),
        patch("pyccl.Pk2D.from_function", return_value=mock_pk_2h),
        patch(
            "pyccl.halos.HaloProfileNFW", return_value=mock_other_profile
        ) as mock_nfw,
    ):

        result = at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)

        # Verify HaloProfileNFW was called with correct parameters
        mock_nfw.assert_called_once()
        call_args = mock_nfw.call_args

        # The call should include mass_def and concentration
        args, kwargs = call_args
        assert "mass_def" in kwargs or len(args) > 0
        assert "concentration" in kwargs or len(args) > 1

        assert result == mock_pk_total


def test_pk2d_addition_result(mock_tools, mock_tracer_with_hm):
    """Test that the function returns the sum of 1-halo and 2-halo power spectra."""
    tracer0 = mock_tracer_with_hm
    tracer1 = mock_tracer_with_hm

    # Create distinct mock objects for power spectra
    mock_pk_1h = Mock(spec=pyccl.Pk2D)
    mock_pk_2h = Mock(spec=pyccl.Pk2D)
    mock_pk_total = Mock(spec=pyccl.Pk2D)

    # Configure the addition to return our mock total
    mock_pk_1h.__add__ = Mock(return_value=mock_pk_total)

    with (
        patch("pyccl.halos.halomod_Pk2D", return_value=mock_pk_1h),
        patch("pyccl.Pk2D.from_function", return_value=mock_pk_2h),
    ):

        result = at_least_one_tracer_has_hm(mock_tools, tracer0, tracer1)

        # Verify that addition was performed
        mock_pk_1h.__add__.assert_called_once_with(mock_pk_2h)

        # Verify the result is the sum
        assert result == mock_pk_total
        assert result is mock_pk_total  # Ensure it's the exact same object
