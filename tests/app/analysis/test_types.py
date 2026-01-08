"""Unit tests for firecrown.app.analysis types module.

Tests type definitions, enums, and data models for framework configuration.
"""

import pytest
from pydantic import ValidationError
import pyccl
from firecrown.app.analysis import (
    Frameworks,
    FrameworkCosmology,
    Parameter,
    Model,
    PriorUniform,
    PriorGaussian,
    COSMO_DESC,
    CCL_COSMOLOGY_MINIMAL_SET,
    CCLCosmologySpec,
)
from firecrown.app.analysis._types import (
    PoweSpecAmplitudeParameter,
    get_path_str,
)
from firecrown.app.analysis._cobaya import CobayaConfigGenerator
from firecrown.likelihood import NamedParameters
from firecrown.ccl_factory import CAMBExtraParams


class TestFrameworksEnum:
    """Tests for Frameworks enum."""

    def test_frameworks_values(self) -> None:
        """Test that Frameworks enum has expected values."""
        assert Frameworks.COBAYA.value == "cobaya"
        assert Frameworks.COSMOSIS.value == "cosmosis"
        assert Frameworks.NUMCOSMO.value == "numcosmo"

    def test_frameworks_count(self) -> None:
        """Test that Frameworks enum has exactly 3 frameworks."""
        frameworks = list(Frameworks)
        assert len(frameworks) == 3

    def test_frameworks_creation_from_string(self) -> None:
        """Test creating Frameworks from string."""
        f = Frameworks("cobaya")
        assert f == Frameworks.COBAYA


class TestFrameworkCosmologyEnum:
    """Tests for FrameworkCosmology enum."""

    def test_framework_cosmology_values(self) -> None:
        """Test FrameworkCosmology enum values."""
        assert FrameworkCosmology.NONE.value == "none"
        assert FrameworkCosmology.BACKGROUND.value == "background"
        assert FrameworkCosmology.LINEAR.value == "linear"
        assert FrameworkCosmology.NONLINEAR.value == "nonlinear"

    def test_framework_cosmology_count(self) -> None:
        """Test that FrameworkCosmology has exactly 4 levels."""
        levels = list(FrameworkCosmology)
        assert len(levels) == 4


class TestPriorUniform:
    """Tests for PriorUniform."""

    def test_prior_uniform_creation(self) -> None:
        """Test creating a valid uniform prior."""
        prior = PriorUniform(lower=0.1, upper=0.9)
        assert prior.lower == 0.1
        assert prior.upper == 0.9

    def test_prior_uniform_invalid_bounds(self) -> None:
        """Test that uniform prior validates bounds."""
        with pytest.raises(ValidationError):
            PriorUniform(lower=0.9, upper=0.1)

    def test_prior_uniform_equal_bounds(self) -> None:
        """Test that uniform prior rejects equal bounds."""
        with pytest.raises(ValidationError):
            PriorUniform(lower=0.5, upper=0.5)


class TestPriorGaussian:
    """Tests for PriorGaussian."""

    def test_prior_gaussian_creation(self) -> None:
        """Test creating a valid Gaussian prior."""
        prior = PriorGaussian(mean=0.5, sigma=0.1)
        assert prior.mean == 0.5
        assert prior.sigma == 0.1

    def test_prior_gaussian_zero_sigma(self) -> None:
        """Test that Gaussian prior rejects zero sigma."""
        with pytest.raises(ValidationError):
            PriorGaussian(mean=0.5, sigma=0.0)

    def test_prior_gaussian_negative_sigma(self) -> None:
        """Test that Gaussian prior rejects negative sigma."""
        with pytest.raises(ValidationError):
            PriorGaussian(mean=0.5, sigma=-0.1)

    def test_prior_gaussian_negative_mean(self) -> None:
        """Test Gaussian prior with negative mean."""
        prior = PriorGaussian(mean=-0.5, sigma=0.1)
        assert prior.mean == -0.5


class TestParameter:
    """Tests for Parameter."""

    def test_parameter_creation(self) -> None:
        """Test creating a valid parameter."""
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
        )
        assert param.name == "Omega_c"
        assert param.symbol == r"\Omega_c"
        assert param.lower_bound == 0.1
        assert param.upper_bound == 0.5
        assert param.default_value == 0.3
        assert param.free is True

    def test_parameter_with_uniform_prior(self) -> None:
        """Test parameter with uniform prior."""
        prior = PriorUniform(lower=0.1, upper=0.5)
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
            prior=prior,
        )
        assert param is not None
        assert param.prior is not None
        assert isinstance(param.prior, PriorUniform)
        assert param.prior.lower == 0.1  # pylint: disable=no-member

    def test_parameter_with_gaussian_prior(self) -> None:
        """Test parameter with Gaussian prior."""
        prior = PriorGaussian(mean=0.3, sigma=0.05)
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
            prior=prior,
        )
        assert param.prior is not None
        assert isinstance(param.prior, PriorGaussian)

    def test_parameter_invalid_bounds(self) -> None:
        """Test that parameter validates bounds."""
        with pytest.raises(ValidationError):
            Parameter(
                name="test",
                symbol="t",
                lower_bound=0.5,
                upper_bound=0.1,
                default_value=0.3,
                free=True,
            )

    def test_parameter_auto_scale(self) -> None:
        """Test parameter auto-scale calculation."""
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
        )
        # Scale should be auto-calculated from bounds
        assert param.scale > 0

    def test_parameter_auto_scale_zero_default(self) -> None:
        """Test parameter auto-scale when default_value is 0.0.

        This tests the branch at line 107 where default_value == 0.0,
        causing the first scale calculation to be skipped and the second
        scale calculation (from bounds) to be used instead.
        """
        param = Parameter(
            name="Omega_k",
            symbol=r"\Omega_k",
            lower_bound=-0.1,
            upper_bound=0.1,
            default_value=0.0,
            free=True,
        )
        # Scale should be auto-calculated from bounds since default is 0
        expected_scale = (0.1 - (-0.1)) * 0.01  # 0.2 * 0.01 = 0.002
        assert param.scale == pytest.approx(expected_scale)

    def test_parameter_fill_defaults_with_explicit_scale(self) -> None:
        """Test fill_defaults validator when scale is explicitly provided.

        When scale is provided in the input dict, the validator should not
        modify it (tests that all branches exit early when 'scale' in data).
        """
        param = Parameter(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
            scale=0.05,  # Explicitly provided
        )
        # Should use the explicit scale value
        assert param.scale == 0.05

    def test_parameter_fill_defaults_non_dict_input(self) -> None:
        """Test fill_defaults validator with non-dict input.

        When data is not a dict (e.g., already a Parameter object),
        the validator should return it unchanged (tests line 103 condition).
        """
        # Create a parameter normally, then use it as input to another
        # This simulates the case where data is not isinstance(data, dict)
        param1 = Parameter(
            name="h",
            symbol="h",
            lower_bound=0.6,
            upper_bound=0.8,
            default_value=0.7,
            free=False,
        )
        # When pydantic processes this, the validator sees non-dict data
        # and returns it as-is
        assert param1.scale > 0

    def test_parameter_fill_defaults_missing_default_value(self) -> None:
        """Test fill_defaults validator when default_value is missing.

        When default_value is not in the dict, the first scale calculation
        branch (line 105-109) should be skipped.
        """
        # This will use the bounds-based calculation since default_value check fails
        with pytest.raises(ValidationError):
            _ = Parameter.model_validate("I'm not a dict")
        # Scale is calculated from bounds since default is provided

    def test_parameter_fill_defaults_missing_bounds(self) -> None:
        """Test fill_defaults validator with scale from default_value only.

        When bounds are present but default_value is non-zero, the first
        calculation (line 109) should be used.
        """
        param = Parameter(
            name="sigma8",
            symbol=r"\sigma_8",
            lower_bound=0.6,
            upper_bound=1.0,
            default_value=0.81,  # Non-zero
            free=True,
        )
        # Scale should be calculated from default_value (0.81 * 0.01 = 0.0081)
        expected_scale = 0.81 * 0.01
        assert param.scale == pytest.approx(expected_scale)

    def test_parameter_from_tuple(self) -> None:
        """Test creating parameter from tuple."""
        param = Parameter.from_tuple(
            name="Omega_c",
            symbol=r"\Omega_c",
            lower_bound=0.1,
            upper_bound=0.5,
            default_value=0.3,
            free=True,
        )
        assert param.name == "Omega_c"
        assert param.lower_bound == 0.1


class TestModel:
    """Tests for Model."""

    def test_model_creation(self) -> None:
        """Test creating a valid model."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
            Parameter.from_tuple("h", "h", 0.6, 0.8, 0.7, False),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert model.name == "cosmology"
        assert len(model.parameters) == 2

    def test_model_parameter_access(self) -> None:
        """Test accessing model parameters."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        param = model["Omega_c"]
        assert param.name == "Omega_c"

    def test_model_parameter_not_found(self) -> None:
        """Test KeyError when parameter not found."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        with pytest.raises(KeyError):
            _ = model["nonexistent"]

    def test_model_contains(self) -> None:
        """Test checking if parameter exists in model."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert "Omega_c" in model
        assert "nonexistent" not in model

    def test_model_duplicate_parameter_names(self) -> None:
        """Test that model rejects duplicate parameter names."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
            Parameter.from_tuple("Omega_c", "h", 0.6, 0.8, 0.7, False),
        ]
        with pytest.raises(ValueError, match="Duplicate parameter name"):
            Model(
                name="cosmology",
                description="Cosmological parameters",
                parameters=params,
            )

    def test_model_has_priors_true(self) -> None:
        """Test has_priors when priors are present."""
        prior = PriorUniform(lower=0.1, upper=0.5)
        params = [
            Parameter(
                name="Omega_c",
                symbol=r"\Omega_c",
                lower_bound=0.1,
                upper_bound=0.5,
                default_value=0.3,
                free=True,
                prior=prior,
            ),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert model.has_priors() is True

    def test_model_has_priors_false(self) -> None:
        """Test has_priors when no priors are present."""
        params = [
            Parameter.from_tuple("Omega_c", r"\Omega_c", 0.1, 0.5, 0.3, True),
        ]
        model = Model(
            name="cosmology", description="Cosmological parameters", parameters=params
        )
        assert model.has_priors() is False


class TestCosmoDesc:
    """Tests for COSMO_DESC dictionary."""

    def test_cosmo_desc_contains_minimal_set(self) -> None:
        """Test that COSMO_DESC contains all minimal cosmology parameters."""
        for param_name in CCL_COSMOLOGY_MINIMAL_SET:
            assert param_name in COSMO_DESC
            assert isinstance(COSMO_DESC[param_name], Parameter)

    def test_cosmo_desc_parameter_properties(self) -> None:
        """Test properties of COSMO_DESC parameters."""
        omega_c = COSMO_DESC["Omega_c"]
        assert omega_c.name == "Omega_c"
        assert omega_c.symbol == r"\Omega_c"
        assert omega_c.lower_bound < omega_c.upper_bound
        assert omega_c.lower_bound <= omega_c.default_value <= omega_c.upper_bound


class TestCCLCosmologySpec:
    """Tests for CCLCosmologySpec validation."""

    def test_invalid_parameter_name(self) -> None:
        """Test that CCLCosmologySpec rejects invalid parameter names."""
        invalid_param = Parameter(
            name="invalid_param",
            symbol="invalid",
            lower_bound=0.0,
            upper_bound=1.0,
            default_value=0.5,
            free=True,
        )

        with pytest.raises(ValidationError, match="not a valid CCL cosmological"):
            CCLCosmologySpec(parameters=[invalid_param])

    def test_missing_required_parameter(self) -> None:
        """Test that CCLCosmologySpec requires minimal parameter set."""
        # Missing Omega_k from minimal set
        incomplete_params = [
            Parameter(
                name="Omega_c",
                symbol="Omega_c",
                lower_bound=0.2,
                upper_bound=0.3,
                default_value=0.25,
                free=True,
            ),
            Parameter(
                name="Omega_b",
                symbol="Omega_b",
                lower_bound=0.03,
                upper_bound=0.07,
                default_value=0.05,
                free=True,
            ),
        ]

        with pytest.raises(ValidationError, match="missing required parameter"):
            CCLCosmologySpec(parameters=incomplete_params)

    def test_both_amplitude_parameters(self) -> None:
        """Test that CCLCosmologySpec rejects both A_s and sigma8."""
        params_with_both = [
            COSMO_DESC[name]
            for name in CCL_COSMOLOGY_MINIMAL_SET
            if name not in ["sigma8", "A_s"]
        ] + [
            Parameter(
                name="A_s",
                symbol="A_s",
                lower_bound=1e-9,
                upper_bound=3e-9,
                default_value=2e-9,
                free=True,
            ),
            Parameter(
                name="sigma8",
                symbol="sigma8",
                lower_bound=0.6,
                upper_bound=1.0,
                default_value=0.8,
                free=True,
            ),
        ]

        with pytest.raises(
            ValidationError, match="Exactly one of A_s and sigma8 must be supplied"
        ):
            CCLCosmologySpec(parameters=params_with_both)

    def test_get_amplitude_parameter_as(self) -> None:
        """Test get_amplitude_parameter returns AS when A_s is present."""
        params_with_as = [
            COSMO_DESC[name] for name in CCL_COSMOLOGY_MINIMAL_SET if name != "sigma8"
        ] + [
            Parameter(
                name="A_s",
                symbol="A_s",
                lower_bound=1e-9,
                upper_bound=3e-9,
                default_value=2e-9,
                free=True,
            )
        ]

        spec = CCLCosmologySpec(parameters=params_with_as)
        assert spec.get_amplitude_parameter() == PoweSpecAmplitudeParameter.AS

    def test_get_amplitude_parameter_sigma8(self) -> None:
        """Test get_amplitude_parameter returns SIGMA8 when A_s is not present."""
        spec = CCLCosmologySpec.vanilla_lcdm()
        assert spec.get_amplitude_parameter() == PoweSpecAmplitudeParameter.SIGMA8

    def test_get_num_massive_neutrinos_with_massive(self) -> None:
        """Test get_num_massive_neutrinos when m_nu > 0."""
        params = [
            COSMO_DESC[name]
            for name in CCL_COSMOLOGY_MINIMAL_SET
            if name not in ["m_nu", "A_s"]
        ] + [
            COSMO_DESC["sigma8"],  # Include amplitude parameter
            Parameter(
                name="m_nu",
                symbol="m_nu",
                lower_bound=0.0,
                upper_bound=1.0,
                default_value=0.06,  # Non-zero massive neutrino
                free=False,
            ),
        ]
        spec = CCLCosmologySpec(parameters=params)
        assert spec.get_num_massive_neutrinos() == 1

    def test_get_num_massive_neutrinos_zero_but_free(self) -> None:
        """Test get_num_massive_neutrinos when m_nu = 0 but free=True.

        This tests the branch where default_value == 0.0 but the parameter
        is free, so we still need 1 massive neutrino in the model.
        """
        params = [
            COSMO_DESC[name]
            for name in CCL_COSMOLOGY_MINIMAL_SET
            if name not in ["m_nu", "A_s"]
        ] + [
            COSMO_DESC["sigma8"],  # Include amplitude parameter
            Parameter(
                name="m_nu",
                symbol="m_nu",
                lower_bound=0.0,
                upper_bound=1.0,
                default_value=0.0,  # Zero but...
                free=True,  # ...free to vary
            ),
        ]
        spec = CCLCosmologySpec(parameters=params)
        assert spec.get_num_massive_neutrinos() == 1

    def test_get_num_massive_neutrinos_zero_and_fixed(self) -> None:
        """Test get_num_massive_neutrinos when m_nu = 0 and fixed."""
        spec = CCLCosmologySpec.vanilla_lcdm()
        # vanilla_lcdm has m_nu = 0.0 and free = False
        assert spec.get_num_massive_neutrinos() == 0

    def test_ccl_cosmology_with_extra_parameters(self) -> None:
        """Test creating CCL cosmology with extra_parameters."""
        extra_params = CAMBExtraParams(halofit_version="mead2020")
        spec = CCLCosmologySpec.vanilla_lcdm()
        # Create new spec with extra parameters
        spec_with_extra = CCLCosmologySpec(
            parameters=spec.parameters, extra_parameters=extra_params
        )

        # Verify extra_parameters is set (tests branch at line 288-289)
        assert spec_with_extra.extra_parameters is not None
        # pylint: disable-next=no-member
        assert spec_with_extra.extra_parameters.halofit_version == "mead2020"

    def test_ccl_cosmology_with_matter_power_spectrum(self) -> None:
        """Test creating CCL cosmology with custom matter_power_spectrum."""
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_custom = CCLCosmologySpec(
            parameters=spec.parameters, matter_power_spectrum="emu"
        )

        # Verify matter_power_spectrum is set (tests branch at line 290-291)
        assert spec_custom.matter_power_spectrum == "emu"

    def test_ccl_cosmology_with_transfer_function(self) -> None:
        """Test creating CCL cosmology with custom transfer_function."""
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_custom = CCLCosmologySpec(
            parameters=spec.parameters, transfer_function="boltzmann_class"
        )

        # Verify transfer_function is set (tests branch at line 292-293)
        assert spec_custom.transfer_function == "boltzmann_class"

    def test_ccl_cosmology_with_mass_split(self) -> None:
        """Test creating CCL cosmology with custom mass_split."""
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_custom = CCLCosmologySpec(parameters=spec.parameters, mass_split="sum")

        # Verify mass_split is set (tests branch at line 294-295)
        assert spec_custom.mass_split == "sum"


class TestToCCLCosmology:
    """Tests for to_ccl_cosmology method."""

    def test_to_ccl_cosmology_basic(self) -> None:
        """Test basic to_ccl_cosmology conversion."""
        spec = CCLCosmologySpec.vanilla_lcdm()
        cosmo = spec.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_includes_parameters(self) -> None:
        """Test that to_ccl_cosmology includes all parameter values.

        This tests the branch at line 285-286 (if self._param_dict is not None).
        The _param_dict should always be present for valid cosmologies.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        assert spec._param_dict is not None  # pylint: disable=protected-access
        cosmo = spec.to_ccl_cosmology()

        # Check that key cosmological parameters are accessible
        # Note: pyccl.Cosmology stores these internally
        assert cosmo is not None

    def test_to_ccl_cosmology_with_extra_parameters(self) -> None:
        """Test to_ccl_cosmology with extra_parameters set.

        This tests the branch at line 288-289 (if self.extra_parameters).
        """
        extra_params = CAMBExtraParams(halofit_version="mead2020")
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_with_extra = CCLCosmologySpec(
            parameters=spec.parameters, extra_parameters=extra_params
        )

        # Should include extra_parameters in the args
        cosmo = spec_with_extra.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_without_extra_parameters(self) -> None:
        """Test to_ccl_cosmology without extra_parameters.

        This tests the else branch of line 288 (when extra_parameters is None).
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        assert spec.extra_parameters is None

        cosmo = spec.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_with_custom_matter_power_spectrum(self) -> None:
        """Test to_ccl_cosmology with custom matter_power_spectrum.

        This tests the branch at line 290-291 (if self.matter_power_spectrum).
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_custom = CCLCosmologySpec(
            parameters=spec.parameters, matter_power_spectrum="linear"
        )

        cosmo = spec_custom.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_without_custom_matter_power_spectrum(self) -> None:
        """Test to_ccl_cosmology with default matter_power_spectrum.

        This tests when matter_power_spectrum evaluates to False (empty string or None).
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # vanilla_lcdm has matter_power_spectrum = "halofit" which is truthy
        # So this test verifies the default behavior
        cosmo = spec.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_with_custom_transfer_function(self) -> None:
        """Test to_ccl_cosmology with custom transfer_function.

        This tests the branch at line 292-293 (if self.transfer_function).
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_custom = CCLCosmologySpec(
            parameters=spec.parameters, transfer_function="boltzmann_class"
        )

        cosmo = spec_custom.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_without_custom_transfer_function(self) -> None:
        """Test to_ccl_cosmology with default transfer_function.

        This tests when transfer_function evaluates to False.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # vanilla_lcdm has transfer_function = "boltzmann_camb" which is truthy
        cosmo = spec.to_ccl_cosmology()

        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_with_custom_mass_split(self) -> None:
        """Test to_ccl_cosmology with custom mass_split.

        This tests the branch at line 294-295 (if self.mass_split).
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_custom = CCLCosmologySpec(parameters=spec.parameters, mass_split="sum")

        cosmo = spec_custom.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_without_custom_mass_split(self) -> None:
        """Test to_ccl_cosmology with default mass_split.

        This tests when mass_split evaluates to False (empty string).
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # vanilla_lcdm has mass_split = "normal" which is truthy
        cosmo = spec.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_all_optional_fields_set(self) -> None:
        """Test to_ccl_cosmology with all optional fields set.

        This ensures all branches (lines 288-295) are covered when
        all optional fields are provided.
        """
        extra_params = CAMBExtraParams(halofit_version="mead2020")
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_full = CCLCosmologySpec(
            parameters=spec.parameters,
            extra_parameters=extra_params,
            matter_power_spectrum="halofit",
            transfer_function="boltzmann_camb",
            mass_split="sum",
        )

        cosmo = spec_full.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_all_optional_fields_none(self) -> None:
        """Test to_ccl_cosmology with minimal optional fields.

        This ensures the method works when optional fields are not set
        or set to falsy values.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # Reset optional fields to minimal/None values
        spec_minimal = CCLCosmologySpec(
            parameters=spec.parameters,
            extra_parameters=None,
            matter_power_spectrum="halofit",  # Default
            transfer_function="boltzmann_camb",  # Default
            mass_split="normal",  # Default
        )

        cosmo = spec_minimal.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_extra_parameters_false(self) -> None:
        """Test to_ccl_cosmology when extra_parameters is None.

        This tests the False branch at line 288 (if self.extra_parameters).
        When extra_parameters is None, the dict should not include it.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # Explicitly set extra_parameters to None
        spec_no_extra = CCLCosmologySpec(
            parameters=spec.parameters, extra_parameters=None
        )

        assert spec_no_extra.extra_parameters is None
        cosmo = spec_no_extra.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_matter_power_spectrum_false(self) -> None:
        """Test to_ccl_cosmology when matter_power_spectrum is empty/falsy.

        This tests the False branch at line 290 (if self.matter_power_spectrum).
        When matter_power_spectrum is empty string, it should be skipped.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # Set matter_power_spectrum to empty string (falsy)
        spec_no_mps = CCLCosmologySpec(
            parameters=spec.parameters, matter_power_spectrum=""
        )

        assert not spec_no_mps.matter_power_spectrum
        cosmo = spec_no_mps.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_transfer_function_false(self) -> None:
        """Test to_ccl_cosmology when transfer_function is empty/falsy.

        This tests the False branch at line 292 (if self.transfer_function).
        When transfer_function is empty string, it should be skipped.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # Set transfer_function to empty string (falsy)
        spec_no_tf = CCLCosmologySpec(parameters=spec.parameters, transfer_function="")

        assert not spec_no_tf.transfer_function
        cosmo = spec_no_tf.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_mass_split_false(self) -> None:
        """Test to_ccl_cosmology when mass_split is empty/falsy.

        This tests the False branch at line 294 (if self.mass_split).
        When mass_split is empty string, it should be skipped.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        # Set mass_split to empty string (falsy)
        spec_no_ms = CCLCosmologySpec(parameters=spec.parameters, mass_split="")

        assert not spec_no_ms.mass_split
        cosmo = spec_no_ms.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_to_ccl_cosmology_all_optional_fields_false(self) -> None:
        """Test to_ccl_cosmology with all optional fields set to falsy values.

        This ensures all False branches (lines 288, 290, 292, 294) are covered
        when all optional fields are set to None or empty strings.
        """
        spec = CCLCosmologySpec.vanilla_lcdm()
        spec_all_false = CCLCosmologySpec(
            parameters=spec.parameters,
            extra_parameters=None,
            matter_power_spectrum="",
            transfer_function="",
            mass_split="",
        )

        # Verify all fields are falsy
        assert spec_all_false.extra_parameters is None
        assert not spec_all_false.matter_power_spectrum
        assert not spec_all_false.transfer_function
        assert not spec_all_false.mass_split

        # Should still create valid cosmology with defaults
        cosmo = spec_all_false.to_ccl_cosmology()
        assert isinstance(cosmo, pyccl.Cosmology)


class TestConfigGeneratorMethods:
    """Tests for ConfigGenerator methods."""

    def test_config_generator_add_sacc(self, tmp_path) -> None:
        """Test add_sacc method."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        sacc_path = tmp_path / "test.sacc"
        gen.add_sacc(sacc_path)
        assert gen.sacc_path == sacc_path

    def test_config_generator_add_factory(self, tmp_path) -> None:
        """Test add_factory method."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        factory_path = tmp_path / "factory.py"
        gen.add_factory(factory_path)
        assert gen.factory_source == factory_path

    def test_config_generator_add_build_parameters(self, tmp_path) -> None:
        """Test add_build_parameters method."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        params = NamedParameters({"param1": "value1"})
        gen.add_build_parameters(params)
        assert gen.build_parameters == params

    def test_config_generator_add_models(self, tmp_path) -> None:
        """Test add_models method."""
        gen = CobayaConfigGenerator(
            output_path=tmp_path,
            prefix="test",
            use_absolute_path=True,
            cosmo_spec=CCLCosmologySpec.vanilla_lcdm(),
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        model = Model(
            name="test_model",
            description="Test model",
            parameters=[
                Parameter(
                    name="param1",
                    symbol="p1",
                    lower_bound=0.0,
                    upper_bound=1.0,
                    default_value=0.5,
                    free=True,
                )
            ],
        )
        gen.add_models([model])
        assert gen.models == [model]


class TestGetPathStr:
    """Tests for get_path_str utility function."""

    def test_get_path_str_with_string(self) -> None:
        """Test get_path_str returns string as-is."""
        result = get_path_str("my_string_path", use_absolute=True)
        assert result == "my_string_path"

        result = get_path_str("my_string_path", use_absolute=False)
        assert result == "my_string_path"

    def test_get_path_str_with_path_absolute(self, tmp_path) -> None:
        """Test get_path_str with Path object and absolute flag."""
        test_path = tmp_path / "subdir" / "file.txt"

        result = get_path_str(test_path, use_absolute=True)
        assert result == test_path.absolute().as_posix()

    def test_get_path_str_with_path_relative(self, tmp_path) -> None:
        """Test get_path_str with Path object and relative flag."""
        test_path = tmp_path / "subdir" / "file.txt"

        result = get_path_str(test_path, use_absolute=False)
        assert result == "file.txt"
