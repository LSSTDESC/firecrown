"""Unit tests for NumCosmo parameter handling.

Tests for amplitude parameters (A_s, sigma8), neutrino parameters,
and cosmology specification handling in firecrown.app.analysis._numcosmo module.
"""

from pathlib import Path
import numpy as np
import pytest

from numcosmo_py import Ncm, Nc

from firecrown.likelihood import NamedParameters
from firecrown.app.analysis._numcosmo import (
    ConfigOptions,
    NumCosmoConfigGenerator,
    _set_amplitude_A_s,
)
from firecrown.app.analysis._types import (
    FrameworkCosmology,
    CCLCosmologySpec,
    Parameter,
    PriorGaussian,
    PriorUniform,
)


@pytest.fixture(name="numcosmo_init", scope="session")
def fixture_numcosmo_init() -> bool:
    """Fixture to initialize NumCosmo for testing."""
    Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

    return True


@pytest.fixture(name="vanilla_cosmo")
def fixture_vanilla_cosmo() -> CCLCosmologySpec:
    """Create vanilla LCDM cosmology spec for testing."""
    return CCLCosmologySpec.vanilla_lcdm()


class TestAmplitudeParameterHandling:
    """Tests for A_s and sigma8 amplitude parameter handling."""

    def test_set_amplitude_a_s_no_prior(self, numcosmo_init: bool) -> None:
        """Test _set_amplitude_A_s when A_s parameter has no prior.

        This test verifies that _set_amplitude_A_s returns early (line 292-293)
        when the A_s parameter exists but has no prior (prior is None).
        The function should still set the parameter value but not add any priors.
        """
        assert numcosmo_init

        # Create a cosmology spec with A_s but no prior
        params = [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "sigma8"
        ] + [
            Parameter(
                name="A_s",
                symbol="A_s",
                lower_bound=1e-9,
                upper_bound=3e-9,
                default_value=2e-9,
                free=True,
                prior=None,  # No prior
            )
        ]
        cosmo_with_as = CCLCosmologySpec(parameters=params)

        # Create minimal config options
        config_opts = ConfigOptions(
            output_path=Path("/tmp"),
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=cosmo_with_as,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create minimal NumCosmo objects with proper hierarchy
        cosmo = Nc.HICosmoDECpl.new()  # pylint: disable=no-value-for-parameter
        prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
        cosmo.add_submodel(prim)
        mset = Ncm.MSet.new_array([cosmo])

        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _set_amplitude_A_s
        _set_amplitude_A_s(config_opts, mset, prim, priors)

        # Verify A_s was set in the prim model
        expected_ln_value = np.log(1.0e10 * 2e-9)
        actual_ln_value = prim["ln10e10ASA"]
        assert np.isclose(actual_ln_value, expected_ln_value)

        # Verify no prior was added (early return at line 292-293)
        assert len(priors) == initial_priors_count


class TestAmplitudeParametersGenerator:
    """Tests for A_s and sigma8 parameter handling in generator."""

    def test_generator_with_a_s_parameter(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with A_s instead of sigma8."""
        assert numcosmo_init

        # Create cosmology with A_s
        params = [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "sigma8"
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
        cosmo_with_as = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_as",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert "A_s" in gen.cosmo_spec
        assert "sigma8" not in gen.cosmo_spec

    def test_generator_with_a_s_gaussian_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with Gaussian prior on A_s."""
        assert numcosmo_init

        prior_as = PriorGaussian(mean=2e-9, sigma=0.1e-9)
        params = [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "sigma8"
        ] + [
            Parameter(
                name="A_s",
                symbol="A_s",
                lower_bound=1e-9,
                upper_bound=3e-9,
                default_value=2e-9,
                free=True,
                prior=prior_as,
            )
        ]
        cosmo_with_as = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_as_prior",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec["A_s"].prior is not None

    def test_generator_with_a_s_uniform_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with uniform prior on A_s."""
        assert numcosmo_init

        prior_as = PriorUniform(lower=1.5e-9, upper=2.5e-9)
        params = [
            p for p in CCLCosmologySpec.vanilla_lcdm().parameters if p.name != "sigma8"
        ] + [
            Parameter(
                name="A_s",
                symbol="A_s",
                lower_bound=1e-9,
                upper_bound=3e-9,
                default_value=2e-9,
                free=True,
                prior=prior_as,
            )
        ]
        cosmo_with_as = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_as_uniform",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_as,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec["A_s"].prior is not None


class TestNeutrinoHandling:
    """Tests for massive neutrino parameter handling."""

    def test_generator_with_massive_neutrinos(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with massive neutrinos."""
        assert numcosmo_init

        cosmo_with_nu = CCLCosmologySpec.vanilla_lcdm_with_neutrinos()

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_neutrinos",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_nu,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec.get_num_massive_neutrinos() > 0

    def test_generator_with_neutrino_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test generator with prior on neutrino mass."""
        assert numcosmo_init

        prior_mnu = PriorGaussian(mean=0.06, sigma=0.01)
        params = [
            p if p.name != "m_nu" else p.model_copy(update={"prior": prior_mnu})
            for p in CCLCosmologySpec.vanilla_lcdm_with_neutrinos().parameters
        ]
        cosmo_with_nu_prior = CCLCosmologySpec(parameters=params)

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_nu_prior",
            use_absolute_path=True,
            cosmo_spec=cosmo_with_nu_prior,
            required_cosmology=FrameworkCosmology.NONLINEAR,
        )

        assert gen.cosmo_spec["m_nu"].prior is not None


class TestCosmologyNone:
    """Tests for NONE cosmology framework."""

    def test_generator_with_none_cosmology(
        self, numcosmo_init: bool, tmp_path: Path, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test generator with NONE cosmology framework."""
        assert numcosmo_init

        gen = NumCosmoConfigGenerator(
            output_path=tmp_path,
            prefix="test_none",
            use_absolute_path=True,
            cosmo_spec=vanilla_cosmo,
            required_cosmology=FrameworkCosmology.NONE,
        )

        assert gen.required_cosmology == FrameworkCosmology.NONE


class TestCosmologySpecHandling:
    """Tests for cosmology specification handling."""

    def test_vanilla_lcdm_parameters(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test vanilla LCDM has expected parameters."""
        param_names = {p.name for p in vanilla_cosmo.parameters}

        # Should have minimal required set
        assert "Omega_c" in param_names
        assert "Omega_b" in param_names
        assert "h" in param_names
        assert "n_s" in param_names

        # Should have either sigma8 or A_s
        assert ("sigma8" in param_names) or ("A_s" in param_names)

    def test_cosmology_parameter_access(self, vanilla_cosmo: CCLCosmologySpec) -> None:
        """Test accessing cosmology parameters."""
        assert "Omega_c" in vanilla_cosmo
        omega_c = vanilla_cosmo["Omega_c"]
        assert omega_c.name == "Omega_c"

    def test_cosmology_num_massive_neutrinos(
        self, vanilla_cosmo: CCLCosmologySpec
    ) -> None:
        """Test getting number of massive neutrinos."""
        num_nu = vanilla_cosmo.get_num_massive_neutrinos()
        assert isinstance(num_nu, int)
        assert num_nu >= 0
