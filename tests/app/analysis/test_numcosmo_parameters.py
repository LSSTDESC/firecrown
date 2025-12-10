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
    _set_amplitude_sigma8,
    _set_neutrino_masses,
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

    def test_set_amplitude_a_s_no_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
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
            output_path=tmp_path,
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

    def test_set_amplitude_sigma8_missing_p_ml(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test _set_amplitude_sigma8 raises ValueError when p_ml is missing.

        This test verifies that _set_amplitude_sigma8 raises a ValueError
        (line 343-344) when sigma8 is specified in the cosmology spec but
        the mapping object either doesn't exist or has p_ml set to None.
        """
        assert numcosmo_init

        # Create a cosmology spec with sigma8
        cosmo_with_sigma8 = CCLCosmologySpec.vanilla_lcdm()
        assert "sigma8" in cosmo_with_sigma8

        # Create minimal config options
        config_opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=cosmo_with_sigma8,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create minimal NumCosmo objects
        cosmo = Nc.HICosmoDECpl.new()  # pylint: disable=no-value-for-parameter
        prim = Nc.HIPrimPowerLaw.new()  # pylint: disable=no-value-for-parameter
        priors: list[Ncm.Prior] = []

        # Test with mapping=None
        with pytest.raises(ValueError, match="Mapping must have p_ml set for sigma8"):
            _set_amplitude_sigma8(config_opts, cosmo, prim, None, priors)

        # Test with mapping.p_ml=None
        # Create a mock-like object with p_ml=None
        class MockMapping:
            """Mock mapping with p_ml set to None."""

            p_ml = None

        with pytest.raises(ValueError, match="Mapping must have p_ml set for sigma8"):
            _set_amplitude_sigma8(
                config_opts,
                cosmo,
                prim,
                MockMapping(),  # type: ignore[arg-type]
                priors,
            )


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

    def test_set_neutrino_masses_with_gaussian_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test _set_neutrino_masses with Gaussian prior on m_nu.

        This test verifies that _set_neutrino_masses correctly:
        1. Sets the neutrino mass parameter value
        2. Adds a Gaussian prior to the priors list when m_nu has a prior
        """
        assert numcosmo_init

        # Create cosmology with massive neutrinos and Gaussian prior
        prior_mnu = PriorGaussian(mean=0.06, sigma=0.01)
        params = [
            p if p.name != "m_nu" else p.model_copy(update={"prior": prior_mnu})
            for p in CCLCosmologySpec.vanilla_lcdm_with_neutrinos().parameters
        ]
        cosmo_with_nu_prior = CCLCosmologySpec(parameters=params)

        # Create minimal config options
        config_opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=cosmo_with_nu_prior,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo objects with neutrino support
        num_nu = cosmo_with_nu_prior.get_num_massive_neutrinos()
        cosmo = Nc.HICosmoDECpl(massnu_length=num_nu)
        mset = Ncm.MSet.new_array([cosmo])
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _set_neutrino_masses
        _set_neutrino_masses(config_opts, cosmo, mset, priors)

        # Verify neutrino mass was set
        assert cosmo["massnu_0"] == 0.06

        # Verify prior was added
        assert len(priors) == initial_priors_count + 1
        assert isinstance(priors[-1], Ncm.PriorGauss)

    def test_set_neutrino_masses_with_uniform_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test _set_neutrino_masses with uniform prior on m_nu.

        This test verifies that _set_neutrino_masses correctly adds
        a uniform prior to the priors list when m_nu has a uniform prior.
        """
        assert numcosmo_init

        # Create cosmology with massive neutrinos and uniform prior
        prior_mnu = PriorUniform(lower=0.05, upper=0.15)
        params = [
            p if p.name != "m_nu" else p.model_copy(update={"prior": prior_mnu})
            for p in CCLCosmologySpec.vanilla_lcdm_with_neutrinos().parameters
        ]
        cosmo_with_nu_prior = CCLCosmologySpec(parameters=params)

        # Create minimal config options
        config_opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=cosmo_with_nu_prior,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo objects with neutrino support
        num_nu = cosmo_with_nu_prior.get_num_massive_neutrinos()
        cosmo = Nc.HICosmoDECpl(massnu_length=num_nu)
        mset = Ncm.MSet.new_array([cosmo])
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _set_neutrino_masses
        _set_neutrino_masses(config_opts, cosmo, mset, priors)

        # Verify prior was added
        assert len(priors) == initial_priors_count + 1
        assert isinstance(priors[-1], Ncm.PriorFlat)

    def test_set_neutrino_masses_no_prior(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test _set_neutrino_masses when m_nu has no prior.

        This test verifies that _set_neutrino_masses does not add a prior
        when the m_nu parameter has no prior specified.
        """
        assert numcosmo_init

        # Create cosmology with massive neutrinos but no prior
        cosmo_with_nu = CCLCosmologySpec.vanilla_lcdm_with_neutrinos()

        # Create minimal config options
        config_opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=cosmo_with_nu,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo objects with neutrino support
        num_nu = cosmo_with_nu.get_num_massive_neutrinos()
        cosmo = Nc.HICosmoDECpl(massnu_length=num_nu)
        mset = Ncm.MSet.new_array([cosmo])
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _set_neutrino_masses
        _set_neutrino_masses(config_opts, cosmo, mset, priors)

        # Verify no prior was added
        assert len(priors) == initial_priors_count

    def test_set_neutrino_masses_no_massive_neutrinos(
        self, numcosmo_init: bool, tmp_path: Path
    ) -> None:
        """Test _set_neutrino_masses with no massive neutrinos.

        This test verifies that _set_neutrino_masses returns early
        when there are no massive neutrinos in the cosmology.
        """
        assert numcosmo_init

        # Create cosmology without massive neutrinos
        vanilla_cosmo = CCLCosmologySpec.vanilla_lcdm()

        # Create minimal config options
        config_opts = ConfigOptions(
            output_path=tmp_path,
            factory_source=Path("factory.py"),
            build_parameters=NamedParameters({}),
            models=[],
            cosmo_spec=vanilla_cosmo,
            use_absolute_path=True,
            required_cosmology=FrameworkCosmology.NONLINEAR,
            prefix="test",
        )

        # Create NumCosmo objects
        cosmo = Nc.HICosmoDECpl.new()  # pylint: disable=no-value-for-parameter
        mset = Ncm.MSet.new_array([cosmo])
        priors: list[Ncm.Prior] = []
        initial_priors_count = len(priors)

        # Call _set_neutrino_masses - should return early
        _set_neutrino_masses(config_opts, cosmo, mset, priors)

        # Verify nothing was changed
        assert len(priors) == initial_priors_count


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
