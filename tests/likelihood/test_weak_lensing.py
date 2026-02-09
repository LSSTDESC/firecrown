"""Tests for the weak lensing source and systematics."""

import pytest
import numpy as np
import sacc
from unittest.mock import Mock, patch
import pyccl
import pyccl.nl_pt

import firecrown.likelihood._weak_lensing as wl
from firecrown.metadata_types import InferredGalaxyZDist, Galaxies
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap


@pytest.fixture
def mock_tools():
    """Create a mock ModelingTools object."""
    tools = Mock(spec=ModelingTools)
    mock_cosmo = Mock(spec=pyccl.Cosmology)
    mock_cosmo.__getitem__ = Mock(return_value=0.3)
    tools.get_ccl_cosmology.return_value = mock_cosmo

    # Mock HM calculator
    mock_hm_calc = Mock()
    mock_hm_calc.mass_def = "200m"
    tools.get_hm_calculator.return_value = mock_hm_calc

    # Mock concentration-mass relation
    tools.get_cM_relation.return_value = "Duffy08"

    return tools


@pytest.fixture
def sample_tracer_args():
    """Create sample WeakLensingArgs for testing."""
    z = np.linspace(0.1, 2.0, 50)
    dndz = np.exp(-(((z - 0.5) / 0.2) ** 2))
    return wl.WeakLensingArgs(
        z=z,
        dndz=dndz,
        scale=1.0,
        ia_bias=None,
        has_pt=False,
        has_hm=False,
    )


@pytest.fixture
def mock_sacc_data():
    """Create a mock SACC data object."""
    mock_sacc = Mock(spec=sacc.Sacc)
    mock_tracer = Mock()
    mock_tracer.z = np.linspace(0.1, 2.0, 50)
    mock_tracer.nz = np.exp(-(((mock_tracer.z - 0.5) / 0.2) ** 2))
    mock_sacc.get_tracer.return_value = mock_tracer
    return mock_sacc


def test_multiplicative_shear_bias_init():
    """Test MultiplicativeShearBias initialization."""
    systematic = wl.MultiplicativeShearBias("test_tracer")
    assert hasattr(systematic, "mult_bias")
    assert hasattr(systematic, "parameter_prefix")


def test_multiplicative_shear_bias_apply(mock_tools, sample_tracer_args):
    """Test MultiplicativeShearBias apply method - covers line 112."""
    systematic = wl.MultiplicativeShearBias("test_tracer")
    # Update parameter values using ParamsMap
    params = ParamsMap({"test_tracer_mult_bias": 0.1})
    systematic.update(params)

    result = systematic.apply(mock_tools, sample_tracer_args)

    # Should multiply scale by (1 + mult_bias)
    expected_scale = sample_tracer_args.scale * (1.0 + 0.1)
    assert result.scale == expected_scale
    assert result.z is sample_tracer_args.z
    assert result.dndz is sample_tracer_args.dndz


def test_linear_alignment_systematic_init_default():
    """Test LinearAlignmentSystematic default initialization."""
    systematic = wl.LinearAlignmentSystematic()
    assert hasattr(systematic, "ia_bias")
    assert hasattr(systematic, "alphaz")
    assert hasattr(systematic, "alphag")
    assert hasattr(systematic, "z_piv")


def test_linear_alignment_systematic_init_with_tracer():
    """Test LinearAlignmentSystematic initialization with tracer."""
    systematic = wl.LinearAlignmentSystematic("test_tracer", alphag=2.0)
    assert hasattr(systematic, "ia_bias")


@patch("pyccl.growth_factor")
def test_linear_alignment_systematic_apply(
    mock_growth_factor, mock_tools, sample_tracer_args
):
    mock_growth_factor.return_value = 0.8

    systematic = wl.LinearAlignmentSystematic("test_tracer")
    # Update parameter values using ParamsMap
    params = ParamsMap(
        {
            "test_tracer_ia_bias": 1.0,
            "test_tracer_alphaz": 0.5,
            "test_tracer_alphag": 1.2,
            "test_tracer_z_piv": 0.3,
        }
    )
    systematic.update(params)

    result = systematic.apply(mock_tools, sample_tracer_args)

    # Check that ia_bias is set with correct structure
    assert result.ia_bias is not None
    assert len(result.ia_bias) == 2
    assert np.array_equal(result.ia_bias[0], sample_tracer_args.z)

    # Verify that growth_factor was called
    mock_growth_factor.assert_called()


def test_massdep_linear_alignment_systematic_init_default():
    """Test MassDependentLinearAlignmentSystematic default initialization."""
    systematic = wl.MassDependentLinearAlignmentSystematic()
    assert hasattr(systematic, "ia_amplitude")
    assert hasattr(systematic, "ia_mass_scaling")
    assert hasattr(systematic, "red_fraction")
    assert hasattr(systematic, "average_halo_mass")


def test_massdep_linear_alignment_systematic_apply(mock_tools, sample_tracer_args):
    systematic = wl.MassDependentLinearAlignmentSystematic("test_tracer")
    # Update parameter values using ParamsMap
    params = ParamsMap(
        {
            "ia_amplitude": 6.0,
            "ia_mass_scaling": 0.5,
            "test_tracer_red_fraction": 0.3,
            "test_tracer_average_halo_mass": 1e13,
        }
    )
    systematic.update(params)

    result = systematic.apply(mock_tools, sample_tracer_args)

    # Check that ia_bias is set with correct structure
    assert result.ia_bias is not None
    assert len(result.ia_bias) == 2
    assert np.array_equal(result.ia_bias[0], sample_tracer_args.z)


def test_tatt_alignment_systematic_init_default():
    """Test TattAlignmentSystematic default initialization."""
    systematic = wl.TattAlignmentSystematic()
    assert hasattr(systematic, "ia_a_1")
    assert hasattr(systematic, "ia_a_2")
    assert hasattr(systematic, "ia_a_d")


def test_tatt_alignment_systematic_init_with_z_dependence():
    """Test TattAlignmentSystematic with z dependence enabled."""
    systematic = wl.TattAlignmentSystematic(include_z_dependence=True)
    assert hasattr(systematic, "ia_a_1")


@patch("pyccl.nl_pt.translate_IA_norm")
def test_tatt_alignment_systematic_apply(
    mock_translate_ia, mock_tools, sample_tracer_args
):
    """Test TattAlignmentSystematic apply method."""
    # Mock the translate_IA_norm function return values
    c_1 = np.ones_like(sample_tracer_args.z) * 0.1
    c_d = np.ones_like(sample_tracer_args.z) * 0.2
    c_2 = np.ones_like(sample_tracer_args.z) * 0.3
    mock_translate_ia.return_value = (c_1, c_d, c_2)

    systematic = wl.TattAlignmentSystematic("test_tracer")
    # Update parameter values using ParamsMap
    params = ParamsMap(
        {
            "test_tracer_ia_a_1": 1.0,
            "test_tracer_ia_a_2": 0.5,
            "test_tracer_ia_a_d": 0.3,
            "test_tracer_ia_zpiv_1": 0.5,
            "test_tracer_ia_zpiv_2": 0.5,
            "test_tracer_ia_zpiv_d": 0.5,
            "test_tracer_ia_alphaz_1": 0.0,
            "test_tracer_ia_alphaz_2": 0.0,
            "test_tracer_ia_alphaz_d": 0.0,
        }
    )
    systematic.update(params)

    result = systematic.apply(mock_tools, sample_tracer_args)

    # Check that PT is enabled and coefficients are set
    assert result.has_pt is True
    assert result.ia_pt_c_1 is not None
    assert result.ia_pt_c_d is not None
    assert result.ia_pt_c_2 is not None

    # Verify the structure of the PT coefficients
    assert len(result.ia_pt_c_1) == 2
    assert len(result.ia_pt_c_d) == 2
    assert len(result.ia_pt_c_2) == 2


def test_hm_alignment_systematic_init():
    """Test HMAlignmentSystematic initialization - covers lines 326, 328, 331."""
    systematic = wl.HMAlignmentSystematic()
    assert hasattr(systematic, "ia_a_1h")
    assert hasattr(systematic, "ia_a_2h")


def test_hm_alignment_systematic_init_with_tracer():
    """Test HMAlignmentSystematic initialization with ignored tracer parameter."""
    systematic = wl.HMAlignmentSystematic("ignored_tracer")
    assert hasattr(systematic, "ia_a_1h")
    assert hasattr(systematic, "ia_a_2h")


def test_hm_alignment_systematic_apply(mock_tools, sample_tracer_args):
    """Test HMAlignmentSystematic apply method - covers line 344."""
    systematic = wl.HMAlignmentSystematic()
    # Update parameter values using ParamsMap
    params = ParamsMap(
        {
            "ia_a_1h": 1e-4,
            "ia_a_2h": 1.0,
        }
    )
    systematic.update(params)

    result = systematic.apply(mock_tools, sample_tracer_args)

    # Check that HM is enabled and amplitudes are set
    assert result.has_hm is True
    assert result.ia_a_1h == 1e-4
    assert result.ia_a_2h == 1.0


class TestWeakLensing:
    """Test WeakLensing source class."""

    def test_init(self):
        """Test WeakLensing initialization."""
        weak_lensing = wl.WeakLensing(sacc_tracer="test_tracer", scale=2.0)
        assert weak_lensing.sacc_tracer == "test_tracer"
        assert weak_lensing.scale == 2.0
        assert weak_lensing.current_tracer_args is None

    def test_create_ready(self):
        """Test WeakLensing.create_ready class method."""
        zdist = InferredGalaxyZDist(
            bin_name="test_bin",
            z=np.linspace(0.1, 2.0, 50),
            dndz=np.exp(-(((np.linspace(0.1, 2.0, 50) - 0.5) / 0.2) ** 2)),
            measurements={Galaxies.SHEAR_E},
        )

        weak_lensing = wl.WeakLensing.create_ready(zdist)
        assert weak_lensing.sacc_tracer == "test_bin"
        assert hasattr(weak_lensing, "tracer_args")
        assert weak_lensing.tracer_args.ia_bias is None

    def test_read(self, mock_sacc_data):
        """Test WeakLensing _read method."""
        weak_lensing = wl.WeakLensing(sacc_tracer="test_tracer")
        weak_lensing._read(mock_sacc_data)

        assert hasattr(weak_lensing, "tracer_args")
        assert weak_lensing.tracer_args.scale == weak_lensing.scale
        mock_sacc_data.get_tracer.assert_called_with("test_tracer")

    @patch("pyccl.WeakLensingTracer")
    def test_create_tracers_basic(self, mock_wl_tracer, mock_tools):
        """Test WeakLensing create_tracers with basic configuration."""
        mock_ccl_tracer = Mock()
        mock_wl_tracer.return_value = mock_ccl_tracer

        weak_lensing = wl.WeakLensing(sacc_tracer="test_tracer", systematics=[])
        weak_lensing.tracer_args = wl.WeakLensingArgs(
            z=np.linspace(0.1, 2.0, 50),
            dndz=np.exp(-(((np.linspace(0.1, 2.0, 50) - 0.5) / 0.2) ** 2)),
            scale=1.0,
        )

        tracers, tracer_args = weak_lensing.create_tracers(mock_tools)

        assert len(tracers) == 1
        assert tracers[0].tracer_name == "shear"
        assert weak_lensing.current_tracer_args is not None

    @patch("pyccl.nl_pt.PTIntrinsicAlignmentTracer")
    @patch("pyccl.WeakLensingTracer")
    def test_create_tracers_with_pt(self, mock_wl_tracer, mock_pt_tracer, mock_tools):
        """Test WeakLensing create_tracers with PT - covers lines 450-452, 455."""
        mock_ccl_tracer = Mock()
        mock_wl_tracer.return_value = mock_ccl_tracer
        mock_pt_tracer.return_value = Mock()

        weak_lensing = wl.WeakLensing(sacc_tracer="test_tracer", systematics=[])
        z = np.linspace(0.1, 2.0, 50)
        c_vals = np.ones_like(z) * 0.1
        weak_lensing.tracer_args = wl.WeakLensingArgs(
            z=z,
            dndz=np.exp(-(((z - 0.5) / 0.2) ** 2)),
            scale=1.0,
            has_pt=True,
            ia_pt_c_1=(z, c_vals),
            ia_pt_c_d=(z, c_vals),
            ia_pt_c_2=(z, c_vals),
        )

        tracers, tracer_args = weak_lensing.create_tracers(mock_tools)

        # Should have 2 tracers: shear + intrinsic_pt
        assert len(tracers) == 2
        assert tracers[0].tracer_name == "shear"
        assert tracers[1].tracer_name == "intrinsic_pt"

        # Verify PT tracer was created
        mock_pt_tracer.assert_called_once()

    @patch("pyccl.halos.SatelliteShearHOD")
    @patch("pyccl.WeakLensingTracer")
    def test_create_tracers_with_hm(
        self, mock_wl_tracer, mock_halo_profile, mock_tools
    ):
        """Test WeakLensing create_tracers with HM - covers lines 462, 469, 472."""
        mock_ccl_tracer = Mock()
        mock_wl_tracer.return_value = mock_ccl_tracer
        mock_profile = Mock()
        mock_halo_profile.return_value = mock_profile

        weak_lensing = wl.WeakLensing(sacc_tracer="test_tracer", systematics=[])
        z = np.linspace(0.1, 2.0, 50)
        weak_lensing.tracer_args = wl.WeakLensingArgs(
            z=z,
            dndz=np.exp(-(((z - 0.5) / 0.2) ** 2)),
            scale=1.0,
            has_hm=True,
            ia_a_1h=np.array([1e-4]),
            ia_a_2h=np.array([1.0]),
        )

        tracers, tracer_args = weak_lensing.create_tracers(mock_tools)

        # Should have 2 tracers: shear + intrinsic_alignment_hm
        assert len(tracers) == 2
        assert tracers[0].tracer_name == "shear"
        assert tracers[1].tracer_name == "intrinsic_alignment_hm"

        # Verify halo profile was created and configured
        mock_halo_profile.assert_called_once()
        # Check that ia_a_2h was attached to the profile
        assert hasattr(mock_profile, "ia_a_2h")
        assert mock_profile.ia_a_2h == 1.0

    def test_get_scale(self):
        """Test WeakLensing get_scale method."""
        weak_lensing = wl.WeakLensing(sacc_tracer="test_tracer", scale=2.0)
        weak_lensing.current_tracer_args = wl.WeakLensingArgs(
            z=np.linspace(0.1, 2.0, 50),
            dndz=np.exp(-(((np.linspace(0.1, 2.0, 50) - 0.5) / 0.2) ** 2)),
            scale=3.0,
        )

        assert weak_lensing.get_scale() == 3.0

    def test_update_with_systematics(self):
        """Test WeakLensing update method with systematics - covers line 398."""
        systematic = wl.MultiplicativeShearBias("test_tracer")
        weak_lensing = wl.WeakLensing(
            sacc_tracer="test_tracer", systematics=[systematic]
        )

        params = ParamsMap({"test_tracer_mult_bias": 0.1})
        weak_lensing.update(params)

        # Should not raise any exceptions
        assert weak_lensing.systematics is not None

    @patch("pyccl.WeakLensingTracer")
    def test_create_tracers_with_systematics(self, mock_wl_tracer, mock_tools):
        """Test WeakLensing create_tracers with systematics - covers line 421."""
        mock_ccl_tracer = Mock()
        mock_wl_tracer.return_value = mock_ccl_tracer

        systematic = wl.MultiplicativeShearBias("test_tracer")
        weak_lensing = wl.WeakLensing(
            sacc_tracer="test_tracer", systematics=[systematic]
        )
        weak_lensing.tracer_args = wl.WeakLensingArgs(
            z=np.linspace(0.1, 2.0, 50),
            dndz=np.exp(-(((np.linspace(0.1, 2.0, 50) - 0.5) / 0.2) ** 2)),
            scale=1.0,
        )

        # Update systematic parameters first
        params = ParamsMap({"test_tracer_mult_bias": 0.1})
        systematic.update(params)

        tracers, tracer_args = weak_lensing.create_tracers(mock_tools)

        assert len(tracers) == 1
        assert tracers[0].tracer_name == "shear"
        assert weak_lensing.current_tracer_args is not None
        # The scale should have been modified by the systematic
        assert weak_lensing.current_tracer_args.scale == 1.1  # 1.0 * (1 + 0.1)


def test_multiplicative_shear_bias_factory():
    """Test MultiplicativeShearBiasFactory."""
    factory = wl.MultiplicativeShearBiasFactory()
    bias = factory.create("test_bin")
    assert isinstance(bias, wl.MultiplicativeShearBias)


def test_multiplicative_shear_bias_factory_global_error():
    factory = wl.MultiplicativeShearBiasFactory()
    with pytest.raises(ValueError, match="MultiplicativeShearBias cannot be global"):
        factory.create_global()


def test_linear_alignment_factory():
    """Test LinearAlignmentSystematicFactory."""
    factory = wl.LinearAlignmentSystematicFactory(alphag=2.0)
    systematic = factory.create("test_bin")
    assert isinstance(systematic, wl.LinearAlignmentSystematic)

    global_systematic = factory.create_global()
    assert isinstance(global_systematic, wl.LinearAlignmentSystematic)


def test_tatt_alignment_factory():
    """Test TattAlignmentSystematicFactory."""
    factory = wl.TattAlignmentSystematicFactory(include_z_dependence=True)
    systematic = factory.create("test_bin")
    assert isinstance(systematic, wl.TattAlignmentSystematic)

    global_systematic = factory.create_global()
    assert isinstance(global_systematic, wl.TattAlignmentSystematic)


def test_weak_lensing_factory():
    """Test WeakLensingFactory."""
    factory = wl.WeakLensingFactory(
        per_bin_systematics=[wl.MultiplicativeShearBiasFactory()],
        global_systematics=[wl.LinearAlignmentSystematicFactory()],
    )

    zdist = InferredGalaxyZDist(
        bin_name="test_bin",
        z=np.linspace(0.1, 2.0, 50),
        dndz=np.exp(-(((np.linspace(0.1, 2.0, 50) - 0.5) / 0.2) ** 2)),
        measurements={Galaxies.SHEAR_E},
    )

    weak_lensing = factory.create(zdist)
    assert isinstance(weak_lensing, wl.WeakLensing)
    assert weak_lensing.sacc_tracer == "test_bin"

    # Test caching
    wl2 = factory.create(zdist)
    assert weak_lensing is wl2


def test_weak_lensing_factory_from_metadata():
    """Test WeakLensingFactory create_from_metadata_only."""
    factory = wl.WeakLensingFactory()

    weak_lensing = factory.create_from_metadata_only("test_tracer")
    assert isinstance(weak_lensing, wl.WeakLensing)
    assert weak_lensing.sacc_tracer == "test_tracer"

    # Test caching
    wl2 = factory.create_from_metadata_only("test_tracer")
    assert weak_lensing is wl2


def test_photoz_shift():
    """Test PhotoZShift systematic."""
    systematic = wl.PhotoZShift("test_tracer")
    assert hasattr(systematic, "delta_z")


def test_photoz_shift_and_stretch():
    """Test PhotoZShiftandStretch systematic."""
    systematic = wl.PhotoZShiftandStretch("test_tracer")
    assert hasattr(systematic, "delta_z")
    assert hasattr(systematic, "sigma_z")


def test_select_field():
    """Test SelectField systematic."""
    systematic = wl.SelectField()
    assert hasattr(systematic, "field")


def test_weak_lensing_args_immutable():
    """Test that WeakLensingArgs is immutable (frozen dataclass)."""
    args = wl.WeakLensingArgs(
        z=np.array([0.1, 0.5, 1.0]),
        dndz=np.array([0.1, 0.5, 0.1]),
        scale=1.0,
    )

    # Should not be able to modify after creation
    with pytest.raises(
        Exception
    ):  # Could be AttributeError or dataclasses.FrozenInstanceError
        args.scale = 2.0  # type: ignore[misc]


def test_systematic_with_none_tracer():
    """Test systematics with None tracer name."""
    # LinearAlignmentSystematic should accept None tracer
    systematic = wl.LinearAlignmentSystematic(sacc_tracer=None)
    assert hasattr(systematic, "ia_bias")

    # TattAlignmentSystematic should accept None tracer
    systematic2 = wl.TattAlignmentSystematic(sacc_tracer=None)
    assert hasattr(systematic2, "ia_a_1")


@patch("pyccl.growth_factor")
def test_linear_alignment_complex_calculation(mock_growth_factor, mock_tools):
    """Test LinearAlignmentSystematic with complex parameter values."""
    mock_growth_factor.return_value = np.array([0.7, 0.8, 0.9])

    systematic = wl.LinearAlignmentSystematic("test_tracer")
    # Update parameter values using ParamsMap
    params = ParamsMap(
        {
            "test_tracer_ia_bias": 2.0,
            "test_tracer_alphaz": 1.5,
            "test_tracer_alphag": 0.5,
            "test_tracer_z_piv": 1.0,
        }
    )
    systematic.update(params)

    z = np.array([0.5, 1.0, 1.5])
    tracer_args = wl.WeakLensingArgs(
        z=z,
        dndz=np.array([0.1, 0.5, 0.1]),
        scale=1.0,
    )

    result = systematic.apply(mock_tools, tracer_args)

    # Verify that the calculation was performed
    assert result.ia_bias is not None
    assert len(result.ia_bias) == 2
    assert np.array_equal(result.ia_bias[0], z)
    # The second element should be the calculated ia_bias_array
    assert isinstance(result.ia_bias[1], np.ndarray)
    assert len(result.ia_bias[1]) == len(z)
