"""Unit tests for firecrown.app.examples._cmb_cross module.

Tests ExampleCMBCross example generator and the resulting likelihood, built
through Firecrown's YAML-driven TwoPointFactory system.
"""

from pathlib import Path
from unittest.mock import patch

import sacc

from firecrown.app.analysis import (
    FrameworkCosmology,
    Frameworks,
)
from firecrown.app.examples._cmb_cross import (
    BUILD_LIKELIHOOD_FACTORY,
    ExampleCMBCross,
)
from firecrown.likelihood import ConstGaussian, NamedParameters
from firecrown.likelihood.factories import build_two_point_likelihood
from firecrown.metadata_types import CMBLensing, TomographicBin
from firecrown.modeling_tools import ModelingTools


class TestExampleCMBCross:
    """Tests for ExampleCMBCross class."""

    def test_class_attributes(self) -> None:
        """Test that class has required attributes."""
        assert hasattr(ExampleCMBCross, "description")
        assert "CMB" in ExampleCMBCross.description

    def test_default_parameters(self, tmp_path: Path) -> None:
        """Test that default parameters are set correctly."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                target_framework=Frameworks.COSMOSIS,
            )

        assert builder.prefix == "cmb_cross"
        assert builder.seed == 42
        assert builder.n_bins == 2
        assert builder.z_max == 2.0
        assert builder.z_source == 1100.0
        assert builder.include_cmb_auto is True

    def test_custom_parameters(self, tmp_path: Path) -> None:
        """Test that custom parameters are applied."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="custom_cmb",
                seed=123,
                n_bins=3,
                z_max=3.0,
                z_source=1090.0,
                include_cmb_auto=False,
                target_framework=Frameworks.COBAYA,
            )

        assert builder.prefix == "custom_cmb"
        assert builder.seed == 123
        assert builder.n_bins == 3
        assert builder.z_max == 3.0
        assert builder.z_source == 1090.0
        assert builder.include_cmb_auto is False

    def test_generate_sacc_returns_path(self, tmp_path: Path) -> None:
        """Test that generate_sacc returns expected path."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                n_ell_points=5,
                n_z_points=100,
                target_framework=Frameworks.COSMOSIS,
            )

        result = builder.generate_sacc(tmp_path)
        assert result == tmp_path / "test_cmb.sacc"

    def test_generate_sacc_tracers_and_data_types(self, tmp_path: Path) -> None:
        """Test that generate_sacc produces the expected tracers and data types."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                n_bins=2,
                n_ell_points=5,
                n_z_points=100,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_file = builder.generate_sacc(tmp_path)
        sacc_data = sacc.Sacc.load_fits(str(sacc_file))

        assert set(sacc_data.tracers) == {"trc0", "trc1", "cmb_convergence"}
        assert isinstance(sacc_data.tracers["trc0"], sacc.tracers.NZTracer)
        assert isinstance(sacc_data.tracers["cmb_convergence"], sacc.tracers.MapTracer)
        assert sacc_data.tracers["cmb_convergence"].metadata["z_lss"] == 1100.0

        data_types = set(sacc_data.get_data_types())
        assert data_types == {
            "galaxy_shear_cl_ee",
            "cmbGalaxy_convergenceShear_cl_e",
            "cmb_convergence_cl",
        }

    def test_generate_sacc_without_cmb_auto(self, tmp_path: Path) -> None:
        """Test that include_cmb_auto=False omits the CMB auto-correlation."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                n_bins=2,
                n_ell_points=5,
                n_z_points=100,
                include_cmb_auto=False,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_file = builder.generate_sacc(tmp_path)
        sacc_data = sacc.Sacc.load_fits(str(sacc_file))

        data_types = set(sacc_data.get_data_types())
        assert "cmb_convergence_cl" not in data_types

    def test_generate_sacc_extracts_as_projected_fields(self, tmp_path: Path) -> None:
        """The generated SACC file must round-trip through the generalized
        extraction function as a mix of TomographicBin and CMBLensing."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                n_bins=2,
                n_ell_points=5,
                n_z_points=100,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_file = builder.generate_sacc(tmp_path)
        sacc_data = sacc.Sacc.load_fits(str(sacc_file))

        # pylint: disable-next=import-outside-toplevel
        from firecrown.metadata_functions import extract_all_tracers_projected_fields

        all_fields = extract_all_tracers_projected_fields(sacc_data)
        fields_by_name = {f.bin_name: f for f in all_fields}

        assert isinstance(fields_by_name["trc0"], TomographicBin)
        assert isinstance(fields_by_name["cmb_convergence"], CMBLensing)

    def test_generate_factory_writes_yaml_config(self, tmp_path: Path) -> None:
        """Test that generate_factory writes a YAML likelihood configuration
        and returns the YAML-driven factory function reference."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                z_source=1090.0,
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test_cmb.sacc"
        result = builder.generate_factory(tmp_path, sacc_path)

        assert result == BUILD_LIKELIHOOD_FACTORY
        assert result == "firecrown.likelihood.factories.build_two_point_likelihood"

        yaml_file = tmp_path / "test_cmb_experiment.yaml"
        assert yaml_file.is_file()
        content = yaml_file.read_text()
        assert "cmb_factories" in content
        assert "weak_lensing_factories" in content
        assert "z_source: 1090.0" in content
        assert "correlation_space: harmonic" in content

    def test_get_build_parameters(self, tmp_path: Path) -> None:
        """Test that get_build_parameters returns the likelihood_config path."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                target_framework=Frameworks.COSMOSIS,
            )

        sacc_path = tmp_path / "test_cmb.sacc"
        params = builder.get_build_parameters(sacc_path)

        assert isinstance(params, NamedParameters)
        param_dict = params.convert_to_basic_dict()
        assert param_dict["likelihood_config"] == str(
            (tmp_path / "test_cmb_experiment.yaml").absolute()
        )

    def test_get_models_returns_delta_z_params(self, tmp_path: Path) -> None:
        """Test that get_models returns delta_z parameters for photo-z shifts."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                n_bins=2,
                target_framework=Frameworks.COSMOSIS,
            )

        models = builder.get_models()

        assert isinstance(models, list)
        assert len(models) == 1
        param_names = {p.name for p in models[0].parameters}
        assert any("delta_z" in name for name in param_names)

    def test_required_cosmology(self, tmp_path: Path) -> None:
        """Test that required_cosmology returns NONLINEAR."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                target_framework=Frameworks.COSMOSIS,
            )

        result = builder.required_cosmology()
        assert result == FrameworkCosmology.NONLINEAR

    def test_get_options_desc(self, tmp_path: Path) -> None:
        """Test that get_options_desc returns configuration info."""
        with patch.object(ExampleCMBCross, "_proceed_generation"):
            builder = ExampleCMBCross(
                output_path=tmp_path,
                prefix="test_cmb",
                n_bins=3,
                seed=100,
                target_framework=Frameworks.COSMOSIS,
            )

        options = builder.get_options_desc()

        assert isinstance(options, list)
        assert len(options) > 0
        option_names = [name for name, _ in options]
        assert any("bin" in name.lower() for name in option_names)
        assert any("cmb" in name.lower() for name in option_names)

    def test_build_likelihood_execution(self, tmp_path: Path) -> None:
        """Test that the auto-discovered likelihood is built successfully,
        picking up both the galaxy and CMB lensing tracers via the
        WeakLensingFactory/CMBConvergenceFactory dispatch."""
        builder = ExampleCMBCross(
            output_path=tmp_path,
            prefix="test_cmb",
            n_bins=2,
            seed=42,
            n_ell_points=5,
            n_z_points=100,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        factory_ref = builder.generate_factory(tmp_path, sacc_file)
        assert factory_ref == BUILD_LIKELIHOOD_FACTORY
        params = builder.get_build_parameters(sacc_file)

        likelihood, modeling_tools = build_two_point_likelihood(params)

        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(modeling_tools, ModelingTools)

        # n_bins galaxy auto/cross + n_bins CMB-galaxy cross + 1 CMB auto
        expected_n_stats = 2 * (2 + 1) // 2 + 2 + 1
        assert len(likelihood.statistics) == expected_n_stats

    def test_build_likelihood_without_cmb_auto(self, tmp_path: Path) -> None:
        """Test that the auto-discovered likelihood correctly omits the CMB
        auto-correlation when it is not present in the SACC file."""
        builder = ExampleCMBCross(
            output_path=tmp_path,
            prefix="test_cmb",
            n_bins=2,
            seed=42,
            n_ell_points=5,
            n_z_points=100,
            include_cmb_auto=False,
            target_framework=Frameworks.COSMOSIS,
        )

        sacc_file = builder.generate_sacc(tmp_path)
        builder.generate_factory(tmp_path, sacc_file)
        params = builder.get_build_parameters(sacc_file)

        likelihood, _ = build_two_point_likelihood(params)
        assert isinstance(likelihood, ConstGaussian)

        # n_bins galaxy auto/cross + n_bins CMB-galaxy cross, no CMB auto
        expected_n_stats = 2 * (2 + 1) // 2 + 2
        assert len(likelihood.statistics) == expected_n_stats
