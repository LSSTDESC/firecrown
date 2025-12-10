"""Unit tests for firecrown.app.experiment command module.

Tests experiment file loading, viewing, and error handling.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import yaml

from firecrown.app.experiment import Load, View

# pylint: disable=unused-argument
# pylint: disable=protected-access


@pytest.fixture
def sample_experiment_yaml(tmp_path: Path) -> Path:
    """Create a sample experiment YAML file for testing."""
    experiment_data = {
        "name": "test_experiment",
        "data_source": {
            "sacc_data_file": str(tmp_path / "data.sacc"),
            "filters": None,
        },
        "two_point_factory": {
            "module": "firecrown.likelihood.factories",
            "class": "TwoPointFactory",
            "weak_lensing_factories": [],
            "number_counts_factories": [],
            "cmb_factories": [],
        },
    }

    experiment_file = tmp_path / "experiment.yaml"
    experiment_file.write_text(yaml.dump(experiment_data))
    return experiment_file


class TestLoadInitialization:
    """Tests for Load command initialization and file loading."""

    def test_load_requires_experiment_file(self, tmp_path: Path) -> None:
        """Test that Load requires experiment_file parameter."""
        # Create a minimal valid experiment file
        exp_file = tmp_path / "test.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc"},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch("firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"):
            load = Load(experiment_file=exp_file)
            assert load.experiment_file == exp_file

    def test_load_stores_experiment_file_path(self, tmp_path: Path) -> None:
        """Test that Load stores the experiment file path."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc"},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch("firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"):
            load = Load(experiment_file=exp_file)
            assert load.experiment_file == exp_file

    def test_load_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading a non-existent file raises BadParameter."""
        exp_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(Exception, match="Experiment file not found"):
            Load(experiment_file=exp_file)

    def test_load_calls_factory_load_from_yaml(self, tmp_path: Path) -> None:
        """Test that Load calls TwoPointExperiment.load_from_yaml."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc"},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            Load(experiment_file=exp_file)
            mock_load.assert_called_once_with(exp_file)


class TestLoadExperiment:
    """Tests for Load._load_experiment method."""

    def test_load_experiment_sets_tp_experiment_attribute(self, tmp_path: Path) -> None:
        """Test that _load_experiment sets tp_experiment attribute."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc"},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            load = Load(experiment_file=exp_file)
            assert hasattr(load, "tp_experiment")
            assert load.tp_experiment == mock_load.return_value

    def test_load_experiment_invalid_yaml_raises_exception(
        self, tmp_path: Path
    ) -> None:
        """Test that invalid YAML content raises an exception."""
        exp_file = tmp_path / "invalid.yaml"
        exp_file.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):
            Load(experiment_file=exp_file)

    def test_load_experiment_prints_file_path(self, tmp_path: Path, capsys) -> None:
        """Test that load prints the file path to console."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc"},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch("firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"):
            Load(experiment_file=exp_file)
            captured = capsys.readouterr()
            one_line_output = captured.out.replace("\n", " ")
            assert str(exp_file) in one_line_output


class TestViewInitialization:
    """Tests for View command initialization."""

    def test_view_inherits_from_load(self) -> None:
        """Test that View is a subclass of Load."""
        assert issubclass(View, Load)

    def test_view_calls_print_factories(self, tmp_path: Path) -> None:
        """Test that View calls _print_factories during initialization."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            # Mock the experiment structure
            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = []
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            view = View(experiment_file=exp_file)
            # View should have loaded and printed
            assert hasattr(view, "tp_experiment")


class TestViewPrintFactories:
    """Tests for View._print_factories method."""

    def test_print_factories_displays_data_source(self, tmp_path: Path, capsys) -> None:
        """Test that _print_factories displays data source information."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = []
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "Data Source" in captured.out
            assert "data.sacc" in captured.out

    def test_print_factories_displays_filters(self, tmp_path: Path, capsys) -> None:
        """Test that filters are displayed when present."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {
                "sacc_data_file": "data.sacc",
                "filters": "scale_cuts.yaml",
            },
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = []
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = "scale_cuts.yaml"

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "Filters:" in captured.out
            assert "scale_cuts.yaml" in captured.out

    def test_print_factories_displays_weak_lensing(
        self, tmp_path: Path, capsys
    ) -> None:
        """Test that weak lensing factories are displayed."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            # Create mock weak lensing factory
            mock_wl = MagicMock()
            mock_wl.type_source = "src0"
            mock_wl.per_bin_systematics = []
            mock_wl.global_systematics = []

            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = [mock_wl]
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "WeakLensing" in captured.out
            assert "src0" in captured.out

    def test_print_factories_displays_number_counts(
        self, tmp_path: Path, capsys
    ) -> None:
        """Test that number counts factories are displayed."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            mock_nc = MagicMock()
            mock_nc.type_source = "lens0"
            mock_nc.per_bin_systematics = []
            mock_nc.global_systematics = []

            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = []
            mock_experiment.two_point_factory.number_counts_factories = [mock_nc]
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "NumberCounts" in captured.out
            assert "lens0" in captured.out

    def test_print_factories_displays_cmb(self, tmp_path: Path, capsys) -> None:
        """Test that CMB factories are displayed."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            mock_cmb = MagicMock()
            mock_cmb.type_source = "cmb_convergence"

            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = []
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = [mock_cmb]
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "CMBConvergence" in captured.out
            assert "cmb_convergence" in captured.out

    def test_print_factories_displays_systematics(self, tmp_path: Path, capsys) -> None:
        """Test that systematics are displayed."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            mock_systematic = MagicMock()
            mock_systematic.type = "multiplicative_bias"

            mock_wl = MagicMock()
            mock_wl.type_source = "src0"
            mock_wl.per_bin_systematics = [mock_systematic]
            mock_wl.global_systematics = []

            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = [mock_wl]
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "multiplicative_bias" in captured.out

    def test_print_factories_empty_systematics_shows_dash(
        self, tmp_path: Path, capsys
    ) -> None:
        """Test that empty systematics list displays a dash."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            mock_wl = MagicMock()
            mock_wl.type_source = "src0"
            mock_wl.per_bin_systematics = []
            mock_wl.global_systematics = []

            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = [mock_wl]
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            # The table should have dash for empty systematics
            assert "WeakLensing" in captured.out

    def test_print_factories_multiple_factories(self, tmp_path: Path, capsys) -> None:
        """Test displaying multiple factories of different types."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            mock_wl1 = MagicMock()
            mock_wl1.type_source = "src0"
            mock_wl1.per_bin_systematics = []
            mock_wl1.global_systematics = []

            mock_wl2 = MagicMock()
            mock_wl2.type_source = "src1"
            mock_wl2.per_bin_systematics = []
            mock_wl2.global_systematics = []

            mock_nc = MagicMock()
            mock_nc.type_source = "lens0"
            mock_nc.per_bin_systematics = []
            mock_nc.global_systematics = []

            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = [
                mock_wl1,
                mock_wl2,
            ]
            mock_experiment.two_point_factory.number_counts_factories = [mock_nc]
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "src0" in captured.out
            assert "src1" in captured.out
            assert "lens0" in captured.out


class TestFormatSysFunction:
    """Tests for the fmt_sys helper function within _print_factories."""

    def test_fmt_sys_with_multiple_systematics(self, tmp_path: Path, capsys) -> None:
        """Test that multiple systematics are formatted correctly."""
        exp_file = tmp_path / "exp.yaml"
        exp_data = {
            "name": "test",
            "data_source": {"sacc_data_file": "data.sacc", "filters": None},
            "two_point_factory": {
                "weak_lensing_factories": [],
                "number_counts_factories": [],
                "cmb_factories": [],
            },
        }
        exp_file.write_text(yaml.dump(exp_data))

        with patch(
            "firecrown.likelihood.factories.TwoPointExperiment.load_from_yaml"
        ) as mock_load:
            sys1 = MagicMock()
            sys1.type = "multiplicative_bias"

            sys2 = MagicMock()
            sys2.type = "linear_bias"

            mock_wl = MagicMock()
            mock_wl.type_source = "src0"
            mock_wl.per_bin_systematics = [sys1, sys2]
            mock_wl.global_systematics = []

            mock_experiment = mock_load.return_value
            mock_experiment.two_point_factory.weak_lensing_factories = [mock_wl]
            mock_experiment.two_point_factory.number_counts_factories = []
            mock_experiment.two_point_factory.cmb_factories = []
            mock_experiment.data_source.sacc_data_file = "data.sacc"
            mock_experiment.data_source.filters = None

            View(experiment_file=exp_file)
            captured = capsys.readouterr()
            assert "multiplicative_bias" in captured.out
            assert "linear_bias" in captured.out
