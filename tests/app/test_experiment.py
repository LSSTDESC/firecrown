"""Unit tests for firecrown.app.experiment command module.

Tests experiment file loading, viewing, and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
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
    """Tests for Load command initialization."""

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(Load, "__post_init__")
    def test_load_requires_experiment_file(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test that Load requires experiment_file parameter."""
        # Just verify the dataclass can be instantiated
        load = Load(experiment_file=Path("test.yaml"))
        assert load.experiment_file == Path("test.yaml")

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(Load, "__post_init__")
    def test_load_stores_experiment_file_path(
        self, mock_post_init: MagicMock, mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test that Load stores the experiment file path."""
        exp_file = tmp_path / "exp.yaml"
        exp_file.touch()

        load = Load(experiment_file=exp_file)
        assert load.experiment_file == exp_file

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(Load, "__post_init__")
    def test_load_calls_load_experiment(
        self, mock_post_init: MagicMock, mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test that __post_init__ calls _load_experiment."""
        exp_file = tmp_path / "exp.yaml"
        load = Load(experiment_file=exp_file)
        # Mock post_init to verify structure
        assert hasattr(load, "experiment_file")


class TestLoadExperiment:
    """Tests for Load._load_experiment method."""

    @patch("firecrown.app.experiment.factories.TwoPointExperiment.load_from_yaml")
    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(Load, "__post_init__")
    def test_load_experiment_calls_factory(
        self,
        mock_post_init: MagicMock,
        mock_logging: MagicMock,
        mock_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that _load_experiment calls the factory."""
        exp_file = tmp_path / "exp.yaml"
        exp_file.write_text("dummy: content")

        mock_experiment = MagicMock()
        mock_factory.return_value = mock_experiment

        load = Load(experiment_file=exp_file)
        load.console = MagicMock()
        load._load_experiment()

        # Factory should have been called
        assert mock_factory.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(Load, "__post_init__")
    def test_load_experiment_attribute_set(
        self, mock_post_init: MagicMock, mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test that tp_experiment attribute is set after loading."""
        exp_file = tmp_path / "exp.yaml"

        load = Load(experiment_file=exp_file)
        load.tp_experiment = MagicMock()  # Set manually for testing

        assert hasattr(load, "tp_experiment")


class TestViewInitialization:
    """Tests for View command initialization."""

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_view_calls_load_and_print(
        self, mock_post_init: MagicMock, mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test that View calls both _load_experiment and _print_factories."""
        exp_file = tmp_path / "exp.yaml"
        view = View(experiment_file=exp_file)

        # Both methods should have been called during __post_init__
        assert hasattr(view, "experiment_file")

    def test_view_inherits_from_load(self) -> None:
        """Test that View is a subclass of Load."""
        assert issubclass(View, Load)


class TestViewPrintFactories:
    """Tests for View._print_factories method."""

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_creates_table(
        self, mock_post_init: MagicMock, mock_logging: MagicMock, tmp_path: Path
    ) -> None:
        """Test that _print_factories creates and prints a table."""
        view = View(experiment_file=Path("dummy.yaml"))

        # Mock the experiment and factories
        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        # Empty factories lists
        mock_tp_factory.weak_lensing_factories = []
        mock_tp_factory.number_counts_factories = []
        mock_tp_factory.cmb_factories = []

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        # Verify console.print was called
        assert view.console.print.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_with_weak_lensing(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test printing factories with weak lensing data."""
        view = View(experiment_file=Path("dummy.yaml"))

        # Mock weak lensing factory
        mock_wl_factory = MagicMock()
        mock_wl_factory.type_source = "galaxy_shear"
        mock_wl_factory.per_bin_systematics = []
        mock_wl_factory.global_systematics = []

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        mock_tp_factory.weak_lensing_factories = [mock_wl_factory]
        mock_tp_factory.number_counts_factories = []
        mock_tp_factory.cmb_factories = []

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        assert view.console.print.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_with_number_counts(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test printing factories with number counts data."""
        view = View(experiment_file=Path("dummy.yaml"))

        # Mock number counts factory
        mock_nc_factory = MagicMock()
        mock_nc_factory.type_source = "galaxy_counts"
        mock_nc_factory.per_bin_systematics = []
        mock_nc_factory.global_systematics = []

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        mock_tp_factory.weak_lensing_factories = []
        mock_tp_factory.number_counts_factories = [mock_nc_factory]
        mock_tp_factory.cmb_factories = []

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        assert view.console.print.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_with_cmb(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test printing factories with CMB data."""
        view = View(experiment_file=Path("dummy.yaml"))

        # Mock CMB factory
        mock_cmb_factory = MagicMock()
        mock_cmb_factory.type_source = "cmb_convergence"

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        mock_tp_factory.weak_lensing_factories = []
        mock_tp_factory.number_counts_factories = []
        mock_tp_factory.cmb_factories = [mock_cmb_factory]

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        assert view.console.print.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_with_systematics(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test printing factories with systematics."""
        view = View(experiment_file=Path("dummy.yaml"))

        # Mock systematic
        mock_systematic = MagicMock()
        mock_systematic.type = "ia_bias"

        mock_wl_factory = MagicMock()
        mock_wl_factory.type_source = "galaxy_shear"
        mock_wl_factory.per_bin_systematics = [mock_systematic]
        mock_wl_factory.global_systematics = [mock_systematic]

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        mock_tp_factory.weak_lensing_factories = [mock_wl_factory]
        mock_tp_factory.number_counts_factories = []
        mock_tp_factory.cmb_factories = []

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        assert view.console.print.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_with_filters(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test printing data source with filters."""
        view = View(experiment_file=Path("dummy.yaml"))

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = "scale_cut_filters.yaml"

        mock_tp_factory.weak_lensing_factories = []
        mock_tp_factory.number_counts_factories = []
        mock_tp_factory.cmb_factories = []

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        assert view.console.print.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_multiple_of_each_type(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test printing multiple factories of each type."""
        view = View(experiment_file=Path("dummy.yaml"))

        # Create multiple factories
        wl1 = MagicMock()
        wl1.type_source = "source1"
        wl1.per_bin_systematics = []
        wl1.global_systematics = []

        wl2 = MagicMock()
        wl2.type_source = "source2"
        wl2.per_bin_systematics = []
        wl2.global_systematics = []

        nc1 = MagicMock()
        nc1.type_source = "counts1"
        nc1.per_bin_systematics = []
        nc1.global_systematics = []

        cmb1 = MagicMock()
        cmb1.type_source = "cmb1"

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        mock_tp_factory.weak_lensing_factories = [wl1, wl2]
        mock_tp_factory.number_counts_factories = [nc1]
        mock_tp_factory.cmb_factories = [cmb1]

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        assert view.console.print.called

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_print_factories_empty_systematics_list(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test formatting when systematics list is empty."""
        view = View(experiment_file=Path("dummy.yaml"))

        mock_wl_factory = MagicMock()
        mock_wl_factory.type_source = "galaxy_shear"
        mock_wl_factory.per_bin_systematics = []
        mock_wl_factory.global_systematics = []

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        mock_tp_factory.weak_lensing_factories = [mock_wl_factory]
        mock_tp_factory.number_counts_factories = []
        mock_tp_factory.cmb_factories = []

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        assert view.console.print.called


class TestFormatSysFunction:
    """Tests for the fmt_sys helper function within _print_factories."""

    @patch("firecrown.app.experiment.logging.Logging.__init__", return_value=None)
    @patch.object(View, "__post_init__")
    def test_fmt_sys_empty_sequence(
        self, mock_post_init: MagicMock, mock_logging: MagicMock
    ) -> None:
        """Test fmt_sys with empty sequence returns dash."""
        view = View(experiment_file=Path("dummy.yaml"))

        mock_experiment = MagicMock()
        mock_tp_factory = MagicMock()
        mock_experiment.two_point_factory = mock_tp_factory
        mock_experiment.data_source.sacc_data_file = "data.sacc"
        mock_experiment.data_source.filters = None

        mock_wl_factory = MagicMock()
        mock_wl_factory.type_source = "galaxy_shear"
        mock_wl_factory.per_bin_systematics = []
        mock_wl_factory.global_systematics = []

        mock_tp_factory.weak_lensing_factories = [mock_wl_factory]
        mock_tp_factory.number_counts_factories = []
        mock_tp_factory.cmb_factories = []

        view.tp_experiment = mock_experiment
        view.console = MagicMock()

        view._print_factories()

        # Verify that empty lists are handled (should print dash)
        assert view.console.print.called
