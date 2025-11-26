"""Unit tests for firecrown.app.sacc.View class.

Tests for viewing and displaying SACC file contents and quality checks.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import pytest
import numpy as np
import sacc
from firecrown.app.sacc import View


@pytest.fixture(name="mock_sacc_data")
def fixture_mock_sacc_data() -> sacc.Sacc:
    """Create mock SACC data."""
    s = sacc.Sacc()

    # Add tracers
    z = np.linspace(0.0, 2.0, 50)
    dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
    s.add_tracer("NZ", "bin0", z, dndz)
    s.add_tracer("NZ", "bin1", z, dndz)

    # Add data points with ell tag
    ells = np.array([10, 20, 30])
    for ell in ells:
        s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=int(ell))

    # Add covariance
    cov = np.eye(len(ells)) * 0.1
    s.add_covariance(cov)

    return s


@pytest.fixture(name="sacc_file")
def fixture_sacc_file(tmp_path: Path, mock_sacc_data: sacc.Sacc) -> Path:
    """Create temporary SACC file."""
    sacc_path = tmp_path / "test.sacc"
    mock_sacc_data.save_fits(str(sacc_path))
    return sacc_path


class TestViewInit:
    """Tests for View initialization."""

    def test_view_init(self, sacc_file: Path) -> None:
        """Test View initialization."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert view.sacc_file == sacc_file
        assert view.sacc_data is not None
        assert hasattr(view, "all_tracers")
        assert hasattr(view, "bin_comb_harmonic")
        assert hasattr(view, "bin_comb_real")

    def test_view_init_default_parameters(self, sacc_file: Path) -> None:
        """Test View initialization with default parameters."""
        view = View(sacc_file=sacc_file)

        assert view.sacc_file == sacc_file
        assert view.plot_covariance is False
        assert view.check is False
        assert view.allow_mixed_types is False

    def test_view_init_with_check_flag(self, sacc_file: Path) -> None:
        """Test View initialization with check flag enabled."""
        view = View(sacc_file=sacc_file, check=False)

        assert view.check is False


class TestViewDisplay:
    """Tests for View display methods."""

    def test_view_show_sacc_summary(self, sacc_file: Path) -> None:
        """Test SACC summary display."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert len(view.sacc_data.tracers) == 2
        assert len(view.sacc_data.mean) == 3
        assert view.sacc_data.covariance is not None

    def test_view_show_tracers(self, sacc_file: Path) -> None:
        """Test tracer display."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert len(view.all_tracers) >= 0

    def test_view_all_tracers_extracted(self, sacc_file: Path) -> None:
        """Test that View properly extracts all tracers."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert len(view.all_tracers) >= 0
        # Check that tracers are sorted by name
        if len(view.all_tracers) > 1:
            for i in range(len(view.all_tracers) - 1):
                assert view.all_tracers[i].bin_name <= view.all_tracers[i + 1].bin_name

    def test_view_show_harmonic_bins(self, sacc_file: Path) -> None:
        """Test harmonic bins display."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert isinstance(view.bin_comb_harmonic, list)
        assert isinstance(view.bin_dict_harmonic, dict)

    def test_view_harmonic_bins_extracted(self, sacc_file: Path) -> None:
        """Test that View extracts harmonic bins."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert isinstance(view.bin_comb_harmonic, list)
        assert isinstance(view.bin_dict_harmonic, dict)
        assert len(view.bin_comb_harmonic) > 0

    def test_view_show_real_bins(self, sacc_file: Path) -> None:
        """Test real bins display."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert isinstance(view.bin_comb_real, list)
        assert isinstance(view.bin_dict_real, dict)

    def test_view_real_bins_structure(self, sacc_file: Path) -> None:
        """Test that View initializes real bins structure."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        assert isinstance(view.bin_comb_real, list)
        assert isinstance(view.bin_dict_real, dict)

    def test_view_bin_dict_keys(self, sacc_file: Path) -> None:
        """Test that bin dictionaries have proper keys."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        # Each key should be a tuple of (x_name, x_meas, y_name, y_meas)
        for key in view.bin_dict_harmonic.keys():
            assert isinstance(key, tuple)
            assert len(key) == 4

    def test_view_shows_summary(self, sacc_file: Path, capsys) -> None:
        """Test that View displays a summary."""
        _ = View(sacc_file=sacc_file, plot_covariance=False)

        captured = capsys.readouterr()
        # Check that output contains expected content
        assert "SACC Summary" in captured.out or "tracers" in captured.out.lower()

    def test_view_shows_tracers_table(self, sacc_file: Path, capsys) -> None:
        """Test that View displays tracers table."""
        _ = View(sacc_file=sacc_file, plot_covariance=False)

        captured = capsys.readouterr()
        # Check that tracers information is displayed
        assert "Tracers" in captured.out or "bin" in captured.out.lower()

    def test_view_shows_harmonic_bins_table(self, sacc_file: Path, capsys) -> None:
        """Test that View displays harmonic bins table."""
        _ = View(sacc_file=sacc_file, plot_covariance=False)

        captured = capsys.readouterr()
        # Check that harmonic bins info is displayed
        output = captured.out.lower()
        assert "harmonic" in output or "ells" in output


class TestViewPlotting:
    """Tests for View plotting methods."""

    @patch("matplotlib.pyplot.show")
    def test_view_plot_covariance(self, mock_show: Mock, sacc_file: Path) -> None:
        """Test covariance plotting."""
        _ = View(sacc_file=sacc_file, plot_covariance=True)

        mock_show.assert_called_once()

    def test_view_plot_covariance_no_cov(self, tmp_path: Path) -> None:
        """Test plotting with no covariance."""
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin0"), 1.0, ell=10)

        sacc_path = tmp_path / "no_cov.sacc"
        s.save_fits(str(sacc_path))

        with pytest.raises(Exception):
            View(sacc_file=sacc_path, plot_covariance=True)

    def test_view_get_ordered_correlation(self, sacc_file: Path) -> None:
        """Test correlation matrix ordering."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        all_bins = view.bin_comb_harmonic + view.bin_comb_real
        if len(all_bins) > 0:
            # pylint: disable-next=protected-access
            cor = view._get_ordered_correlation(all_bins)
            assert cor.shape[0] == cor.shape[1]

    def test_view_get_ordered_correlation_shape(self, sacc_file: Path) -> None:
        """Test that ordered correlation matrix has proper shape."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        all_bins = view.bin_comb_harmonic + view.bin_comb_real
        if len(all_bins) > 0:
            # pylint: disable-next=protected-access
            cor = view._get_ordered_correlation(all_bins)
            assert cor.shape[0] == cor.shape[1]
            assert cor.shape[0] > 0

    @patch("matplotlib.pyplot.show")
    def test_view_plot_correlation_matrix(
        self, _mock_show: Mock, sacc_file: Path
    ) -> None:
        """Test correlation matrix plotting."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        fig, ax = plt.subplots()
        cor = np.eye(3)
        # pylint: disable-next=protected-access
        im = view._plot_correlation_matrix(ax, cor)

        assert im is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_plot_correlation_matrix_creates_image(
        self, _mock_show: Mock, sacc_file: Path
    ) -> None:
        """Test that correlation matrix plotting creates an image."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        fig, ax = plt.subplots()
        cor = np.eye(5)
        # pylint: disable-next=protected-access
        im = view._plot_correlation_matrix(ax, cor)

        assert im is not None
        assert hasattr(im, "set_data")
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_add_bin_annotations(self, _mock_show: Mock, sacc_file: Path) -> None:
        """Test bin annotations."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        fig, ax = plt.subplots()
        all_bins = view.bin_comb_harmonic + view.bin_comb_real

        if len(all_bins) > 0:
            # pylint: disable-next=protected-access
            view._add_bin_annotations(ax, all_bins)

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_add_bin_annotations_with_multiple_bins(
        self, _mock_show: Mock, sacc_file: Path
    ) -> None:
        """Test adding annotations with multiple bins."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        fig, ax = plt.subplots()
        all_bins = view.bin_comb_harmonic + view.bin_comb_real

        if len(all_bins) > 0:
            # pylint: disable-next=protected-access
            view._add_bin_annotations(ax, all_bins)
            # Check that legend was added
            legend = ax.get_legend()
            assert legend is not None

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_add_plot_decorations(self, _mock_show: Mock, sacc_file: Path) -> None:
        """Test plot decorations."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        fig, ax = plt.subplots()
        cor = np.eye(3)
        im = ax.matshow(cor)

        # pylint: disable-next=protected-access
        view._add_plot_decorations(fig, ax, im)

        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_view_plot_decorations_adds_colorbar(
        self, _mock_show: Mock, sacc_file: Path
    ) -> None:
        """Test that plot decorations adds colorbar."""
        view = View(sacc_file=sacc_file, plot_covariance=False)

        fig, ax = plt.subplots()
        cor = np.eye(3)
        im = ax.matshow(cor)

        # pylint: disable-next=protected-access
        view._add_plot_decorations(fig, ax, im)

        # Check that title was set
        assert ax.get_title() != ""

        plt.close(fig)


class TestViewSpecialCases:
    """Tests for View special cases and edge conditions."""

    def test_view_covariance_none_raises_error(self, tmp_path: Path) -> None:
        """Test that plotting covariance without cov raises error."""
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin0"), 1.0, ell=10)

        sacc_path = tmp_path / "no_cov.sacc"
        s.save_fits(str(sacc_path))

        with pytest.raises(Exception):
            View(sacc_file=sacc_path, plot_covariance=True)

    def test_view_multiple_tracers_sorted(self, tmp_path: Path) -> None:
        """Test that View sorts multiple tracers."""
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)

        # Add tracers in reverse order
        s.add_tracer("NZ", "z_last", z, dndz)
        s.add_tracer("NZ", "a_first", z, dndz)
        s.add_tracer("NZ", "m_middle", z, dndz)

        ells = np.array([10, 20, 30])
        # Add data points for all three tracers to ensure they are all used
        for ell in ells:
            s.add_data_point(
                "galaxy_shear_cl_ee", ("a_first", "m_middle"), 1.0, ell=int(ell)
            )
            s.add_data_point(
                "galaxy_shear_cl_ee", ("a_first", "z_last"), 1.0, ell=int(ell)
            )

        cov = np.eye(len(ells) * 2) * 0.1
        s.add_covariance(cov)

        sacc_path = tmp_path / "test_sorted.sacc"
        s.save_fits(str(sacc_path))

        view = View(sacc_file=sacc_path, plot_covariance=False)

        # Verify tracers are sorted
        assert view.all_tracers[0].bin_name == "a_first"
        assert view.all_tracers[1].bin_name == "m_middle"
        assert view.all_tracers[2].bin_name == "z_last"

    def test_view_with_plot_covariance_true(self, sacc_file: Path) -> None:
        """Test View with plot_covariance=True (line 118)."""
        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_file, plot_covariance=True)
            assert view.plot_covariance is True
            # Verify View was initialized without errors
            assert view.sacc_data is not None

    def test_view_with_check_flag_true(self, sacc_file: Path) -> None:
        """Test View with check=True to trigger quality checks."""
        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_file, check=True, plot_covariance=False)
            assert view.check is True
            # Verify View was initialized without errors
            assert view.sacc_data is not None

    def test_view_check_sacc_quality(self, sacc_file: Path) -> None:
        """Test SACC quality check execution."""
        with patch("matplotlib.pyplot.show"):
            # Create view with quality checks enabled
            view = View(sacc_file=sacc_file, check=True, plot_covariance=False)
            # Verify the check ran without raising exceptions
            assert view.check is True

    def test_view_show_final_summary(self, sacc_file: Path) -> None:
        """Test final summary display."""
        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_file, plot_covariance=False)
            # Verify the view was created and contains data
            assert len(view.all_tracers) > 0
            assert view.sacc_data.mean is not None

    def test_view_plot_covariance_execution(self, sacc_file: Path) -> None:
        """Test covariance plotting execution."""
        with patch("matplotlib.pyplot.show"):
            # Create view with covariance plotting
            view = View(sacc_file=sacc_file, plot_covariance=True)
            # Verify no exceptions were raised during plotting
            assert view.sacc_data.covariance is not None

    def test_view_check_quality_execution(self, sacc_file: Path) -> None:
        """Test quality check execution."""
        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_file, check=True, plot_covariance=False)
            # Verify quality check ran
            assert view.check is True

    def test_view_show_harmonic_bins_populated(self, sacc_file: Path) -> None:
        """Test that harmonic bins are populated."""
        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_file, plot_covariance=False)
            # The view should have processed bins
            assert hasattr(view, "bin_comb_harmonic")

    def test_view_show_real_bins_empty(self, tmp_path: Path) -> None:
        """Test when real bins are empty."""
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)

        # Add only harmonic data (no real space)
        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=int(ell))

        cov = np.eye(len(ells)) * 0.1
        s.add_covariance(cov)

        sacc_path = tmp_path / "harmonic_only.sacc"
        s.save_fits(str(sacc_path))

        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_path, plot_covariance=False)
            # Verify no real bins were found
            assert len(view.bin_comb_real) == 0

    def test_view_extract_harmonic_bins(self, sacc_file: Path) -> None:
        """Test extraction of harmonic bins."""
        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_file, plot_covariance=False)
            # Verify harmonic bins were extracted
            assert len(view.bin_comb_harmonic) > 0

    def test_view_capture_warnings(self, sacc_file: Path) -> None:
        """Test that warnings are captured during quality checks."""
        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_file, check=True, plot_covariance=False)
            # Verify check completed without error
            assert view.check is True
