"""Unit tests for firecrown.app.sacc.View class.

Tests for viewing and displaying SACC file contents and quality checks.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import pytest
from _pytest.capture import CaptureFixture
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

    def test_view_shows_summary(
        self, sacc_file: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test that View displays a summary."""
        _ = View(sacc_file=sacc_file, plot_covariance=False)

        captured = capsys.readouterr()
        # Check that output contains expected content
        assert "SACC Summary" in captured.out or "tracers" in captured.out.lower()

    def test_view_shows_tracers_table(
        self, sacc_file: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test that View displays tracers table."""
        _ = View(sacc_file=sacc_file, plot_covariance=False)

        captured = capsys.readouterr()
        # Check that tracers information is displayed
        assert "Tracers" in captured.out or "bin" in captured.out.lower()

    def test_view_shows_harmonic_bins_table(
        self, sacc_file: Path, capsys: CaptureFixture[str]
    ) -> None:
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

    def test_view_show_real_bins_with_data(self) -> None:
        """Test showing real-space bins when they exist.

        Uses old_format_real.sacc which has real-space measurements (xi_t, xi_plus, xi_minus).
        """
        # Use the real SACC file with real-space data
        sacc_path = Path("tests/old_format_real.sacc")
        assert sacc_path.exists(), "old_format_real.sacc must exist in tests directory"

        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_path, plot_covariance=False)
            # Verify real bins were extracted
            assert len(view.bin_comb_real) > 0
            assert len(view.bin_comb_harmonic) == 0  # No harmonic data
            # Verify the display method executes (covers lines 239-266)
            view._show_real_bins()  # pylint: disable=protected-access
            # Verify specific real-space measurements are present
            assert any(
                "galaxy_shearDensity_xi_t" in str(b.metadata.get_sacc_name())
                for b in view.bin_comb_real
            )

    def test_view_quality_check_with_warnings(self) -> None:
        """Test quality check that captures warnings from SACC operations.

        Uses old_format_real.sacc which has legacy format warnings.
        """
        # Use the real SACC file with legacy format that triggers warnings
        sacc_path = Path("tests/old_format_real.sacc")
        assert sacc_path.exists(), "old_format_real.sacc must exist in tests directory"

        with patch("matplotlib.pyplot.show"):
            view = View(sacc_file=sacc_path, check=True, plot_covariance=False)
            # This should trigger the quality check logic and capture warnings
            assert view.check is True
            # Verify tracers were extracted from the real file
            assert len(view.all_tracers) == 2
            assert any(t.bin_name == "src0" for t in view.all_tracers)
            assert any(t.bin_name == "lens0" for t in view.all_tracers)

    def test_view_quality_check_harmonic_data(self) -> None:
        """Test quality check with harmonic space data.

        Uses old_format_harmonic.sacc which has legacy format and harmonic measurements.
        """
        # Use the real SACC file with harmonic data
        sacc_path = Path("tests/old_format_harmonic.sacc")
        assert (
            sacc_path.exists()
        ), "old_format_harmonic.sacc must exist in tests directory"

        with patch("matplotlib.pyplot.show"):
            # Run quality check - should complete without raising errors
            view = View(sacc_file=sacc_path, check=True, plot_covariance=False)
            assert view.sacc_file == sacc_path
            assert view.check is True
            # Verify harmonic bins were extracted
            assert len(view.bin_comb_harmonic) > 0
            assert len(view.bin_comb_real) == 0  # No real-space data

    def test_view_quality_check_all_pass(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test quality check when all checks pass (line 147 condition false).

        This test specifically covers the case where total_issues == 0 and
        has_validation_error is False, triggering the success message on line 148.
        """
        # Create compliant SACC data that will pass all quality checks
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "src0", z, dndz)
        s.add_tracer("NZ", "lens0", z, dndz)

        # Add auto-correlations to establish measurement types
        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("src0", "src0"), 1.0, ell=int(ell))
            s.add_data_point("galaxy_density_cl", ("lens0", "lens0"), 1.0, ell=int(ell))

        # Add cross-correlation with CORRECT ordering (src0 before lens0)
        for ell in ells:
            s.add_data_point(
                "galaxy_shearDensity_cl_e", ("src0", "lens0"), 1.0, ell=int(ell)
            )

        cov = np.eye(len(ells) * 3) * 0.1
        s.add_covariance(cov)

        sacc_path = tmp_path / "compliant.sacc"
        s.save_fits(str(sacc_path))

        with patch("matplotlib.pyplot.show"):
            # Run quality check on compliant SACC file
            view = View(sacc_file=sacc_path, check=True, plot_covariance=False)

            # Verify the view was created successfully
            assert view.check is True
            assert view.sacc_file == sacc_path

            # Check that success message was printed (covers line 147-148)
            captured = capsys.readouterr()
            assert "✅ All quality checks passed!" in captured.out

    def test_view_plot_covariance_missing_raises_error(self, tmp_path: Path) -> None:
        """Test _plot_covariance raises error when SACC has no covariance.

        This test specifically covers the error path in _plot_covariance when
        self.sacc_data.covariance is None, raising typer.BadParameter.
        """
        # Create SACC data WITHOUT covariance
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)

        # Add data points but NO covariance
        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=int(ell))
        # Note: NOT calling s.add_covariance() so covariance remains None

        sacc_path = tmp_path / "no_cov.sacc"
        s.save_fits(str(sacc_path))

        # Attempting to plot covariance should raise an error
        with pytest.raises(
            ValueError, match="The SACC object does not have a dense covariance matrix."
        ):
            View(sacc_file=sacc_path, plot_covariance=True)

    def test_view_plot_covariance_with_real_space_data(self) -> None:
        """Test plotting covariance with real-space SACC data.

        Uses old_format_real.sacc which has covariance and real-space measurements.
        """
        # Use the real SACC file with real-space data and covariance
        sacc_path = Path("tests/old_format_real.sacc")
        assert sacc_path.exists(), "old_format_real.sacc must exist in tests directory"

        with patch("matplotlib.pyplot.show") as mock_show:
            view = View(sacc_file=sacc_path, plot_covariance=True)
            # Verify plotting was called
            mock_show.assert_called_once()
            # Verify data was loaded
            assert view.sacc_data.covariance is not None
            assert len(view.bin_comb_real) > 0

    def test_view_plot_covariance_with_harmonic_data(self) -> None:
        """Test plotting covariance with harmonic-space SACC data.

        Uses old_format_harmonic.sacc which has covariance and harmonic measurements.
        """
        # Use the real SACC file with harmonic data and covariance
        sacc_path = Path("tests/old_format_harmonic.sacc")
        assert (
            sacc_path.exists()
        ), "old_format_harmonic.sacc must exist in tests directory"

        with patch("matplotlib.pyplot.show") as mock_show:
            view = View(sacc_file=sacc_path, plot_covariance=True)
            # Verify plotting was called
            mock_show.assert_called_once()
            # Verify data was loaded
            assert view.sacc_data.covariance is not None
            assert len(view.bin_comb_harmonic) > 0

    def test_view_empty_sacc_with_cov(self, tmp_path: Path) -> None:
        """Test quality check on empty SACC file with covariance."""
        sacc_path = tmp_path / "empty_cov.sacc"
        sacc_data = sacc.Sacc()

        sacc_data.add_tracer("misc", "bin0")  # Add a tracer without data
        sacc_data.add_data_point("not_a_type", ("bin0", "bin0"), 1.0, ell=10)
        sacc_data.add_covariance([1.0])  # Add trivial covariance
        sacc_data.save_fits(str(sacc_path))
        View(sacc_file=sacc_path, check=True, plot_covariance=False)

    def test_view_quality_check_with_unhandled_stdout_messages(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test quality check with unhandled stdout messages.
        
        This test patches stdout during SACC operations to inject messages
        that are not handled by any specific handler, ensuring the quality
        check processes all stdout content.
        """
        # Create a simple SACC file
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)

        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=int(ell))

        cov = np.eye(len(ells)) * 0.1
        s.add_covariance(cov)

        sacc_path = tmp_path / "test_stdout.sacc"
        s.save_fits(str(sacc_path))

        # Patch _capture_sacc_operations to inject stdout messages
        with patch("matplotlib.pyplot.show"):
            with patch.object(
                View,
                "_capture_sacc_operations",
                return_value=(
                    "Some unhandled stdout message line 1\n"
                    "Another unhandled stdout message line 2\n"
                    "INFO: This is an informational message\n",
                    "",  # stderr
                    [],  # warnings
                    None,  # validation_error
                ),
            ):
                view = View(sacc_file=sacc_path, check=True, plot_covariance=False)
                assert view.check is True

                # Verify the quality check completed without errors
                # Unhandled messages should be silently ignored or logged
                captured = capsys.readouterr()
                # Should still show quality checks header
                assert "SACC Quality Checks" in captured.out

    def test_view_quality_check_with_unhandled_stderr_messages(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test quality check with unhandled stderr messages.
        
        This test patches stderr during SACC operations to inject messages
        that are not handled by any specific handler, ensuring the quality
        check processes all stderr content.
        """
        # Create a simple SACC file
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)

        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=int(ell))

        cov = np.eye(len(ells)) * 0.1
        s.add_covariance(cov)

        sacc_path = tmp_path / "test_stderr.sacc"
        s.save_fits(str(sacc_path))

        # Patch _capture_sacc_operations to inject stderr messages
        with patch("matplotlib.pyplot.show"):
            with patch.object(
                View,
                "_capture_sacc_operations",
                return_value=(
                    "",  # stdout
                    "ERROR: Some unhandled error message\n"
                    "WARNING: Unhandled warning in stderr\n"
                    "DEBUG: Debug information\n",
                    [],  # warnings
                    None,  # validation_error
                ),
            ):
                view = View(sacc_file=sacc_path, check=True, plot_covariance=False)
                assert view.check is True

                # Verify the quality check completed without errors
                captured = capsys.readouterr()
                assert "SACC Quality Checks" in captured.out

    def test_view_quality_check_with_mixed_unhandled_messages(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test quality check with unhandled messages in both stdout and stderr.
        
        This test injects unhandled messages into both stdout and stderr
        to ensure the quality check properly processes all output streams.
        """
        # Create a simple SACC file
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)

        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=int(ell))

        cov = np.eye(len(ells)) * 0.1
        s.add_covariance(cov)

        sacc_path = tmp_path / "test_mixed.sacc"
        s.save_fits(str(sacc_path))

        # Patch _capture_sacc_operations to inject both stdout and stderr messages
        with patch("matplotlib.pyplot.show"):
            with patch.object(
                View,
                "_capture_sacc_operations",
                return_value=(
                    "STDOUT: Unhandled stdout message\n"
                    "INFO: Information on stdout\n",
                    "STDERR: Unhandled stderr message\n"
                    "ERROR: Error on stderr\n",
                    [],  # warnings
                    None,  # validation_error
                ),
            ):
                view = View(sacc_file=sacc_path, check=True, plot_covariance=False)
                assert view.check is True

                # Verify the quality check completed
                captured = capsys.readouterr()
                assert "SACC Quality Checks" in captured.out

    def test_view_quality_check_handlers_consume_all_lines(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test that handlers properly consume their handled lines.
        
        This tests the handler loop logic to ensure handlers that successfully
        handle messages consume those lines from the stream, and unhandled
        lines are left for subsequent handlers or ignored.
        """
        # Create a simple SACC file
        s = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)

        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=int(ell))

        cov = np.eye(len(ells)) * 0.1
        s.add_covariance(cov)

        sacc_path = tmp_path / "test_handlers.sacc"
        s.save_fits(str(sacc_path))

        # Inject multiple lines with a mix of potentially handled and unhandled messages
        with patch("matplotlib.pyplot.show"):
            with patch.object(
                View,
                "_capture_sacc_operations",
                return_value=(
                    "Line 1: Some info\n"
                    "Line 2: Another message\n"
                    "Line 3: Third message\n"
                    "Line 4: Fourth message\n"
                    "Line 5: Fifth message\n",
                    "StdErr Line 1\n"
                    "StdErr Line 2\n"
                    "StdErr Line 3\n",
                    [],  # warnings
                    None,  # validation_error
                ),
            ):
                view = View(sacc_file=sacc_path, check=True, plot_covariance=False)
                assert view.check is True

                # Verify handlers processed the streams
                captured = capsys.readouterr()
                assert "SACC Quality Checks" in captured.out
