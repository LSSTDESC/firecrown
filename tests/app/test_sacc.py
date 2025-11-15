"""Unit tests for firecrown.app.sacc module."""

from pathlib import Path
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import pytest
import numpy as np
import sacc
from firecrown import metadata_types as mdt
from firecrown.app.sacc import mean_std_tracer, Load, View


@pytest.fixture(name="mock_tracer")
def fixture_mock_tracer() -> mdt.InferredGalaxyZDist:
    """Create mock tracer with Gaussian distribution."""
    z = np.linspace(0.0, 2.0, 100)
    mean = 1.0
    sigma = 0.2
    dndz = np.exp(-0.5 * ((z - mean) / sigma) ** 2)
    dndz /= np.trapezoid(dndz, z)

    return mdt.InferredGalaxyZDist(
        bin_name="bin0",
        z=z,
        dndz=dndz,
        measurements=set([mdt.Galaxies.COUNTS]),
        type_source=mdt.TypeSource("firecrown"),
    )


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


def test_mean_std_tracer_gaussian(mock_tracer: mdt.InferredGalaxyZDist) -> None:
    """Test mean_std_tracer with Gaussian distribution."""
    mean, std = mean_std_tracer(mock_tracer)

    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert 0.9 < mean < 1.1  # Expected mean ~1.0
    assert 0.15 < std < 0.25  # Expected std ~0.2


def test_mean_std_tracer_uniform() -> None:
    """Test mean_std_tracer with uniform distribution."""
    z = np.linspace(0.5, 1.5, 100)
    dndz = np.ones_like(z)
    dndz /= np.trapezoid(dndz, z)

    tracer = mdt.InferredGalaxyZDist(
        bin_name="uniform",
        z=z,
        dndz=dndz,
        measurements=set([mdt.Galaxies.COUNTS]),
        type_source=mdt.TypeSource("test"),
    )

    mean, std = mean_std_tracer(tracer)

    assert 0.95 < mean < 1.05  # Expected mean = 1.0
    assert 0.25 < std < 0.35  # Expected std ~0.289


def test_load_init(sacc_file: Path) -> None:
    """Test Load initialization."""
    load = Load(sacc_file=sacc_file)

    assert load.sacc_file == sacc_file
    assert load.sacc_data is not None
    assert len(load.sacc_data.tracers) == 2


def test_load_file_not_found(tmp_path: Path) -> None:
    """Test Load with non-existent file."""
    missing_file = tmp_path / "missing.sacc"

    with pytest.raises(Exception):
        Load(sacc_file=missing_file)


def test_load_invalid_file(tmp_path: Path) -> None:
    """Test Load with invalid SACC file."""
    invalid_file = tmp_path / "invalid.sacc"
    invalid_file.write_text("not a sacc file")

    with pytest.raises(Exception):
        Load(sacc_file=invalid_file)


def test_view_init(sacc_file: Path) -> None:
    """Test View initialization."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    assert view.sacc_file == sacc_file
    assert view.sacc_data is not None
    assert hasattr(view, "all_tracers")
    assert hasattr(view, "bin_comb_harmonic")
    assert hasattr(view, "bin_comb_real")


def test_view_show_sacc_summary(sacc_file: Path) -> None:
    """Test SACC summary display."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    assert len(view.sacc_data.tracers) == 2
    assert len(view.sacc_data.mean) == 3
    assert view.sacc_data.covariance is not None


def test_view_show_tracers(sacc_file: Path) -> None:
    """Test tracer display."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    assert len(view.all_tracers) >= 0


def test_view_show_harmonic_bins(sacc_file: Path) -> None:
    """Test harmonic bins display."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    assert isinstance(view.bin_comb_harmonic, list)
    assert isinstance(view.bin_dict_harmonic, dict)


def test_view_show_real_bins(sacc_file: Path) -> None:
    """Test real bins display."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    assert isinstance(view.bin_comb_real, list)
    assert isinstance(view.bin_dict_real, dict)


@patch("matplotlib.pyplot.show")
def test_view_plot_covariance(mock_show: Mock, sacc_file: Path) -> None:
    """Test covariance plotting."""
    _ = View(sacc_file=sacc_file, plot_covariance=True)

    mock_show.assert_called_once()


def test_view_plot_covariance_no_cov(tmp_path: Path) -> None:
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


def test_view_get_ordered_correlation(sacc_file: Path) -> None:
    """Test correlation matrix ordering."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    all_bins = view.bin_comb_harmonic + view.bin_comb_real
    if len(all_bins) > 0:
        # pylint: disable-next=protected-access
        cor = view._get_ordered_correlation(all_bins)
        assert cor.shape[0] == cor.shape[1]


@patch("matplotlib.pyplot.show")
def test_view_plot_correlation_matrix(_mock_show: Mock, sacc_file: Path) -> None:
    """Test correlation matrix plotting."""

    view = View(sacc_file=sacc_file, plot_covariance=False)

    fig, ax = plt.subplots()
    cor = np.eye(3)
    # pylint: disable-next=protected-access
    im = view._plot_correlation_matrix(ax, cor)

    assert im is not None
    plt.close(fig)


@patch("matplotlib.pyplot.show")
def test_view_add_bin_annotations(_mock_show: Mock, sacc_file: Path) -> None:
    """Test bin annotations."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    fig, ax = plt.subplots()
    all_bins = view.bin_comb_harmonic + view.bin_comb_real

    if len(all_bins) > 0:
        # pylint: disable-next=protected-access
        view._add_bin_annotations(ax, all_bins)

    plt.close(fig)


@patch("matplotlib.pyplot.show")
def test_view_add_plot_decorations(_mock_show: Mock, sacc_file: Path) -> None:
    """Test plot decorations."""
    view = View(sacc_file=sacc_file, plot_covariance=False)

    fig, ax = plt.subplots()
    cor = np.eye(3)
    im = ax.matshow(cor)

    # pylint: disable-next=protected-access
    view._add_plot_decorations(fig, ax, im)

    plt.close(fig)
