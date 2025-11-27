"""Unit tests for firecrown.app.sacc.Load class.

Tests for loading and initializing SACC files.
"""

from pathlib import Path
import pytest
import numpy as np
import sacc
from firecrown.app.sacc import Load


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


class TestLoadInit:
    """Tests for Load initialization."""

    def test_load_init(self, sacc_file: Path) -> None:
        """Test Load initialization."""
        load = Load(sacc_file=sacc_file)

        assert load.sacc_file == sacc_file
        assert load.sacc_data is not None
        assert len(load.sacc_data.tracers) == 2

    def test_load_init_allow_mixed_types(self, sacc_file: Path) -> None:
        """Test Load initialization with allow_mixed_types parameter."""
        load = Load(sacc_file=sacc_file, allow_mixed_types=True)

        assert load.sacc_file == sacc_file
        assert load.allow_mixed_types is True

    def test_load_sacc_data_loaded(self, sacc_file: Path) -> None:
        """Test that Load properly initializes sacc_data."""
        load = Load(sacc_file=sacc_file)

        assert load.sacc_data is not None
        assert len(load.sacc_data.tracers) == 2
        assert len(load.sacc_data.mean) == 3


class TestLoadErrors:
    """Tests for Load error handling."""

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test Load with non-existent file."""
        missing_file = tmp_path / "missing.sacc"

        with pytest.raises(Exception):
            Load(sacc_file=missing_file)

    def test_load_invalid_file(self, tmp_path: Path) -> None:
        """Test Load with invalid SACC file."""
        invalid_file = tmp_path / "invalid.sacc"
        invalid_file.write_text("not a sacc file")

        with pytest.raises(Exception):
            Load(sacc_file=invalid_file)

    def test_load_with_nonexistent_file(self, tmp_path: Path) -> None:
        """Test Load raises error with nonexistent file."""
        missing_file = tmp_path / "nonexistent.sacc"

        with pytest.raises(Exception):
            Load(sacc_file=missing_file)

    def test_load_with_corrupted_file(self, tmp_path: Path) -> None:
        """Test Load raises error with corrupted SACC file."""
        corrupted_file = tmp_path / "corrupted.sacc"
        corrupted_file.write_text("This is not a valid SACC file at all!")

        with pytest.raises(Exception):
            Load(sacc_file=corrupted_file)
