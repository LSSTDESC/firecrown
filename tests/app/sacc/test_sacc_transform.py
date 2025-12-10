"""Unit tests for firecrown.app.sacc.Transform module.

Tests the SACC file transform command.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from pathlib import Path
import time

import numpy as np
import pytest
import sacc
from _pytest.capture import CaptureFixture, CaptureResult

from firecrown.app.sacc import Transform, SaccFormat


class TestTransformBasic:
    """Basic tests for Transform class."""

    def test_transform_fits_to_hdf5(self, tmp_path: Path) -> None:
        """Test transforming FITS to HDF5."""
        input_file: Path = tmp_path / "test.fits"
        output_file: Path = tmp_path / "test.hdf5"

        # Create a minimal SACC file in FITS format
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Transform
        Transform(
            sacc_file=input_file,
            output=output_file,
            output_format=SaccFormat.HDF5,
        )

        # Verify output exists and can be read
        assert output_file.exists()
        s2: sacc.Sacc = sacc.Sacc.load_hdf5(str(output_file))
        assert len(s2.tracers) == 1

    def test_transform_hdf5_to_fits(self, tmp_path: Path) -> None:
        """Test transforming HDF5 to FITS."""
        input_file: Path = tmp_path / "test.hdf5"
        output_file: Path = tmp_path / "test.fits"

        # Create a minimal SACC file in HDF5 format
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Transform
        Transform(
            sacc_file=input_file,
            output=output_file,
            output_format=SaccFormat.FITS,
        )

        # Verify output exists and can be read
        assert output_file.exists()
        s2: sacc.Sacc = sacc.Sacc.load_fits(str(output_file))
        assert len(s2.tracers) == 1

    def test_transform_invalid_fits_file_exits(self, tmp_path: Path) -> None:
        """Test that invalid FITS file causes exit."""
        input_file: Path = tmp_path / "invalid.fits"
        output_file: Path = tmp_path / "output.hdf5"

        # Create a file that's not a valid SACC FITS file
        input_file.write_text("not a fits file")

        with pytest.raises(SystemExit):
            Transform(
                sacc_file=input_file,
                output=output_file,
                output_format=SaccFormat.HDF5,
            )

    def test_transform_invalid_hdf5_file_exits(self, tmp_path: Path) -> None:
        """Test that invalid HDF5 file causes exit."""
        input_file: Path = tmp_path / "invalid.hdf5"
        output_file: Path = tmp_path / "output.fits"

        # Create a file that's not a valid SACC HDF5 file
        input_file.write_text("not an hdf5 file")

        with pytest.raises(SystemExit):
            Transform(
                sacc_file=input_file,
                output=output_file,
                output_format=SaccFormat.FITS,
            )

    def test_transform_nonexistent_file_exits(self, tmp_path: Path) -> None:
        """Test that nonexistent input file causes exit."""
        input_file: Path = tmp_path / "nonexistent.fits"
        output_file: Path = tmp_path / "output.hdf5"

        with pytest.raises(SystemExit):
            Transform(
                sacc_file=input_file,
                output=output_file,
                output_format=SaccFormat.HDF5,
            )


class TestTransformAutoOutput:
    """Tests for automatic output path determination."""

    def test_auto_output_fits_to_hdf5(self, tmp_path: Path) -> None:
        """Test automatic output path generation FITS to HDF5."""
        input_file: Path = tmp_path / "test.fits"
        expected_output: Path = tmp_path / "test.hdf5"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Transform without specifying output
        Transform(
            sacc_file=input_file,
            output_format=SaccFormat.HDF5,
        )

        # Verify output was created with expected name
        assert expected_output.exists()

    def test_auto_output_hdf5_to_fits(self, tmp_path: Path) -> None:
        """Test automatic output path generation HDF5 to FITS."""
        input_file: Path = tmp_path / "test.hdf5"
        expected_output: Path = tmp_path / "test.fits"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Transform without specifying output
        Transform(
            sacc_file=input_file,
            output_format=SaccFormat.FITS,
        )

        # Verify output was created with expected name
        assert expected_output.exists()

    def test_same_format_no_extension_change(self, tmp_path: Path) -> None:
        """Test that same format doesn't change file extension."""
        input_file: Path = tmp_path / "test.hdf5"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Transform to same format without explicit output
        Transform(
            sacc_file=input_file,
            output_format=SaccFormat.HDF5,
            overwrite=True,
        )

        # Verify file still exists as HDF5
        assert input_file.exists()
        s2: sacc.Sacc = sacc.Sacc.load_hdf5(str(input_file))
        assert len(s2.tracers) == 1


class TestTransformOverwrite:
    """Tests for overwrite handling."""

    def test_refuses_overwrite_without_flag_same_file(self, tmp_path: Path) -> None:
        """Test that output to same file requires --overwrite."""
        input_file: Path = tmp_path / "test.hdf5"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Try to transform to same file without overwrite flag
        with pytest.raises(SystemExit):
            Transform(
                sacc_file=input_file,
                output=input_file,
                overwrite=False,
            )

    def test_refuses_overwrite_without_flag_existing_output(
        self, tmp_path: Path
    ) -> None:
        """Test that existing output file requires --overwrite."""
        input_file: Path = tmp_path / "test.fits"
        output_file: Path = tmp_path / "test.hdf5"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Create existing output file
        output_file.write_text("existing content")

        # Try to transform without overwrite flag
        with pytest.raises(SystemExit):
            Transform(
                sacc_file=input_file,
                output=output_file,
                overwrite=False,
            )

        # Verify output wasn't changed
        assert output_file.read_text() == "existing content"

    def test_overwrites_with_flag(self, tmp_path: Path) -> None:
        """Test that existing output is overwritten with --overwrite flag."""
        input_file: Path = tmp_path / "test.fits"
        output_file: Path = tmp_path / "test.hdf5"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Create existing output file
        output_file.write_text("existing content")

        # Transform with overwrite
        Transform(
            sacc_file=input_file,
            output=output_file,
            overwrite=True,
            output_format=SaccFormat.HDF5,
        )

        # Verify output was overwritten (and is now a valid SACC file)
        s2: sacc.Sacc = sacc.Sacc.load_hdf5(str(output_file))
        assert len(s2.tracers) == 1


class TestTransformFormatDetection:
    """Tests for format detection."""

    def test_detect_fits_extension(self, tmp_path: Path) -> None:
        """Test detecting FITS format from .fits extension."""
        input_file: Path = tmp_path / "test.fits"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Transform without specifying input format (should auto-detect)
        output_file: Path = tmp_path / "output.hdf5"
        Transform(
            sacc_file=input_file,
            output=output_file,
        )

        assert output_file.exists()

    def test_detect_hdf5_extension(self, tmp_path: Path) -> None:
        """Test detecting HDF5 format from .hdf5 extension."""
        input_file: Path = tmp_path / "test.hdf5"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Transform without specifying input format (should auto-detect)
        output_file: Path = tmp_path / "output.fits"
        Transform(
            sacc_file=input_file,
            output=output_file,
        )

        assert output_file.exists()

    def test_detect_h5_extension(self, tmp_path: Path) -> None:
        """Test detecting HDF5 format from .h5 extension."""
        input_file: Path = tmp_path / "test.h5"

        # Create input file with .h5 extension
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Transform without specifying input format (should auto-detect)
        output_file: Path = tmp_path / "output.fits"
        Transform(
            sacc_file=input_file,
            output=output_file,
        )

        assert output_file.exists()

    def test_forced_input_format(self, tmp_path: Path) -> None:
        """Test forcing input format with non-standard extension."""
        # Create a FITS file with unusual extension
        input_file: Path = tmp_path / "data.dat"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Transform with forced input format
        output_file: Path = tmp_path / "output.hdf5"
        Transform(
            sacc_file=input_file,
            input_format=SaccFormat.FITS,
            output=output_file,
        )

        assert output_file.exists()


class TestTransformIntegration:
    """Integration tests for Transform functionality."""

    def test_round_trip_conversion(self, tmp_path: Path) -> None:
        """Test converting FITS→HDF5→FITS preserves data."""
        original_file: Path = tmp_path / "original.fits"
        hdf5_file: Path = tmp_path / "converted.hdf5"
        final_file: Path = tmp_path / "final.fits"

        # Create original SACC file with some data
        s1: sacc.Sacc = sacc.Sacc()
        s1.add_tracer("misc", "tracer1")
        s1.add_tracer("misc", "tracer2")
        s1.save_fits(str(original_file), overwrite=False)

        # Convert FITS → HDF5
        Transform(
            sacc_file=original_file,
            output=hdf5_file,
            output_format=SaccFormat.HDF5,
        )
        assert hdf5_file.exists()

        # Convert HDF5 → FITS
        Transform(
            sacc_file=hdf5_file,
            output=final_file,
            output_format=SaccFormat.FITS,
        )
        assert final_file.exists()

        # Verify data is preserved
        s_final: sacc.Sacc = sacc.Sacc.load_fits(str(final_file))
        assert len(s_final.tracers) == 2

    def test_file_size_reporting(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test that file size reduction is reported."""
        input_file: Path = tmp_path / "test.fits"
        output_file: Path = tmp_path / "test.hdf5"

        # Create input file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        input_size: int = input_file.stat().st_size

        # Transform
        Transform(
            sacc_file=input_file,
            output=output_file,
        )

        output_size: int = output_file.stat().st_size

        # Capture output to verify summary is shown
        captured: CaptureResult[str] = capsys.readouterr()
        assert "Transformation successful!" in captured.out
        assert str(input_size) in captured.out or "bytes" in captured.out
        assert str(output_size) in captured.out or "bytes" in captured.out


class TestTransformFixOrdering:
    """Tests for Transform fix_ordering functionality."""

    def test_transform_with_fix_ordering_false(self, tmp_path: Path) -> None:
        """Test Transform without fixing ordering."""
        input_file: Path = tmp_path / "test.fits"
        output_file: Path = tmp_path / "test_output.fits"

        # Create a minimal SACC file
        s: sacc.Sacc = sacc.Sacc()
        z = np.linspace(0.0, 1.0, 10)
        dndz = np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)
        s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=10)
        s.save_fits(str(input_file), overwrite=False)

        # Transform without fix_ordering
        Transform(
            sacc_file=input_file,
            output=output_file,
            fix_ordering=False,
        )

        assert output_file.exists()
        s2: sacc.Sacc = sacc.Sacc.load_fits(str(output_file))
        assert len(s2.tracers) == 2

    def test_transform_with_fix_ordering_true(self, tmp_path: Path) -> None:
        """Test Transform with fix_ordering enabled."""
        input_file: Path = tmp_path / "test_ordering.fits"
        output_file: Path = tmp_path / "test_ordering_fixed.fits"

        # Create SACC with NZ tracers (needed for ordering checks)
        s: sacc.Sacc = sacc.Sacc()
        z = np.linspace(0.0, 1.0, 10)
        dndz = np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)
        s.add_tracer("NZ", "bin0", z, dndz)
        s.add_tracer("NZ", "bin1", z, dndz)
        s.add_data_point("galaxy_shear_cl_ee", ("bin0", "bin1"), 1.0, ell=10)
        s.save_fits(str(input_file), overwrite=False)

        # Transform with fix_ordering
        Transform(
            sacc_file=input_file,
            output=output_file,
            fix_ordering=True,
        )

        assert output_file.exists()

    def test_transform_overwrite_flag(self, tmp_path: Path) -> None:
        """Test Transform with overwrite flag."""
        input_file: Path = tmp_path / "test_overwrite.fits"

        # Create a SACC file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Transform with overwrite=True (should modify in place)
        # Small delay to ensure mtime changes if file is modified
        time.sleep(0.01)

        Transform(
            sacc_file=input_file,
            overwrite=True,
            output_format=SaccFormat.FITS,
        )

        # File should still exist
        assert input_file.exists()

    def test_transform_with_hdf5_format(self, tmp_path: Path) -> None:
        """Test Transform detecting and writing HDF5 format."""
        input_file: Path = tmp_path / "test.hdf5"
        output_file: Path = tmp_path / "test_output.hdf5"

        # Create a minimal SACC file in HDF5 format
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Transform with HDF5 input and output
        Transform(
            sacc_file=input_file,
            output=output_file,
            output_format=SaccFormat.HDF5,
        )

        assert output_file.exists()
        s2: sacc.Sacc = sacc.Sacc.load_hdf5(str(output_file))
        assert len(s2.tracers) == 1

    def test_detect_format_fits(self, tmp_path: Path) -> None:
        """Test format detection for FITS files."""
        input_file: Path = tmp_path / "test.fits"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Detect format
        detected = Transform.detect_format(input_file)
        assert detected == SaccFormat.FITS

    def test_detect_format_hdf5(self, tmp_path: Path) -> None:
        """Test format detection for HDF5 files."""
        input_file: Path = tmp_path / "test.hdf5"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Detect format
        detected = Transform.detect_format(input_file)
        assert detected == SaccFormat.HDF5

    def test_detect_format_h5_extension(self, tmp_path: Path) -> None:
        """Test format detection for .h5 extension."""
        input_file: Path = tmp_path / "test.h5"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Detect format
        detected = Transform.detect_format(input_file)
        assert detected == SaccFormat.HDF5


class TestTransformErrorHandling:
    """Tests for Transform error handling edge cases."""

    def test_detect_format_no_extension_tries_both(self, tmp_path: Path) -> None:
        """Test format detection without extension tries both formats."""
        # Create a FITS file without extension
        input_file: Path = tmp_path / "test_no_ext"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file))

        # Should detect as FITS after trying (covers lines 170-181)
        detected = Transform.detect_format(input_file)
        assert detected == SaccFormat.FITS

    def test_detect_format_no_extension_invalid_raises(self, tmp_path: Path) -> None:
        """Test that invalid file without extension raises ValueError."""
        input_file: Path = tmp_path / "invalid_no_ext"
        input_file.write_text("not a SACC file")

        # Should raise ValueError after trying both formats (covers lines 170-181)
        with pytest.raises(ValueError, match="Cannot detect format"):
            Transform.detect_format(input_file)

    def test_transform_with_explicit_output_format(self, tmp_path: Path) -> None:
        """Test transform with explicit output format specification."""
        # Create a test file
        input_file: Path = tmp_path / "test.fits"
        output_file: Path = tmp_path / "test.hdf5"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file))

        # Transform with specified output format (covers lines 200-201)
        Transform(
            sacc_file=input_file,
            output=output_file,
            output_format=SaccFormat.HDF5,
        )

        assert output_file.exists()

    def test_transform_with_explicit_input_format(self, tmp_path: Path) -> None:
        """Test transform with explicit input format specification."""
        # Create a test file with no extension
        input_file: Path = tmp_path / "test_data"
        output_file: Path = tmp_path / "test_out.hdf5"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file))

        # Transform with specified input format (covers lines 200-201, 220-222)
        Transform(
            sacc_file=input_file,
            output=output_file,
            input_format=SaccFormat.FITS,
            output_format=SaccFormat.HDF5,
        )

        assert output_file.exists()

    def test_transform_format_detection_error_exits(self, tmp_path: Path) -> None:
        """Test that format detection errors exit with error message."""
        # Create invalid file
        input_file: Path = tmp_path / "invalid"
        input_file.write_text("not a SACC file")

        # Should exit with sys.exit(1) when format detection fails (covers lines
        # 220-222)
        with pytest.raises(SystemExit) as exc_info:
            Transform(sacc_file=input_file)
        assert exc_info.value.code == 1

    def test_transform_read_invalid_format_exits(self, tmp_path: Path) -> None:
        """Test that reading with mismatched format specification exits with error."""
        # Create a valid FITS file
        input_file: Path = tmp_path / "test.fits"
        output_file: Path = tmp_path / "test_out.hdf5"
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file))

        # Try to read as HDF5 (should fail and exit) - covers lines 265-273
        with pytest.raises(SystemExit) as exc_info:
            Transform(
                sacc_file=input_file,
                output=output_file,
                input_format=SaccFormat.HDF5,  # Wrong format!
            )
        assert exc_info.value.code == 1

    def test_transform_write_to_invalid_path_exits(self, tmp_path: Path) -> None:
        """Test that write failures to invalid paths exit with error message."""
        # Create SACC data
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")

        input_file: Path = tmp_path / "test.fits"
        s.save_fits(str(input_file))

        # Try to write to invalid location (covers lines 289-296)
        invalid_output = Path("/invalid/nonexistent/path/output.fits")

        with pytest.raises(SystemExit) as exc_info:
            Transform(sacc_file=input_file, output=invalid_output)
        assert exc_info.value.code == 1

    def test_fix_ordering_no_issues_detected(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """Test fix_ordering when no issues exist."""
        # Create SACC data with proper ordering
        s: sacc.Sacc = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
        s.add_tracer("NZ", "src0", z, dndz)
        s.add_tracer("NZ", "lens0", z, dndz)

        # Add data with correct ordering (covers lines 365-379)
        ells = np.array([10, 20, 30])
        for ell in ells:
            s.add_data_point("galaxy_shear_cl_ee", ("src0", "src0"), 1.0, ell=int(ell))

        input_file: Path = tmp_path / "ordered.fits"
        output_file: Path = tmp_path / "ordered_out.fits"
        s.save_fits(str(input_file))

        Transform(sacc_file=input_file, output=output_file, fix_ordering=True)

        # Check console output
        captured = capsys.readouterr()
        assert "No tracer ordering issues detected" in captured.out

    def test_fix_ordering_with_corrections(self, tmp_path: Path) -> None:
        """Test fix_ordering detects and corrects actual tracer ordering violations.

        This test verifies the SACC tracer ordering convention:
        - src0 has SHEAR_E (from auto-correlation src0 × src0)
        - lens0 has COUNTS (from auto-correlation lens0 × lens0)
        - Since SHEAR_E < COUNTS, cross-correlation must be (src0, lens0)
        - Test creates data with WRONG order (lens0, src0) to trigger correction
        """
        s: sacc.Sacc = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)

        # Create tracers with specific measurement types
        s.add_tracer("NZ", "src0", z, dndz, quantity="galaxy_shear")
        s.add_tracer("NZ", "lens0", z, dndz, quantity="galaxy_density")

        ells = np.array([10, 20, 30])
        Cells = np.zeros_like(ells)

        # Add auto-correlations to establish measurement types
        # src0 × src0 → SHEAR_E
        s.add_ell_cl("galaxy_shear_cl_ee", "src0", "src0", ells, Cells)
        # lens0 × lens0 → COUNTS
        s.add_ell_cl("galaxy_density_cl", "lens0", "lens0", ells, Cells)
        # Add cross-correlation with WRONG tracer order (lens0, src0)
        # This violates convention: SHEAR_E < COUNTS means src0 must come first
        s.add_ell_cl("galaxy_shearDensity_cl_e", "lens0", "src0", ells, Cells)

        input_file: Path = tmp_path / "test_order.fits"
        output_file: Path = tmp_path / "test_order_out.fits"
        s.save_fits(str(input_file))

        # Transform with fix_ordering should detect and fix the violation
        transform_log: Path = tmp_path / "transform.log"
        with pytest.warns(DeprecationWarning, match="AUTO-CORRECTION PERFORMED"):
            Transform(
                sacc_file=input_file,
                output=output_file,
                fix_ordering=True,
                log_file=transform_log,
            )

        # Verify the transform completed successfully
        assert output_file.exists()

        # Extract output from transform, and remove all newlines for easier searching
        captured = transform_log.read_text()
        captured = captured.replace("\n", "")

        assert "Fixing tracer ordering" in captured
        assert "galaxy_shearDensity_cl_e" in captured
        assert "data points were flipped" in captured
        # Verify corrected file has proper ordering
        s_fixed: sacc.Sacc = sacc.Sacc.load_fits(str(output_file))
        cross_corr_points = [
            dp
            for dp in s_fixed.get_data_points()
            if dp.data_type == "galaxy_shearDensity_cl_e"
        ]
        # After correction, all cross-correlations should be (src0, lens0)
        for dp in cross_corr_points:
            assert dp.tracers == (
                "src0",
                "lens0",
            ), f"Expected (src0, lens0) but got {dp.tracers}"

    def test_fix_ordering_with_corrections_real_space(self, tmp_path: Path) -> None:
        """Test fix_ordering.

        This test detects and corrects tracer ordering violations in real space.

        This test verifies the SACC tracer ordering convention for real-space
        correlations:

        - src0 has SHEAR_T (from auto-correlation src0 × src0 with xi_t)
        - lens0 has COUNTS (from auto-correlation lens0 × lens0)
        - Since SHEAR_T < COUNTS, cross-correlation must be (src0, lens0)
        - Test creates data with WRONG order (lens0, src0) to trigger correction
        """
        s: sacc.Sacc = sacc.Sacc()
        z = np.linspace(0.0, 2.0, 50)
        dndz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)

        # Create tracers with specific measurement types
        s.add_tracer("NZ", "src0", z, dndz, quantity="galaxy_shear")
        s.add_tracer("NZ", "lens0", z, dndz, quantity="galaxy_density")

        thetas = np.array([1.0, 2.0, 3.0])
        xis = np.zeros_like(thetas)

        # Add auto-correlations to establish measurement types
        # src0 × src0 → SHEAR_T
        s.add_theta_xi("galaxy_shear_xi_t", "src0", "src0", thetas, xis)
        # lens0 × lens0 → COUNTS
        s.add_theta_xi("galaxy_density_xi", "lens0", "lens0", thetas, xis)
        # Add cross-correlation with WRONG tracer order (lens0, src0)
        # This violates convention: SHEAR_T < COUNTS means src0 must come first
        s.add_theta_xi("galaxy_shearDensity_xi_t", "lens0", "src0", thetas, xis)

        input_file: Path = tmp_path / "test_order_real.fits"
        output_file: Path = tmp_path / "test_order_real_out.fits"
        s.save_fits(str(input_file))

        # Transform with fix_ordering should detect and fix the violation
        transform_log: Path = tmp_path / "transform_real.log"
        with pytest.warns(DeprecationWarning, match="AUTO-CORRECTION PERFORMED"):
            Transform(
                sacc_file=input_file,
                output=output_file,
                fix_ordering=True,
                log_file=transform_log,
            )

        # Verify the transform completed successfully
        assert output_file.exists()

        # Extract output from transform, and remove all newlines for easier searching
        captured = transform_log.read_text()
        captured = captured.replace("\n", "")

        assert "Fixing tracer ordering" in captured
        assert "galaxy_shearDensity_xi_t" in captured
        assert "data points were flipped" in captured

        # Verify corrected file has proper ordering
        s_fixed: sacc.Sacc = sacc.Sacc.load_fits(str(output_file))
        cross_corr_points = [
            dp
            for dp in s_fixed.get_data_points()
            if dp.data_type == "galaxy_shearDensity_xi_t"
        ]
        # After correction, all cross-correlations should be (src0, lens0)
        for dp in cross_corr_points:
            assert dp.tracers == (
                "src0",
                "lens0",
            ), f"Expected (src0, lens0) but got {dp.tracers}"

    def test_read_sacc_data_with_unknown_format(self, tmp_path: Path) -> None:
        """Test _read_sacc_data raises ValueError for unknown format string.

        This test verifies that passing an unknown format string to _read_sacc_data
        triggers the default case in the match statement and raises a ValueError.
        """
        # Create a valid SACC file
        s: sacc.Sacc = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        input_file: Path = tmp_path / "test.fits"
        s.save_fits(str(input_file))

        # Create Transform instance
        transform = Transform(
            sacc_file=input_file,
            output=tmp_path / "test_out.fits",
        )

        # Call _read_sacc_data with an unknown format string
        # This should raise ValueError from the default case in the match statement
        with pytest.raises(ValueError, match="Unknown input format: unknown_format"):
            transform._read_sacc_data(  # pylint: disable=protected-access
                "unknown_format"
            )
