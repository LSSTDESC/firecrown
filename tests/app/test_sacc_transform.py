"""Unit tests for firecrown.app.sacc.Transform module.

Tests the SACC file transform command.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from pathlib import Path

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
