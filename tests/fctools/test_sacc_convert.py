"""Unit tests for firecrown.fctools.sacc_convert module.

Tests the SACC file format conversion tool.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Check if sacc is available
pytest.importorskip("sacc")

import sacc  # noqa: E402  # pylint: disable=wrong-import-position

# pylint: disable=wrong-import-position
from firecrown.fctools.sacc_convert import (  # noqa: E402
    _display_conversion_summary,
    _read_and_convert_file,
    detect_format,
    determine_output_path,
    main,
)


class TestDetectFormat:  # pylint: disable=import-outside-toplevel
    """Tests for detect_format function."""

    def test_detect_fits(self):
        """Test detecting FITS format."""
        path = Path("test.fits")
        assert detect_format(path) == "fits"

    def test_detect_fits_uppercase(self):
        """Test detecting FITS format with uppercase extension."""
        path = Path("test.FITS")
        assert detect_format(path) == "fits"

    def test_detect_hdf5(self):
        """Test detecting HDF5 format with .hdf5 extension."""
        path = Path("test.hdf5")
        assert detect_format(path) == "hdf5"

    def test_detect_h5(self):
        """Test detecting HDF5 format with .h5 extension."""
        path = Path("test.h5")
        assert detect_format(path) == "hdf5"

    def test_detect_h5_uppercase(self):
        """Test detecting HDF5 format with uppercase extension."""
        path = Path("test.H5")
        assert detect_format(path) == "hdf5"

    def test_unknown_extension_raises(self):
        """Test that unknown extension raises ValueError."""
        path = Path("test.dat")
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(path)

    def test_no_extension_raises(self):
        """Test that file with no extension raises ValueError."""
        path = Path("testfile")
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(path)


class TestDetermineOutputPath:
    """Tests for determine_output_path function."""

    def test_uses_provided_output(self):
        """Test that provided output path is used."""
        input_path = Path("input.fits")
        output = Path("custom_output.hdf5")
        result = determine_output_path(input_path, output, "hdf5")
        assert result == output

    def test_auto_generate_fits_to_hdf5(self):
        """Test auto-generating HDF5 output from FITS input."""
        input_path = Path("/data/test.fits")
        result = determine_output_path(input_path, None, "hdf5")
        assert result == Path("/data/test.hdf5")

    def test_auto_generate_hdf5_to_fits(self):
        """Test auto-generating FITS output from HDF5 input."""
        input_path = Path("/data/test.hdf5")
        result = determine_output_path(input_path, None, "fits")
        assert result == Path("/data/test.fits")

    def test_preserves_directory(self):
        """Test that output stays in same directory as input."""
        input_path = Path("/some/path/data.h5")
        result = determine_output_path(input_path, None, "fits")
        assert result.parent == input_path.parent
        assert result.name == "data.fits"


class TestReadAndConvertFile:  # pylint: disable=import-outside-toplevel
    """Tests for _read_and_convert_file function."""

    def test_convert_fits_to_hdf5(self, tmp_path):
        """Test converting FITS to HDF5."""
        input_file = tmp_path / "test.fits"
        output_file = tmp_path / "test.hdf5"

        # Create a minimal SACC file in FITS format
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Convert
        _read_and_convert_file(input_file, "fits", output_file, "hdf5", False)

        # Verify output exists and can be read
        assert output_file.exists()
        s2 = sacc.Sacc.load_hdf5(str(output_file))
        assert len(s2.tracers) == 1

    def test_convert_hdf5_to_fits(self, tmp_path):
        """Test converting HDF5 to FITS."""
        input_file = tmp_path / "test.hdf5"
        output_file = tmp_path / "test.fits"

        # Create a minimal SACC file in HDF5 format
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        # Convert
        _read_and_convert_file(input_file, "hdf5", output_file, "fits", False)

        # Verify output exists and can be read
        assert output_file.exists()
        s2 = sacc.Sacc.load_fits(str(output_file))
        assert len(s2.tracers) == 1

    def test_invalid_fits_file_exits(self, tmp_path, capsys):
        """Test that invalid FITS file causes exit."""
        input_file = tmp_path / "invalid.fits"
        output_file = tmp_path / "output.hdf5"

        # Create a file that's not a valid SACC FITS file
        input_file.write_text("not a fits file")

        with pytest.raises(SystemExit) as exc_info:
            _read_and_convert_file(input_file, "fits", output_file, "hdf5", False)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ERROR: Failed to read input file" in captured.out

    def test_invalid_hdf5_file_exits(self, tmp_path, capsys):
        """Test that invalid HDF5 file causes exit."""
        input_file = tmp_path / "invalid.hdf5"
        output_file = tmp_path / "output.fits"

        # Create a file that's not a valid SACC HDF5 file
        input_file.write_text("not an hdf5 file")

        with pytest.raises(SystemExit) as exc_info:
            _read_and_convert_file(input_file, "hdf5", output_file, "fits", False)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ERROR: Failed to read input file" in captured.out

    def test_write_error_exits(self, tmp_path, capsys):
        """Test that write error causes exit."""
        input_file = tmp_path / "test.fits"
        output_dir = tmp_path / "nonexistent_dir"
        output_file = output_dir / "test.hdf5"

        # Create valid input
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Try to write to non-existent directory
        with pytest.raises(SystemExit) as exc_info:
            _read_and_convert_file(input_file, "fits", output_file, "hdf5", False)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ERROR: Failed to write SACC data" in captured.err


class TestDisplayConversionSummary:
    """Tests for _display_conversion_summary function."""

    def test_displays_summary(self, tmp_path, capsys):
        """Test that summary is displayed with file info."""
        input_file = tmp_path / "input.fits"
        output_file = tmp_path / "output.hdf5"

        # Create files with different sizes
        input_file.write_text("x" * 1000)
        output_file.write_text("y" * 800)

        _display_conversion_summary(input_file, "fits", output_file, "hdf5")

        captured = capsys.readouterr()
        assert "Conversion successful!" in captured.out
        assert "input.fits" in captured.out
        assert "output.hdf5" in captured.out
        assert "FITS" in captured.out
        assert "HDF5" in captured.out

    def test_shows_size_reduction(self, tmp_path, capsys):
        """Test that size reduction is shown."""
        input_file = tmp_path / "input.fits"
        output_file = tmp_path / "output.hdf5"

        input_file.write_text("x" * 1000)
        output_file.write_text("y" * 500)  # 50% smaller

        _display_conversion_summary(input_file, "fits", output_file, "hdf5")

        captured = capsys.readouterr()
        assert "Size reduction:" in captured.out
        assert "50.0%" in captured.out

    def test_shows_size_increase(self, tmp_path, capsys):
        """Test that size increase is shown."""
        input_file = tmp_path / "input.fits"
        output_file = tmp_path / "output.hdf5"

        input_file.write_text("x" * 500)
        output_file.write_text("y" * 1000)  # 100% larger

        _display_conversion_summary(input_file, "fits", output_file, "hdf5")

        captured = capsys.readouterr()
        assert "Size increase:" in captured.out
        assert "100.0%" in captured.out

    def test_shows_unchanged_size(self, tmp_path, capsys):
        """Test that unchanged size is shown."""
        input_file = tmp_path / "input.fits"
        output_file = tmp_path / "output.hdf5"

        input_file.write_text("x" * 1000)
        output_file.write_text("y" * 1000)  # Same size

        _display_conversion_summary(input_file, "fits", output_file, "hdf5")

        captured = capsys.readouterr()
        assert "Size unchanged" in captured.out


class TestMainFunction:  # pylint: disable=import-outside-toplevel
    """Tests for main CLI function."""

    def test_main_converts_fits_to_hdf5(self, tmp_path):
        """Test main function converts FITS to HDF5."""
        input_file = tmp_path / "test.fits"

        # Create input file
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 0
        assert "Detected input format: FITS" in result.output
        assert "Conversion successful!" in result.output

        # Check output file was created
        output_file = tmp_path / "test.hdf5"
        assert output_file.exists()

    def test_main_converts_hdf5_to_fits(self, tmp_path):
        """Test main function converts HDF5 to FITS."""
        input_file = tmp_path / "test.hdf5"

        # Create input file
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 0
        assert "Detected input format: HDF5" in result.output
        assert "Conversion successful!" in result.output

        # Check output file was created
        output_file = tmp_path / "test.fits"
        assert output_file.exists()

    def test_main_with_custom_output(self, tmp_path):
        """Test main with custom output filename."""
        input_file = tmp_path / "input.fits"
        output_file = tmp_path / "custom_name.hdf5"

        # Create input file
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file), "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_main_with_forced_input_format(self, tmp_path):
        """Test main with forced input format."""
        # Create a FITS file with unusual extension
        input_file = tmp_path / "data.dat"

        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file), "--input-format", "fits"])

        assert result.exit_code == 0
        assert "Using specified input format: FITS" in result.output

    def test_main_refuses_overwrite_without_flag(self, tmp_path):
        """Test that existing output file is not overwritten without flag."""
        input_file = tmp_path / "test.fits"
        output_file = tmp_path / "test.hdf5"

        # Create input file
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Create existing output file
        output_file.write_text("existing content")

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 1
        assert "already exists" in result.output
        assert "Use --overwrite" in result.output

        # Verify output wasn't changed
        assert output_file.read_text() == "existing content"

    def test_main_overwrites_with_flag(self, tmp_path):
        """Test that existing output is overwritten with --overwrite flag."""
        input_file = tmp_path / "test.fits"
        output_file = tmp_path / "test.hdf5"

        # Create input file
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        # Create existing output file
        output_file.write_text("existing content")

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file), "--overwrite"])

        assert result.exit_code == 0
        assert "Conversion successful!" in result.output

        # Verify output was overwritten (and is now a valid SACC file)
        s2 = sacc.Sacc.load_hdf5(str(output_file))
        assert len(s2.tracers) == 1

    def test_main_invalid_format_detection(self, tmp_path):
        """Test error handling for undetectable format."""
        input_file = tmp_path / "test.dat"
        input_file.write_text("some content")

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 1
        assert "Cannot detect format" in result.output

    def test_main_nonexistent_input_file(self):
        """Test error with nonexistent input file."""
        runner = CliRunner()
        result = runner.invoke(main, ["nonexistent.fits"])

        assert result.exit_code != 0
        # Click handles this error automatically

    def test_main_with_subprocess(self, tmp_path):
        """Test that the script can be executed directly via subprocess."""
        input_file = tmp_path / "test.fits"

        # Create input file
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_fits(str(input_file), overwrite=False)

        script_path = "firecrown/fctools/sacc_convert.py"
        result = subprocess.run(
            [sys.executable, script_path, str(input_file)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Conversion successful!" in result.stdout

        # Verify output file was created
        output_file = tmp_path / "test.hdf5"
        assert output_file.exists()


class TestIntegration:
    """Integration tests for SACC conversion functionality."""

    def test_round_trip_conversion(self, tmp_path):
        """Test converting FITS→HDF5→FITS preserves data."""
        original_file = tmp_path / "original.fits"
        hdf5_file = tmp_path / "converted.hdf5"
        final_file = tmp_path / "final.fits"

        # Create original SACC file with some data
        s1 = sacc.Sacc()
        s1.add_tracer("misc", "tracer1")
        s1.add_tracer("misc", "tracer2")
        s1.save_fits(str(original_file), overwrite=False)

        # Convert FITS → HDF5
        runner = CliRunner()
        result1 = runner.invoke(main, [str(original_file), "--output", str(hdf5_file)])
        assert result1.exit_code == 0

        # Convert HDF5 → FITS
        result2 = runner.invoke(main, [str(hdf5_file), "--output", str(final_file)])
        assert result2.exit_code == 0

        # Verify data is preserved
        s_final = sacc.Sacc.load_fits(str(final_file))
        assert len(s_final.tracers) == 2

    def test_cli_with_all_options(self, tmp_path):
        """Test CLI with all options combined."""
        input_file = tmp_path / "data.dat"
        output_file = tmp_path / "output.hdf5"

        # Create input as FITS with non-standard extension
        s = sacc.Sacc()
        s.add_tracer("misc", "test_tracer")
        s.save_fits(str(input_file), overwrite=False)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(input_file),
                "--output",
                str(output_file),
                "--input-format",
                "fits",
                "--overwrite",
            ],
        )

        assert result.exit_code == 0
        assert "Using specified input format: FITS" in result.output
        assert "Conversion successful!" in result.output
        assert output_file.exists()

    def test_handles_h5_extension(self, tmp_path):
        """Test that .h5 extension is properly handled."""
        input_file = tmp_path / "test.h5"

        # Create HDF5 file with .h5 extension
        s = sacc.Sacc()
        s.add_tracer("misc", "tracer1")
        s.save_hdf5(str(input_file))

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 0
        assert "Detected input format: HDF5" in result.output

        # Check FITS output was created
        output_file = tmp_path / "test.fits"
        assert output_file.exists()
