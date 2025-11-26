"""Unit tests for firecrown.app.sacc._handlers module.

Tests for handler classes that process SACC operations output including warnings,
stdout, and stderr streams.
"""

from firecrown.app.sacc._handlers import (
    TracerNamingViolationHandler,
    LegacyCovarianceHandler,
    MissingSaccOrderingHandler,
    UnknownWarningHandler,
    UnknownStdoutHandler,
    UnknownStderrHandler,
)


class TestTracerNamingViolationHandler:
    """Tests for TracerNamingViolationHandler."""

    def test_try_handle_matching_message(self) -> None:
        """Test handling of a tracer naming violation warning."""
        handler = TracerNamingViolationHandler()
        message = """
SACC Convention Violation Detected (DEPRECATED AUTO-FIX)

Firecrown detected an inconsistency in how measurement types are assigned to tracers.
Specifically, assigning measurement type 'SHEAR_E' to tracer 'src0' and measurement
type 'COUNTS' to tracer 'lens0' would create mixed-type measurements (multiple distinct
measurement types in the same tomographic bin).

The data type string 'galaxy_shearDensity_cl_e' follows the SACC naming convention, where the order
of measurement types in the string must match the order of tracers. However, your SACC
file/object appears to violate this convention.
"""
        result = handler.try_handle(message)

        assert result is True
        assert handler.count() == 1
        details = handler.get_details()
        assert details is not None
        assert "src0" in details
        assert "lens0" in details
        assert "galaxy_shearDensity_cl_e" in details

    def test_try_handle_non_matching_message(self) -> None:
        """Test non-matching warning message."""
        handler = TracerNamingViolationHandler()
        message = "This is some other warning message"
        result = handler.try_handle(message)

        assert result is False
        assert handler.count() == 0
        assert handler.get_details() is None

    def test_multiple_violations(self) -> None:
        """Test handling multiple violation messages."""
        handler = TracerNamingViolationHandler()
        msg1 = """
SACC Convention Violation Detected (DEPRECATED AUTO-FIX)

Firecrown detected an inconsistency in how measurement types are assigned to tracers.
Specifically, assigning measurement type 'SHEAR_E' to tracer 'src0' and measurement
type 'COUNTS' to tracer 'lens0' would create mixed-type measurements.

The data type string 'galaxy_shearDensity_cl_e' follows the SACC naming convention.
"""
        msg2 = """
SACC Convention Violation Detected (DEPRECATED AUTO-FIX)

Firecrown detected an inconsistency in how measurement types are assigned to tracers.
Specifically, assigning measurement type 'SHEAR_T' to tracer 'src1' and measurement
type 'COUNTS' to tracer 'lens1' would create mixed-type measurements.

The data type string 'galaxy_shearDensity_cl_t' follows the SACC naming convention.
"""
        handler.try_handle(msg1)
        handler.try_handle(msg2)

        assert handler.count() == 2
        details = handler.get_details()
        assert details is not None
        assert "src0" in details
        assert "lens0" in details
        assert "src1" in details
        assert "lens1" in details

    def test_get_title(self) -> None:
        """Test title generation."""
        handler = TracerNamingViolationHandler()
        message = """
SACC Convention Violation Detected (DEPRECATED AUTO-FIX)

Firecrown detected an inconsistency in how measurement types are assigned to tracers.
Specifically, assigning measurement type 'SHEAR_E' to tracer 'src0' and measurement
type 'COUNTS' to tracer 'lens0' would create mixed-type measurements.

The data type string 'galaxy_shearDensity_cl_e' follows the SACC naming convention.
"""
        handler.try_handle(message)
        title = handler.get_title()

        assert "tracer naming convention violation" in title
        assert "1" in title


class TestLegacyCovarianceHandler:
    """Tests for LegacyCovarianceHandler."""

    def test_try_handle_matching_message(self) -> None:
        """Test handling of legacy covariance warning."""
        handler = LegacyCovarianceHandler()
        message = "Warning: You are reading an older sacc legacy sacc file format. The covariance matrix..."
        result = handler.try_handle(message)

        assert result is True
        assert handler.count() == 1
        details = handler.get_details()
        assert details is not None
        assert "older internal format" in details or "SACC file uses" in details

    def test_try_handle_non_matching_message(self) -> None:
        """Test non-matching message."""
        handler = LegacyCovarianceHandler()
        message = "Some other warning about something else"
        result = handler.try_handle(message)

        assert result is False
        assert handler.count() == 0
        assert handler.get_details() is None

    def test_get_title(self) -> None:
        """Test title generation."""
        handler = LegacyCovarianceHandler()
        message = "Warning: You are reading an older sacc legacy sacc file format. The covariance matrix..."
        handler.try_handle(message)
        title = handler.get_title()

        assert "Legacy covariance format" in title

    def test_multiple_detections(self) -> None:
        """Test multiple legacy format detections."""
        handler = LegacyCovarianceHandler()
        msg1 = "Warning: You are reading an older sacc legacy sacc file format. The covariance matrix..."
        msg2 = "Another message about older sacc legacy sacc file format and covariance"

        handler.try_handle(msg1)
        handler.try_handle(msg2)

        assert handler.count() == 2


class TestMissingSaccOrderingHandler:
    """Tests for MissingSaccOrderingHandler."""

    def test_try_handle_matching_lines(self) -> None:
        """Test handling missing sacc_ordering warning."""
        handler = MissingSaccOrderingHandler()
        lines = [
            "Some initial content",
            "The FITS format without the 'sacc_ordering' column is deprecated.",
            "Assuming data rows are in the correct order according to the internal ordering.",
            "Some trailing content",
        ]
        handled, remaining = handler.try_handle(lines)

        assert handled is True
        assert handler.count() == 1
        assert len(remaining) == 2  # Removed the two matched lines
        assert remaining[0] == "Some initial content"
        assert remaining[1] == "Some trailing content"

    def test_try_handle_non_matching_lines(self) -> None:
        """Test non-matching lines."""
        handler = MissingSaccOrderingHandler()
        lines = [
            "Some random content",
            "More random content",
        ]
        handled, remaining = handler.try_handle(lines)

        assert handled is False
        assert remaining == lines
        assert handler.count() == 0

    def test_get_title(self) -> None:
        """Test title generation."""
        handler = MissingSaccOrderingHandler()
        lines = [
            "The FITS format without the 'sacc_ordering' column is deprecated.",
            "Assuming data rows are in the correct order according to the internal ordering.",
        ]
        handler.try_handle(lines)
        title = handler.get_title()

        assert "sacc_ordering" in title

    def test_get_details(self) -> None:
        """Test details generation."""
        handler = MissingSaccOrderingHandler()
        lines = [
            "The FITS format without the 'sacc_ordering' column is deprecated.",
            "Assuming data rows are in the correct order according to the internal ordering.",
        ]
        handler.try_handle(lines)
        details = handler.get_details()

        assert details is not None
        assert "older SACC file format" in details
        assert "pre-1.0" in details


class TestUnknownWarningHandler:
    """Tests for UnknownWarningHandler."""

    def test_try_handle_any_message(self) -> None:
        """Test that unknown handler catches any warning."""
        handler = UnknownWarningHandler()
        message = "Some completely random warning message"
        result = handler.try_handle(message)

        assert result is True
        assert handler.count() == 1

    def test_multiple_warnings(self) -> None:
        """Test handling multiple warnings."""
        handler = UnknownWarningHandler()
        handler.try_handle("Warning 1")
        handler.try_handle("Warning 2")
        handler.try_handle("Warning 3")

        assert handler.count() == 3
        details = handler.get_details()
        assert details is not None
        assert "Warning 1" in details
        assert "Warning 2" in details
        assert "Warning 3" in details

    def test_get_title(self) -> None:
        """Test title generation includes count."""
        handler = UnknownWarningHandler()
        handler.try_handle("Warning 1")
        handler.try_handle("Warning 2")
        title = handler.get_title()

        assert "2" in title
        assert "unknown warning" in title

    def test_get_details_empty(self) -> None:
        """Test details when no warnings captured."""
        handler = UnknownWarningHandler()
        details = handler.get_details()

        assert details is None


class TestUnknownStdoutHandler:
    """Tests for UnknownStdoutHandler."""

    def test_try_handle_lines(self) -> None:
        """Test handling stdout lines."""
        handler = UnknownStdoutHandler()
        lines = [
            "stdout line 1",
            "stdout line 2",
            "",  # Empty line should be ignored
            "stdout line 3",
        ]
        handled, remaining = handler.try_handle(lines)

        assert handled is True
        assert remaining == []
        assert handler.count() == 3  # Only non-empty lines

    def test_try_handle_empty_lines(self) -> None:
        """Test handling only empty lines."""
        handler = UnknownStdoutHandler()
        lines = ["", "", ""]
        handled, remaining = handler.try_handle(lines)

        assert handled is True
        assert remaining == []
        assert handler.count() == 0  # Empty lines not captured

    def test_get_title(self) -> None:
        """Test title generation."""
        handler = UnknownStdoutHandler()
        lines = ["stdout content"]
        handler.try_handle(lines)
        title = handler.get_title()

        assert "Unknown output" in title
        assert "stdout" in title

    def test_get_details(self) -> None:
        """Test details generation."""
        handler = UnknownStdoutHandler()
        lines = ["line 1", "line 2"]
        handler.try_handle(lines)
        details = handler.get_details()

        assert details is not None
        assert "line 1" in details
        assert "line 2" in details


class TestUnknownStderrHandler:
    """Tests for UnknownStderrHandler."""

    def test_try_handle_lines(self) -> None:
        """Test handling stderr lines."""
        handler = UnknownStderrHandler()
        lines = [
            "stderr line 1",
            "stderr line 2",
            "",  # Empty line should be ignored
            "stderr line 3",
        ]
        handled, remaining = handler.try_handle(lines)

        assert handled is True
        assert remaining == []
        assert handler.count() == 3  # Only non-empty lines

    def test_try_handle_empty_lines(self) -> None:
        """Test handling only empty lines."""
        handler = UnknownStderrHandler()
        lines = ["", "", ""]
        handled, remaining = handler.try_handle(lines)

        assert handled is True
        assert remaining == []
        assert handler.count() == 0  # Empty lines not captured

    def test_get_title(self) -> None:
        """Test title generation."""
        handler = UnknownStderrHandler()
        lines = ["stderr content"]
        handler.try_handle(lines)
        title = handler.get_title()

        assert "Unknown output" in title
        assert "stderr" in title

    def test_get_details(self) -> None:
        """Test details generation."""
        handler = UnknownStderrHandler()
        lines = ["error 1", "error 2"]
        handler.try_handle(lines)
        details = handler.get_details()

        assert details is not None
        assert "error 1" in details
        assert "error 2" in details
