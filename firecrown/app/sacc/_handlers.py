"""Quality check handlers for SACC files."""

import re
from abc import ABC, abstractmethod


class OutputHandler(ABC):
    """Abstract base class for handling output from SACC operations.

    Each handler maintains internal state of matched issues and provides
    methods to report them.
    """

    def __init__(self):
        """Initialize the handler with empty state."""
        self._matched_issues: list[str] = []

    def count(self) -> int:
        """Return the number of issues handled by this handler.

        :return: Count of matched issues.
        """
        return len(self._matched_issues)

    @abstractmethod
    def get_title(self) -> str:
        """Get the title for reporting this issue type.

        :return: Title string for console output.
        """

    @abstractmethod
    def get_details(self) -> str | None:
        """Get detailed information about the handled issues.

        :return: Details string for console output, or None if no details.
        """


class MessageHandler(OutputHandler):
    """Handler for complete warning messages."""

    @abstractmethod
    def try_handle(self, message: str) -> bool:
        """Attempt to handle a warning message.

        :param message: The complete warning message to potentially handle.
        :return: True if this handler handled the message, False otherwise.
        """


class StreamHandler(OutputHandler):
    """Handler for line-based output streams (stdout/stderr)."""

    @abstractmethod
    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Attempt to handle lines from a stream.

        :param lines: List of lines from the stream to potentially handle.
        :return: Tuple of (handled, remaining_lines). If handled is True,
            the handler consumed some lines. remaining_lines are the lines
            that were not consumed and should be passed to the next handler.
        """


class TracerNamingViolationHandler(MessageHandler):
    """Handler for SACC convention violation warnings about tracer naming."""

    def __init__(self):
        """Initialize the tracer naming violation handler."""
        super().__init__()
        self._pattern = re.compile(
            (
                r"SACC Convention Violation Detected.*tracer "
                r"'(.*?)'.*tracer '(.*?)'.*data type string '(.*?)'"
            ),
            re.DOTALL | re.IGNORECASE,
        )

    def try_handle(self, message: str) -> bool:
        """Try to handle a tracer naming convention violation warning.

        :param message: The warning message to check.
        :return: True if handled, False otherwise.
        """
        match = self._pattern.search(message)
        if match:
            tracer1, tracer2, data_type = match.groups()
            formatted = (
                f"Tracers '{tracer1}' and '{tracer2}' (data type: '{data_type}')"
            )
            self._matched_issues.append(formatted)
            return True
        return False

    def get_title(self) -> str:
        """Get the title for tracer naming violations.

        :return: Title string with count.
        """
        return f"⚠️  Found {self.count()} tracer naming convention violation(s)"

    def get_details(self) -> str | None:
        """Get details of all tracer naming violations.

        :return: Formatted list of violations.
        """
        if not self._matched_issues:
            return None
        return "\n".join(f"  • {msg}" for msg in sorted(self._matched_issues))


class LegacyCovarianceHandler(MessageHandler):
    """Handler for legacy SACC covariance format warnings."""

    def __init__(self):
        """Initialize the legacy covariance handler."""
        super().__init__()
        self._pattern = re.compile(
            r"older sacc legacy sacc file format.*covariance",
            re.DOTALL | re.IGNORECASE,
        )

    def try_handle(self, message: str) -> bool:
        """Try to handle a legacy covariance format warning.

        :param message: The warning message to check.
        :return: True if handled, False otherwise.
        """
        if self._pattern.search(message):
            self._matched_issues.append("Legacy covariance format detected")
            return True
        return False

    def get_title(self) -> str:
        """Get the title for legacy covariance warnings.

        :return: Title string.
        """
        return "⚠️  Warning: Legacy covariance format detected"

    def get_details(self) -> str | None:
        """Get details for legacy covariance warnings.

        :return: Explanation text.
        """
        if not self._matched_issues:
            return None
        return (
            "  This SACC file uses an older internal format for covariance data.\n"
            "  Consider re-saving the file with a newer SACC version to ensure\n"
            "  compatibility and improved performance."
        )


class MissingSaccOrderingHandler(StreamHandler):
    """Handler for missing sacc_ordering metadata in stdout."""

    def __init__(self):
        """Initialize the missing sacc_ordering handler."""
        super().__init__()
        self._pattern = re.compile(
            r"sacc_ordering.*deprecated", re.IGNORECASE | re.DOTALL
        )

    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Try to handle lines about missing sacc_ordering.

        Looks for multi-line pattern containing 'sacc_ordering' and 'deprecated'.

        :param lines: List of lines from stdout.
        :return: Tuple of (handled, remaining_lines).
        """
        first_line = "The FITS format without the 'sacc_ordering'"
        second_line = "Assuming data rows are in the correct order"
        found_at = -1
        for i, (line, next_line) in enumerate(zip(lines, lines[1:])):
            if (first_line in line) and second_line in next_line:
                self._matched_issues.append("Missing sacc_ordering metadata")
                found_at = i
                break
        if found_at >= 0:
            # Remove the matched lines from the list
            c = found_at + 2
            remaining_lines = lines[:found_at] + lines[c:]
            return True, remaining_lines
        return False, lines

    def get_title(self) -> str:
        """Get the title for missing sacc_ordering.

        :return: Title string.
        """
        return "⚠️  Warning: Missing 'sacc_ordering' metadata"

    def get_details(self) -> str | None:
        """Get details for missing sacc_ordering.

        :return: Explanation text.
        """
        if not self._matched_issues:
            return None
        return (
            "  The 'sacc_ordering' column is missing from all data points.\n"
            "  This indicates an older SACC file format (pre-1.0).\n"
            "  Consider re-saving the file with a newer SACC version."
        )


class UnknownWarningHandler(MessageHandler):
    """Catch-all handler for unrecognized warnings."""

    def __init__(self):
        """Initialize the unknown warning handler."""
        super().__init__()
        self._warnings: list[tuple[str, str]] = []  # (category, message)

    def try_handle(self, message: str) -> bool:
        """Always handle any warning (catch-all).

        :param message: The warning message.
        :return: Always True.
        """
        self._warnings.append(("Unknown", message))
        return True

    def count(self) -> int:
        """Return the number of warnings handled.

        :return: Count of warnings.
        """
        return len(self._warnings)

    def get_title(self) -> str:
        """Get the title for unknown warnings.

        :return: Title string with count.
        """
        return f"⚠️  Found {self.count()} unknown warning(s)"

    def get_details(self) -> str | None:
        """Get details of all unknown warnings.

        :return: Formatted list of warnings.
        """
        if not self._warnings:
            return None
        details = []
        for idx, (category, message) in enumerate(self._warnings, 1):
            details.append(f"  Warning {idx}: {category}")
            details.append(f"    {message}")
        return "\n".join(details)


class UnknownStdoutHandler(StreamHandler):
    """Catch-all handler for unrecognized stdout lines."""

    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Always handle all remaining stdout lines (catch-all).

        :param lines: List of lines from stdout.
        :return: Tuple of (True, []) - consumes all lines.
        """
        for line in lines:
            if line.strip():  # Only capture non-empty lines
                self._matched_issues.append(line)
        return True, []

    def get_title(self) -> str:
        """Get the title for unknown stdout.

        :return: Title string.
        """
        return "⚠️  Unknown output from SACC library (stdout):"

    def get_details(self) -> str | None:
        """Get details of all unknown stdout lines.

        :return: Formatted list of lines.
        """
        if not self._matched_issues:
            return None
        return "\n".join(f"  {line}" for line in self._matched_issues)


class UnknownStderrHandler(StreamHandler):
    """Catch-all handler for unrecognized stderr lines."""

    def try_handle(self, lines: list[str]) -> tuple[bool, list[str]]:
        """Always handle all stderr lines (catch-all).

        :param lines: List of lines from stderr.
        :return: Tuple of (True, []) - consumes all lines.
        """
        for line in lines:
            if line.strip():  # Only capture non-empty lines
                self._matched_issues.append(line)
        return True, []

    def get_title(self) -> str:
        """Get the title for unknown stderr.

        :return: Title string.
        """
        return "⚠️  Unknown output from SACC library (stderr):"

    def get_details(self) -> str | None:
        """Get details of all unknown stderr lines.

        :return: Formatted list of lines.
        """
        if not self._matched_issues:
            return None
        return "\n".join(f"  {line}" for line in self._matched_issues)


# Handler types for different output streams
WARNING_HANDLERS: list[type[MessageHandler]] = [
    TracerNamingViolationHandler,
    LegacyCovarianceHandler,
    UnknownWarningHandler,  # Must be last (catch-all)
]

STDOUT_HANDLERS: list[type[StreamHandler]] = [
    MissingSaccOrderingHandler,
    UnknownStdoutHandler,  # Must be last (catch-all)
]

STDERR_HANDLERS: list[type[StreamHandler]] = [
    UnknownStderrHandler,  # Catch-all for stderr
]
