"""Unit tests for firecrown.app.logging module."""

from pathlib import Path
from typing import Generator
import pytest
from firecrown.app.logging import Logging


@pytest.fixture(name="logging_no_file")
def fixture_logging_no_file() -> Logging:
    """Create Logging instance without file."""
    return Logging()


@pytest.fixture(name="logging_with_file")
def fixture_logging_with_file(
    tmp_path: Path,
) -> Generator[tuple[Logging, Path], None, None]:
    """Create Logging instance with file."""
    log_file = tmp_path / "test.log"
    logging = Logging(log_file=log_file)
    yield logging, log_file
    del logging


def test_logging_without_file(logging_no_file: Logging) -> None:
    """Test Logging initialization without log file."""
    assert logging_no_file.log_file is None
    assert logging_no_file.console_io is None
    assert logging_no_file.console is not None


def test_logging_with_file(logging_with_file: tuple[Logging, Path]) -> None:
    """Test Logging initialization with log file."""
    logging, log_file = logging_with_file

    assert logging.log_file == log_file
    assert logging.console_io is not None
    assert logging.console is not None

    logging.console.print("test message")


def test_logging_file_written(logging_with_file: tuple[Logging, Path]) -> None:
    """Test that console output is written to file."""
    logging, log_file = logging_with_file

    logging.console.print("test message")
    del logging

    assert log_file.exists()
    assert "test message" in log_file.read_text()


def test_logging_file_cleanup(tmp_path: Path) -> None:
    """Test that file handle is closed on deletion."""
    log_file = tmp_path / "test.log"
    logging = Logging(log_file=log_file)

    console_io = logging.console_io
    assert console_io is not None
    assert not console_io.closed

    del logging
    assert console_io.closed


def test_logging_console_output_to_file(tmp_path: Path) -> None:
    """Test that console output is written to file."""
    log_file = tmp_path / "output.log"
    logging = Logging(log_file=log_file)

    logging.console.print("Line 1")
    logging.console.print("Line 2")
    logging.console.print("[bold]Formatted[/bold] text")

    del logging

    content = log_file.read_text()
    assert "Line 1" in content
    assert "Line 2" in content
    assert "Formatted" in content


def test_logging_multiple_instances(tmp_path: Path) -> None:
    """Test multiple Logging instances with different files."""
    log1 = tmp_path / "log1.log"
    log2 = tmp_path / "log2.log"

    logging1 = Logging(log_file=log1)
    logging2 = Logging(log_file=log2)

    logging1.console.print("Message 1")
    logging2.console.print("Message 2")

    del logging1
    del logging2

    assert "Message 1" in log1.read_text()
    assert "Message 2" in log2.read_text()
    assert "Message 2" not in log1.read_text()
    assert "Message 1" not in log2.read_text()
