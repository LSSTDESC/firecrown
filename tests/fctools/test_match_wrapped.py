"""Tests for match_wrapped function in this package."""

from . import match_wrapped


def test_match_wrapped():
    text = (
        "Reading coverage data from \n"
        "/tmp/pytest-of-runner/pytest-0/popen-gw0/"
        "test_main_direct_with_timing0/coverage.\n"
        "json...\n"
        "Reading timing data from \n"
        "/tmp/pytest-of-runner/pytest-0/popen-gw0/"
        "test_main_direct_with_timing0/timing.tx\n"
        "t...\n"
        "Loaded timing data for 1 tests\n"
        "Extracting function-level coverage data...\n"
        "Writing 1 records to \n"
        "/tmp/pytest-of-runner/pytest-0/popen-gw0/"
        "test_main_direct_with_timing0/output.ts\n"
        "v...\n"
        "Successfully converted coverage data to TSV format!\n"
        "Output file: \n"
        "/tmp/pytest-of-runner/pytest-0/popen-gw0/"
        "test_main_direct_with_timing0/output.ts\n"
        "v\n"
        "Records written: 1\n"
        "Records with timing data: 1\n"
    )
    assert match_wrapped(text, "Reading coverage data from")
    assert match_wrapped(text, "Reading timing data from")
    assert match_wrapped(text, "timing.txt")
