"""Unit tests for refactored fctools helper functions.

Tests the helper functions extracted from fctools modules to reduce
cyclomatic complexity. Covers timing data parsing and class code building.
"""

import contextlib
import io
import json

from rich.console import Console

from firecrown.fctools import coverage_to_tsv as ctv
from firecrown.fctools import print_code as pc


def test_parse_duration_line(tmp_path):
    # private helper is nested; test via the public parse_timing_data API
    console = Console()
    tmp = tmp_path / "tmp_test_durations.txt"
    content = (
        "0.12s call tests/test_mod.py::test_func\n"
        "invalid line\n1.5s call tests/test_mod.py::test_other"
    )
    tmp.write_text(content, encoding="utf-8")
    timings = ctv.parse_timing_data(console, tmp)
    assert "tests/test_mod.py::test_func" in timings
    assert timings["tests/test_mod.py::test_func"] == 0.12


def test_load_json_timing(tmp_path):
    console = Console()
    data = {"tests": [{"nodeid": "tests/t.py::test_a", "duration": 0.5}]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data))
    timings = ctv.parse_timing_data(console, p)
    assert timings["tests/t.py::test_a"] == 0.5


class Dummy:
    r"""Docline1

    Docline2
    """

    A = 1


def test_build_class_code():
    # Capture stdout of the public display function and assert the content
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pc.display_class_without_markdown(Dummy)
    output = buf.getvalue()
    assert "class Dummy" in output
    assert "A = 1" in output
