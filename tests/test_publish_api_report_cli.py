"""Offline tests for publisher summary failures and the executable entry point."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_publish_api_report import FakeGitHub, REPOSITORY
from tools.publish_api_report import main


@pytest.mark.parametrize("failure", ["missing", "directory"])
def test_summary_write_failure_changes_success_to_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    """Fail visibly when publication succeeds but its summary cannot be written.

    :param tmp_path: Isolated event file and unwritable summary destination.
    :param capsys: Captured publisher diagnostics.
    :param failure: Missing summary environment key or directory destination.
    """
    fake = FakeGitHub()
    event = tmp_path / "event.json"
    event.write_text(json.dumps(fake.event), encoding="utf-8")
    environment = {
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_TOKEN": "test-token",
    }
    if failure == "directory":
        environment["GITHUB_STEP_SUMMARY"] = str(tmp_path)
    assert main(environment, fake.transport) == 1
    assert len(fake.writes) == 1
    diagnostics = capsys.readouterr().err
    assert "Published public API report for PR #7" in diagnostics
    assert "Cannot write publisher summary:" in diagnostics


def test_executable_entry_point_reports_invalid_event_without_network(
    tmp_path: Path,
) -> None:
    """Exercise process exit and real environment handling before any HTTP request.

    :param tmp_path: Isolated malformed workflow event and writable summary file.
    """
    event, summary = tmp_path / "event.json", tmp_path / "summary.md"
    # A non-object event fails before the default HTTP transport can be invoked.
    event.write_text("[]", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools/publish_api_report.py"),
        ],
        env={
            **os.environ,
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert (
        "Public API report publication failed: Expected a JSON object" in result.stderr
    )
    assert (
        "Public API report publication failed: Expected a JSON object"
        in summary.read_text(encoding="utf-8")
    )
