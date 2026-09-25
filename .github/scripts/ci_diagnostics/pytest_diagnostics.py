"""Temporary CI evidence collection, loaded only by diagnose-ci.sh."""

import json
import os
from pathlib import Path
import subprocess
import time

import pytest


def _record(event, **fields):
    """Keep separate, immediately flushed records for every pytest process."""
    directory = os.environ.get("FIRECROWN_DIAGNOSTICS_DIR")
    if directory:
        with (Path(directory) / f"pytest-{os.getpid()}.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    {"time": time.time(), "pid": os.getpid(), "event": event, **fields}
                )
                + "\n"
            )


def pytest_sessionstart():
    """Identify sessions and xdist workers without recording environment secrets."""
    _record("session_start", worker=os.environ.get("PYTEST_XDIST_WORKER", "controller"))


def pytest_runtest_logstart(nodeid):
    """Record actual start times before make buffers terminal output."""
    _record("test_start", nodeid=nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Expose captured child output when subprocess.run(check=True) fails."""
    outcome = yield
    report = outcome.get_result()
    _record(
        "test_report", nodeid=item.nodeid, phase=report.when, outcome=report.outcome
    )
    if call.excinfo and isinstance(call.excinfo.value, subprocess.CalledProcessError):
        error = call.excinfo.value
        # Do not record command arguments or environment variables. These test
        # subprocesses consume generated example inputs, not credentials.
        _record("subprocess_failure", nodeid=item.nodeid, returncode=error.returncode)
        for name in ("stdout", "stderr"):
            value = getattr(error, name)
            if value:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                report.sections.append((f"[DEBUG-ci-resources] child {name}", value))
                _record(
                    "subprocess_output", nodeid=item.nodeid, stream=name, text=value
                )


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """Retain xdist's crash notification even if make later suppresses a failure."""
    _record("worker_down", worker=node.gateway.id, error=str(error) if error else None)


def pytest_sessionfinish(exitstatus):
    """Record pytest exit status independently of Makefile failure handling."""
    _record("session_finish", exitstatus=int(exitstatus))
