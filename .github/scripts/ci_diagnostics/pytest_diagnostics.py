"""Record per-process pytest diagnostics for ``diagnose-ci.sh``.

The wrapper enables this plugin by adding its directory to ``PYTHONPATH`` and
its module name to ``PYTEST_PLUGINS``. When ``FIRECROWN_DIAGNOSTICS_DIR`` is
set, each pytest process appends newline-delimited JSON records to
``pytest-<pid>.jsonl`` in that directory. Every record has a Unix timestamp,
process ID, event name, and event-specific fields. Separate files and
immediate flushing preserve useful evidence from xdist workers and abrupt
failures.

Events identify session starts and xdist workers, test start times, each test
report phase and outcome, pytest session exit status, and xdist worker-down
notifications. If a test raises ``subprocess.CalledProcessError``, the plugin
also records its return code and any captured stdout/stderr, and appends that
output to the pytest failure report. Command arguments and environment
variables are not recorded. Captured subprocess output can still contain test
data, so diagnostic artifacts should be handled accordingly.

The plugin does not change test outcomes; it adds diagnostic records and
failure-report sections only. Without ``FIRECROWN_DIAGNOSTICS_DIR``, recording
is a no-op.
"""

import json
import os
from pathlib import Path
import subprocess
import time

import pytest


def _record(event, **fields):
    """Append one timestamped JSON event to this process's diagnostics file."""
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
    """Record session startup and identify the xdist worker or controller."""
    _record("session_start", worker=os.environ.get("PYTEST_XDIST_WORKER", "controller"))


def pytest_runtest_logstart(nodeid):
    """Record when a test starts, before terminal output may be buffered."""
    _record("test_start", nodeid=nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record test phase results and expose captured failed-child output."""
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
    """Record xdist worker shutdown or crash details when provided."""
    _record("worker_down", worker=node.gateway.id, error=str(error) if error else None)


def pytest_sessionfinish(exitstatus):
    """Record pytest's exit status independently of Makefile handling."""
    _record("session_finish", exitstatus=int(exitstatus))
