# Temporary CI resource diagnostics

The Ubuntu/Python 3.12 coverage job wraps `make test-ci` with
`bash .github/scripts/diagnose-ci.sh "$RUNNER_TEMP/ci-diagnostics" make test-ci`.
The workflow uploads `ci-diagnostics-<ref>-ubuntu-py3.12-<attempt>` even after test
failure, retaining it for seven days. This is evidence collection for the
September 21, 2026 failure, not a resource-exhaustion fix.

## Evidence

- `runner.log`: kernel, CPU count, resource limits, and cgroup membership.
- `resources.log`: two-second UTC samples of system memory, pressure, process
  PID/parent/group, thread count, RSS, CPU use, and cgroup memory/PID limits and
  event counters (including ancestor cgroups).
- `kernel-before.log`, `kernel-after.log`: kernel OOM/crash messages, or an explicit
  message that access was unavailable. Compare before and after; an old OOM
  message is not evidence of a new OOM kill.
- `command.log`: start/end timestamps and the original command's exit status.
- `tests.log`: make/pytest output including captured child stdout/stderr attached
  to failed subprocess tests.
- `pytest-<pid>.jsonl`: actual test start/report times, worker identity and crash
  notices, session exit status, and failed child output/return codes. Epoch times
  correlate with resource samples without make's output buffering.

Environment variables and process command lines are not dumped. Child output
comes from the generated example tests; review artifacts for secrets before
sharing them outside the repository. Do not run this collector against inputs
containing credentials.

## Interpretation and limitations

A positive subprocess return code of 15 does not establish SIGTERM. A direct
signal termination is reported as a negative return code. The pytest plugin
records the actual return code but does not trace the sender of a signal.
Worker crash notices may likewise lack an exit code. Kernel evidence and cgroup
`oom_kill` counter increases can establish OOM involvement; RSS samples alone
cannot. Two-second sampling can miss short peaks; `memory.peak`, where available,
helps but may predate the test run.

The wrapper preserves make's exit status. For this diagnostic run, the Makefile
restores the `unit-tests-core` and `test-example` recipes from failing commit
`20bf3ae`: Cobaya connector tests participate in parallel coverage, and Cobaya
example failures fail the target. Make's existing parallel scheduling allows
coverage and example targets to overlap. Other changes since that commit remain,
so this is not a complete replay of that revision.

Run the workflow normally on a branch containing these changes. A runner loss or
hard cancellation may prevent final collection/upload; no artifact can be
guaranteed in that case. Linux `/proc`, cgroup v2, GNU ps, and passwordless sudo
for kernel logs are expected on the Ubuntu runner; unavailable probes are logged.
The collector adds sampling and per-test file writes, so timing can be affected.

## Local validation and removal

The plugin was exercised with a passing test, a child exiting 15 with captured
stdout/stderr, and an xdist worker exiting abruptly. Wrapper status propagation
was checked with a command exiting 15. These smoke checks verify the collector,
not reproduction of the Firecrown failure. Linux probes require a CI run.

After diagnosis, restore the workflow's direct `make test-ci` invocation, remove
its diagnostic step ID/environment/artifact-name output and upload step, and
delete `diagnose-ci.sh` and this `ci_diagnostics` directory. Temporary terminal
messages use `[DEBUG-ci-resources]`.
