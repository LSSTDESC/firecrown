# Does setuptools-scm 10.3.4 / vcs-versioning 2.6.0 deadlock in `git check-attr` on macOS?

**Date of research:** 2026-10-28
**Trigger:** commit `fae5b72` ("fix: avoid hang during worktree environment setup") pins
`setuptools-scm==9.2.2` in `pyproject.toml`, with the commit message asserting that
"setuptools-scm 10.3.4 / vcs-versioning 2.6.0 deadlock in git check-attr on macOS."

## Diagnosis

The trigger is the `vcs-versioning` 2.6.0 file finder's
[`_export_ignored_dirs`](https://github.com/pypa/setuptools-scm/blob/fa5c2319fef46bea5e387e1329245e7eaffa6f1e/vcs-versioning/src/vcs_versioning/_file_finders/_git.py#L109-L130):
it sends a batch of directory names to `git check-attr --stdin` while capturing
Git's output in pipes. This new I/O pattern exposes a pipe-readiness problem on
the machine tested. The older 9.2.2 finder instead streams `git archive` output
and does not send this bidirectional request.

On this machine, the causal chain was observed as follows:

1. Git reads directory names and, by default, flushes attribute output after each
   path. See Git's [`check_attr_stdin_paths`](https://github.com/git/git/blob/v2.56.0/builtin/check-attr.c)
   and [`maybe_flush_or_die`](https://github.com/git/git/blob/v2.56.0/write-or-die.c).
2. With both pipes active, CPython's
   [`Popen._communicate`](https://github.com/python/cpython/blob/v3.14.9/Lib/subprocess.py)
   polls them and performs a **blocking** write of at most `PIPE_BUF` bytes when
   stdin is reported writable. A trace showed stdin and Git's stdout both ready,
   followed by a write to stdin that did not return. Git was simultaneously
   blocked flushing stdout. Because Python is stuck inside `os.write`, it does
   not reach its next timeout check or drain the ready stdout pipe.
3. This machine reports a pipe writable even when a write cannot proceed. In a
   separate native C experiment, a nonblocking pipe contained 512 unread bytes;
   `poll` returned `POLLOUT`, but a further **one-byte** write returned
   `EAGAIN` (errno 35). `select` and `kqueue` also reported writable; kqueue
   reported 15,872 bytes available. This result needs neither Git nor Python.
   The test's `PIPE_BUF` was 512: that is an atomic-write bound, not a general
   statement that all macOS pipes have 512-byte capacity.

The nearest available public XNU source provides a **plausible explanation**,
not proof of the running kernel's internal state. In
[`xnu-12377.121.6/bsd/kern/sys_pipe.c`](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.121.6/bsd/kern/sys_pipe.c),
pipe creation starts with a 512-byte buffer; expansion is conditional on
`amountpipekva < maxpipekva`; but readiness is calculated using
`MAX(PIPE_SIZE, pipe_buffer.size)`, with `PIPE_SIZE` defined as 16,384 in
[`pipe.h`](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.121.6/bsd/sys/pipe.h).
If the kernel's pipe-memory threshold prevents expansion, this mismatch could
explain the false writable signal and kqueue's 15,872-byte figure. The running
kernel is `xnu-12377.161.15.700.19`, not the source tag examined, and its
`kern.ipc.pipekva` and `kern.ipc.maxpipekva` counters were unavailable. Neither
the threshold condition nor why it might be reached has been verified.

### Package-independent upstream reproduction

[reproduce-check-attr-deadlock.py](./reproduce-check-attr-deadlock.py) needs
Python, Git, and a Git worktree, but no setuptools-scm or vcs-versioning. It
generates nonexistent directory names, sends the same NUL-delimited input shape
to `git check-attr`, and uses a separate watchdog to terminate the test process
and its Git descendant. Run from a worktree:

```bash
python3 docs/research-notes/reproduce-check-attr-deadlock.py --count 64
python3 docs/research-notes/reproduce-check-attr-deadlock.py --count 65
```

In this environment, 64 paths (1,023 input bytes) completed with exit 0.
65 paths (1,039 bytes) consistently produced the `check-attr`/`_communicate`
hang after three seconds, exit 1. Exit 2 means an error or inconclusive result.
The same red result occurred three times in a disposable clone with **no checked
out or indexed files** and global/system Git configuration disabled. The count
is a reproduction threshold for this input and machine, not a portable limit.

### Discriminating controls

- The original finder loop completed with 9.2.2 and hung three times with
  10.3.4/2.6.0. The direct Python/Git reproduction also hung three times.
- Apple Git 2.54.0 and Homebrew Git 2.56.0 both reproduced; so did system Python
  3.9.6, conda Python 3.14.9, and Homebrew Python 3.15.2. Appending a final
  NUL and replacing Python's poll selector with `select` did not prevent it.
- Writing in 64-byte chunks avoided the hang for the tested input. A diagnostic
  monkeypatch that instead supplied `check-attr`'s stdin from a temporary file
  made the **actual newer finder** return all 455 files in three runs; the
  unmodified finder still hung. This is a proof of concept, not a tested
  production patch: the diagnostic changed subprocess handling and did not
  validate all environment, encoding, and error semantics.
- `GIT_FLUSH=0` made the 68-path case complete, but 500 paths still hung, so it
  is not a generally reliable workaround.

The current pin remains in place. This evidence supports an upstream report
about the packaging call pattern and a separate report about pipe readiness; it
does not establish which upstream project should implement a fix, whether the
condition occurs on other machines, or a safe replacement for the pin.

## Methodology and evidence types

This note mixes two distinct kinds of evidence, kept clearly separated below:

- **§1 Upstream primary-source evidence** — PyPI release JSON, exact tagged source files
  and commit SHAs, and issue/PR search results from `github.com/pypa/setuptools-scm`,
  gathered via the GitHub API and PyPI's JSON API.
- **§2 Local empirical evidence** — commands actually executed on this investigator's
  macOS machine during this same investigation effort (file-finder entry-point calls,
  `faulthandler`/`sample` traces, a `PIPE_BUF`-size experiment, and a real
  `pip install --editable` run), presented as directly observed data, not hearsay.
  The comparisons were made in this session on one machine; they have not been
  independently verified on another machine.

Every upstream claim below is cited with a clickable, version-pinned (not `HEAD`) URL.

## 1. Upstream primary-source findings (pypa/setuptools-scm, PyPI)

### Release facts (CONFIRMED, with exact citations)
- `setuptools-scm` 10.3.4 exists on PyPI, uploaded 2026-09-23T19:21:41Z /
  19:21:42Z. Source: [`https://pypi.org/pypi/setuptools-scm/10.3.4/json`](https://pypi.org/pypi/setuptools-scm/10.3.4/json).
  Its `requires_dist` includes `vcs-versioning<3,>=2.5.0.dev0` — i.e. **the published
  version constraint *permits*, but does not pin, `vcs-versioning` 2.6.0**; which exact
  `vcs-versioning` version a given install resolves to depends on the resolver and on
  what else is already installed/pinned. The experiments explicitly requested 2.6.0;
  they do not establish which version an unconstrained install would select.
- `vcs-versioning` 2.6.0 exists on PyPI, uploaded 2026-10-06T22:21:53Z. Source:
  [`https://pypi.org/pypi/vcs-versioning/2.6.0/json`](https://pypi.org/pypi/vcs-versioning/2.6.0/json).
- `vcs-versioning` is not a separate org/project — `project_urls` in that same JSON point
  to `https://github.com/pypa/setuptools-scm`, and a `vcs-versioning/` directory with its
  own `pyproject.toml` lives inside that repository, confirming it is built from a
  workspace/monorepo layout rather than an independent codebase. (`vcs-versioning`'s own
  `pyproject.toml` at the 2.6.0 tag was not re-fetched in this pass; this claim rests on
  the `project_urls` field of the PyPI JSON above plus the source-tree layout confirmed
  in §1's file citations below.)
- `vcs-versioning` CHANGELOG.md at the exact release tag (commit
  `12fc415544d81f55003bfa2b17bd03dc6b6d94c5`):
  [`https://github.com/pypa/setuptools-scm/blob/12fc415544d81f55003bfa2b17bd03dc6b6d94c5/vcs-versioning/CHANGELOG.md`](https://github.com/pypa/setuptools-scm/blob/12fc415544d81f55003bfa2b17bd03dc6b6d94c5/vcs-versioning/CHANGELOG.md).
  The 2.6.0 entry (2026-10-06) only adds a `towncrier-fragments-zerover` version scheme
  (issue [#1556](https://github.com/pypa/setuptools-scm/issues/1556)); the 2.5.0 entry
  (2026-09-23) adds `discover_file_workdir()`. **Neither entry mentions `check-attr`,
  subprocess handling, deadlocks, hangs, or macOS.**
- `setuptools-scm` 9.2.2 exists, uploaded 2025-10-19T22:08:04Z. Source:
  [`https://pypi.org/pypi/setuptools-scm/9.2.2/json`](https://pypi.org/pypi/setuptools-scm/9.2.2/json).
  Its `requires_dist` has **no `vcs-versioning` dependency**.

### Source-code comparison (CONFIRMED mechanism; the two code paths are structurally different, not equivalent)

**New code (10.3.4-era), at the exact `vcs-versioning-v2.6.0` tag** (commit
`fa5c2319fef46bea5e387e1329245e7eaffa6f1e`):

- [`vcs-versioning/src/vcs_versioning/_file_finders/_git.py`](https://github.com/pypa/setuptools-scm/blob/fa5c2319fef46bea5e387e1329245e7eaffa6f1e/vcs-versioning/src/vcs_versioning/_file_finders/_git.py#L109-L124) —
  `_export_ignored_dirs()` calls:
  ```python
  res = run_git(
      ["check-attr", "-z", "--stdin", "export-ignore"],
      repo,
      timeout=timeout,
      input="\0".join(f"{name}/" for name in sorted(dirs)),
  )
  ```
- [`vcs-versioning/src/vcs_versioning/_run_cmd.py`](https://github.com/pypa/setuptools-scm/blob/fa5c2319fef46bea5e387e1329245e7eaffa6f1e/vcs-versioning/src/vcs_versioning/_run_cmd.py#L148-L175) —
  `run()` wraps this in:
  ```python
  res = subprocess.run(
      cmd, check=False, capture_output=True, input=input, cwd=os.fspath(cwd),
      env=dict(avoid_pip_isolation(no_git_env(os.environ)), LC_ALL="C.UTF-8", ...),
      text=True, encoding="utf-8", errors="surrogateescape", timeout=timeout,
  )
  ```
  This is `subprocess.run(capture_output=True, input=..., timeout=...)`, which
  internally calls `Popen.communicate(input=..., timeout=...)` — a **bidirectional**
  call that both writes to the child's stdin pipe and reads both of its output pipes.

**Old code (pre-split), at the exact `v9.2.2` tag** (commit
`e56b78fcd4c1e79aa36c1e4700fca34c8ff540fe`):

- [`src/setuptools_scm/_file_finders/git.py`](https://github.com/pypa/setuptools-scm/blob/e56b78fcd4c1e79aa36c1e4700fca34c8ff540fe/src/setuptools_scm/_file_finders/git.py#L59-L88) —
  file discovery uses `git archive`, invoked via a raw `subprocess.Popen` with only
  `stdout=subprocess.PIPE` (no stdin pipe, nothing written to the child):
  ```python
  cmd = ["git", "archive", "--prefix", toplevel + os.path.sep, "HEAD"]
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=toplevel, stderr=subprocess.DEVNULL)
  ...
  return _git_interpret_archive(proc.stdout, toplevel)  # streamed incrementally via tarfile
  ```
  then explicitly closes stdout, `terminate()`s, and `wait(timeout=5)`s with a `kill()`
  fallback.
- [`src/setuptools_scm/_run_cmd.py`](https://github.com/pypa/setuptools-scm/blob/e56b78fcd4c1e79aa36c1e4700fca34c8ff540fe/src/setuptools_scm/_run_cmd.py#L135-L163) —
  the generic `run()` wrapper used for other git calls (`rev-parse`, etc.) has **no
  `input` parameter at all** at this tag, so nothing in 9.2.2 writes to a child's stdin
  through this wrapper.

**Conclusion — the two file finders are not structurally equivalent, correcting an
earlier draft of this note:** 9.2.2's `git archive` path is output-only (child writes a
tar stream to one pipe; the parent never writes to the child's stdin) and reads that
stream incrementally via `tarfile`, not via `communicate()`. The 10.3.4/vcs-versioning
2.6.0 `check-attr` path is bidirectional — it writes ~1.5 KB to the child's stdin *and*
captures its stdout/stderr, through `subprocess.run(capture_output=True, input=...)`,
i.e. through `Popen.communicate()`. These are genuinely different I/O shapes, and the
new one is the shape that classic Unix pipe-buffer deadlocks are about (full-duplex
pipes where neither side is draining the other). It would be incorrect to say both
versions use the same I/O pattern. The new path uses `communicate()`, rather than a
simple sequential write-then-read loop; the exact cause of the observed hang remains
undetermined.

**On `communicate()` and deadlocks — official Python documentation, used carefully:**
the Python docs state, in the `Popen.wait()` entry (not `communicate()` itself):

> "This will deadlock when using `stdout=PIPE` or `stderr=PIPE` and the child process
> generates enough output to a pipe such that it blocks waiting for the OS pipe buffer
> to accept more data. Use `Popen.communicate()` when using pipes to avoid that."
> — [docs.python.org/3/library/subprocess.html#subprocess.Popen.wait](https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait)

and describes `communicate()` itself as designed to "Send data to stdin. Read data from
stdout and stderr, until end-of-file is reached" concurrently — see
[docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate](https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate).
**This is the official, documented purpose of `communicate()`: to avoid this
class of pipe-buffer deadlock.** The diagnosis above identifies the observed
exception here: the operating system reports a full pipe writable, and the
subsequent blocking write prevents `communicate()` from draining Git's stdout
or checking its timeout. The kernel-state explanation remains a hypothesis.

### Issue/PR search (searches performed, and their exact results — not claimed exhaustive)

GitHub search API queries run against `pypa/setuptools-scm` on 2026-10-28 (clickable
search links reproduce these queries):

- [`is:issue "check-attr"`](https://github.com/pypa/setuptools-scm/issues?q=is%3Aissue+%22check-attr%22) → 0 results found by this query.
- [`is:issue "deadlock"`](https://github.com/pypa/setuptools-scm/issues?q=is%3Aissue+%22deadlock%22) → 0 results found by this query.
- `macOS hang` → 1 result: issue
  [#399 "Hangs on Windows"](https://github.com/pypa/setuptools-scm/issues/399) (created
  2020-01-20, closed 2020-01-22). This is a **real, historical pipe-related hang** in
  setuptools-scm's git file finder — but it is in the *old* `git archive` + raw-`Popen`
  code path (the same code family 9.2.2 still uses), it occurred on **Windows**, not
  macOS, and the root cause described is `proc.wait()` blocking because the child's
  stdout pipe was full of unread bytes — not a `check-attr` call, and not a
  bidirectional stdin+stdout deadlock. The fix applied upstream (visible in the 9.2.2
  code cited above: `proc.stdout.close()` before `terminate()`/`wait(timeout=5)`)
  appears to already be in place by the 9.2.2 tag.
- `hang OR stuck OR frozen (title)` → also surfaced issue
  [#888 "Docker build stuck on install phase"](https://github.com/pypa/setuptools-scm/issues/888)
  (created 2023-08-24) — a Linux/Docker context, not macOS, and predating the
  `check-attr` code (introduced in PR #1498, 2026-08-13) by about three years.
- PR search for `check-attr` → the only substantive match is
  [PR #1498, "fix: honour export-ignore for directories and submodules in git file
  discovery"](https://github.com/pypa/setuptools-scm/pull/1498) (merged 2026-08-13 by
  `RonnyPfannschmidt`), which **introduced** the `check-attr -z --stdin` call (replacing
  an interim `git ls-files --recurse-submodules` approach from PR #1405). Its own
  description is about an sdist-size regression (vendored submodules not excluded from
  export-ignore'd directories), with a reproduction recipe and before/after file
  listings — it does not mention a hang, deadlock, timeout, or macOS anywhere in its
  body.

**Negative finding, explicitly bounded:** none of the specific searches performed above
turned up an issue or PR in `pypa/setuptools-scm` that names a deadlock/hang tied to
`check-attr`, to `vcs-versioning` 2.6.0, or to macOS. This is **not** a claim that no
such report exists anywhere upstream — GitHub's search API is keyword-based and was
queried with a limited set of terms; a report using different wording (e.g. "never
finishes", "pip install hangs forever" without the words tried here) could have been
missed. No attempt was made to page through the full issue/PR history of the repository.

## 2. Local empirical evidence (this investigator's own macOS machine)

The following comparisons were run on the investigator's macOS machine in this
session. The numbers are directly observed, not independently cross-checked on another
machine.

**Environment:** macOS 26.7.1 (build 25G241), Darwin 25.6.0, arm64; Python 3.14.9;
Git 2.56.0; pip 26.2.2; setuptools 84.2.1; wheel 0.48.0; packaging 26.3. Python's
internal `subprocess._PIPE_BUF` was 512 and `selectors.DefaultSelector` was
`KqueueSelector`; CPython's actual `subprocess._PopenSelector` was
`PollSelector`.

**Reproduction setup:** from the Firecrown worktree, create disposable environments
and a local clone. `uv` uses the selected Python interpreter; the original comparison
used the versions listed above. The explicit `vcs-versioning` pin forces the
combination being investigated.

```bash
test_root=$(mktemp -d "${TMPDIR:-/tmp}/firecrown-scm-repro.XXXXXX")
uv venv --python "$(command -v python3)" "$test_root/old"
uv venv --python "$(command -v python3)" "$test_root/new"
uv pip install --system-certs --python "$test_root/old/bin/python" \
  'setuptools-scm==9.2.2' 'setuptools==84.2.1' 'wheel==0.48.0' 'pip==26.2.2'
uv pip install --system-certs --python "$test_root/new/bin/python" \
  'setuptools-scm==10.3.4' 'vcs-versioning==2.6.0' \
  'setuptools==84.2.1' 'wheel==0.48.0' 'pip==26.2.2'
git clone --shared --no-hardlinks --no-checkout . "$test_root/repo"
git -C "$test_root/repo" switch --detach fae5b72
```

**Warning:** the new-version tests can hang indefinitely even though the library sets
an internal timeout. Run them under an external watchdog that terminates the test
process and its descendants, or be prepared to terminate the identified Python and Git
processes. Do not rely on the library timeout.

**Minimal reproduction** — invoking the `setuptools.file_finders` entry point directly,
in each isolated environment against the same linked Firecrown worktree:

```bash
"$test_root/old/bin/python" -c "import importlib.metadata as m; print(len(next(iter(m.entry_points(group='setuptools.file_finders'))).load()('')))"
"$test_root/new/bin/python" -c "import importlib.metadata as m; print(len(next(iter(m.entry_points(group='setuptools.file_finders'))).load()('')))"
```

- `setuptools-scm==9.2.2`: returned 455 files in 0.366 s.
- `setuptools-scm==10.3.4` + `vcs-versioning==2.6.0`: observed blocked for >120 s,
  despite the code's own internal default `timeout=40` (seconds) not taking effect in
  this run. The later diagnosis above explains why the timeout check did not run.

**Stack samples:** Python `faulthandler`/macOS `sample` output showed the Python main
thread blocked in `os_write` at CPython `subprocess.py:2164`, in the call chain
`vcs_versioning/_run_cmd.py:159` → `_export_ignored_dirs` in `_git.py:123` (matching the
`check-attr` call cited in §1 above); concurrently, the child `git` process was blocked
in `fflush`/`__write_nocancel`. Both sides observed blocked on writing is *consistent
with* (not a certain, isolated proof of) a pipe-buffer backpressure deadlock: the
exact `git check-attr -z --stdin export-ignore` request in this worktree carries 68
tracked ancestor directories (~1503 input bytes).

**Control experiment (isolates, does not alone prove, the pipe as the mechanism):**
redirecting the child's stdout to a regular temporary file instead of a pipe completed
successfully — 3272 output bytes, exit 0. This is a useful control that rules out the
`check-attr` *command itself* hanging regardless of I/O plumbing, and is consistent with
the hang being specific to the piped-I/O path — but on its own it does not pin down the
exact kernel- or CPython-internal mechanism, since other differences between the two
configurations (buffering mode, read granularity) were not separately isolated.

**Further diagnostic control (explicitly not a proposed fix):** monkeypatching the
private `subprocess._PIPE_BUF` constant before invoking the file finder: size 1
completed in 0.113 s, size 64 completed in 0.097 s, while the platform default of 512
hung (observed >8 s before being killed). This is private/undocumented CPython internal
state, not a public API, and the experiment is reported as a diagnostic signal (smaller
internal chunk sizes avoid the hang) rather than a verified root-cause explanation or a
production-safe workaround.

**Full install reproduction**, at commit `fae5b72`:

```bash
"$test_root/old/bin/python" -m pip install --no-deps --no-build-isolation --editable "$test_root/repo"
"$test_root/new/bin/python" -m pip install --no-deps --no-build-isolation --editable "$test_root/repo"
```

Note: `--no-build-isolation` deliberately bypasses `pyproject.toml`'s
`[build-system].requires` pin so that the pre-installed `setuptools-scm`/`vcs-versioning`
versions in each environment are used instead of the pinned `9.2.2` — this is
intentional for the A/B comparison, but it means this specific command does **not**
exercise `pyproject.toml`'s pin as configured, nor the `conda run`/`tools/create-worktree-environment`
lifecycle used in normal setup.

- `setuptools-scm==9.2.2` environment: succeeded (exit 0) in 2.63 s.
- `setuptools-scm==10.3.4` + `vcs-versioning==2.6.0` environment: still blocked at the
  20 s mark when observed, with the process tree `pip` → `pip`'s `_in_process.py
  prepare_metadata_for_build_editable` → `git --git-dir <clone>/.git check-attr -z
  --stdin export-ignore`. The test environments and clone were torn down afterward.

**Explicit scope limits of this local evidence, as already noted during the
investigation:** this is a reduced file-finder / pip-editable-install reproduction, not
a run of the full Worktrunk hook or the `conda run` lifecycle in
`tools/create-worktree-environment`; it was observed on one specific macOS machine/build
combination; and it has not been shown to reproduce on other macOS versions, other
Python builds, or non-macOS platforms.

## Overall assessment

| Claim element | Status |
|---|---|
| setuptools-scm 10.3.4 and vcs-versioning 2.6.0 both exist on PyPI | **CONFIRMED** — [10.3.4 JSON](https://pypi.org/pypi/setuptools-scm/10.3.4/json), [2.6.0 JSON](https://pypi.org/pypi/vcs-versioning/2.6.0/json) |
| 10.3.4's dependency constraint *permits* vcs-versioning 2.6.0 | **CONFIRMED** (constraint `<3,>=2.5.0.dev0`); whether a given install *actually resolves* to exactly 2.6.0 was not independently verified via a live `pip` resolution in this pass |
| vcs-versioning is a package from the same `pypa/setuptools-scm` monorepo | **CONFIRMED** via PyPI `project_urls` and in-repo directory layout |
| setuptools-scm 9.2.2 does not depend on vcs-versioning and uses `git archive` for file discovery | **CONFIRMED** — [9.2.2 JSON](https://pypi.org/pypi/setuptools-scm/9.2.2/json); [v9.2.2 `git.py`](https://github.com/pypa/setuptools-scm/blob/e56b78fcd4c1e79aa36c1e4700fca34c8ff540fe/src/setuptools_scm/_file_finders/git.py) |
| The new `check-attr` call is structurally different (bidirectional stdin+stdout pipe via `communicate()`) from the old `git archive` call (output-only pipe, manual close/terminate/wait) | **CONFIRMED** by direct source comparison, §1 |
| Upstream has a filed, confirmed issue/PR describing a check-attr deadlock on macOS | **NOT FOUND** by the specific searches performed (not claimed exhaustive) |
| setuptools-scm's git file finder has prior reports of hangs | **CONFIRMED**, including Windows ([#399](https://github.com/pypa/setuptools-scm/issues/399), 2020) and Linux/Docker ([#888](https://github.com/pypa/setuptools-scm/issues/888), 2023) — not macOS or `check-attr` |
| A hang reproducibly occurs on this investigator's macOS machine when invoking the new check-attr-based file finder, including through a real `pip install --editable` | **Directly observed** in this investigation (§2); not yet independently re-verified in a second session, and not yet shown on a second machine |
| The packaging call triggers a pipe-readiness failure below the package layer | **REPRODUCED** locally without either package, with a matching native pipe test; the precise kernel state is not verified |

**Summary:** the version/packaging facts in the commit message are confirmed, and the
9.2.2 pin does sidestep the newer, structurally different (bidirectional-pipe)
`check-attr` code path entirely. Upstream `pypa/setuptools-scm` has no issue or PR,
found by the searches run here, that documents this specific deadlock on macOS. At the
same time, the diagnostics above establish a concrete trigger and pipe-readiness
failure on this machine. The exact kernel state is unverified and the issue has
not been shown to affect macOS installations generally. The pin remains a
locally supported workaround, not an upstream-confirmed fix.

## Search limitations (explicitly non-exhaustive)

- GitHub issue/PR search used a small number of keyword queries against the search API;
  it does not cover the repository's full issue/PR history, nor wording variants not
  tried here.
- The official Python `subprocess` documentation was read for the `Popen.wait()` and
  `Popen.communicate()` entries only; this is not a claim that no other part of the
  documentation (or the CPython bug tracker, which was not searched) bears on this
  question.
- The test environments explicitly installed `vcs-versioning==2.6.0`; they did not
  verify the version selected by a fresh install without that constraint.
- The local evidence in §2 reflects one investigation session on one physical machine
  (macOS 26.7.1 / Darwin 25.6.0, arm64) and has not been cross-checked on a second
  machine, a second macOS version, or a non-macOS platform.
- No byte-for-byte diff of every file in `vcs-versioning`/`setuptools_scm` between 9.2.2
  and 10.3.4 was performed; only the specific file-finder and run-command modules
  directly relevant to `check-attr` were compared, at exact tag-pinned commits.
- The public XNU tag inspected is close to, but does not exactly match, the running
  kernel. The relevant pipe-memory counters were unavailable, so the proposed kernel
  mechanism remains unconfirmed.
