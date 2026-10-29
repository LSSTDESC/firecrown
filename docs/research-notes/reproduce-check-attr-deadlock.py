"""Reproduce the check-attr hang without setuptools-scm.

Run from a Git worktree. Exit 1 means the hang was observed; exit 0 means the
Git command completed; exit 2 means the result was inconclusive.
"""

import argparse
import faulthandler
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def run_child(count: int) -> int:
    """Run the same bidirectional Git subprocess used by vcs-versioning."""
    data = "\0".join(f"directory_{i:04d}/" for i in range(count))
    faulthandler.dump_traceback_later(1)
    try:
        result = subprocess.run(
            ["git", "check-attr", "-z", "--stdin", "export-ignore"],
            input=data,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"Git failed ({error.returncode}): {error.stderr}", file=sys.stderr)
        return 2
    finally:
        faulthandler.cancel_dump_traceback_later()
    fields = result.stdout.split("\0")
    if len(fields) != count * 3 + 1 or fields[-1] != "":
        print("Unexpected check-attr output", file=sys.stderr)
        return 2
    print(f"Completed: {count} paths, {len(data.encode())} input bytes")
    return 0


def descendants(parent: int) -> list[tuple[int, str]]:
    """List current descendants, parents before children."""
    rows = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,command="], text=True, timeout=5
    )
    processes = []
    for row in rows.splitlines():
        pid, ppid, command = row.strip().split(None, 2)
        processes.append((int(pid), int(ppid), command))
    found = []
    parents = {parent}
    while True:
        children = [
            (pid, command) for pid, ppid, command in processes if ppid in parents
        ]
        if not children:
            return found
        found.extend(children)
        parents = {pid for pid, _ in children}


def run_supervised(count: int, timeout: float) -> int:
    """Use a separate process as a watchdog for the child and its Git process."""
    env = os.environ.copy()
    env["GIT_FLUSH"] = "1"
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(count)],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )
        children: list[tuple[int, str]] = []
        try:
            try:
                status = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                children = descendants(process.pid)
                output.seek(0)
                trace = output.read()
                git_waiting = any(
                    "check-attr -z --stdin export-ignore" in command
                    for _, command in children
                )
                python_waiting = "_communicate" in trace
                if git_waiting and python_waiting:
                    print(
                        f"Hang observed after {timeout:g}s: Git check-attr is "
                        "running and Python is in subprocess._communicate"
                    )
                    return 1
                print(
                    "Timeout, but expected process and stack not observed",
                    file=sys.stderr,
                )
                print(trace, file=sys.stderr)
                return 2
            output.seek(0)
            print(output.read().strip())
            return 0 if status == 0 else 2
        finally:
            for pid, _ in reversed(children):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> int:
    """Parse command-line options and run the test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=65)
    parser.add_argument("--watchdog", type=float, default=3)
    parser.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        return run_child(args.child)
    if args.count < 1 or args.watchdog <= 2:
        parser.error("--count must be positive and --watchdog must exceed 2 seconds")
    return run_supervised(args.count, args.watchdog)


if __name__ == "__main__":
    raise SystemExit(main())
