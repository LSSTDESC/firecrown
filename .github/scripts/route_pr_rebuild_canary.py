"""Route and admit the pull request rebuild canary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PLATFORM = "Linux"
PYTHON_VERSION = "3.12"
RELEVANT_ROOT_FILES = {
    "Makefile",
    "GNUmakefile",
    "pre-commit-check",
    "check-docs",
    "docs/Makefile",
    "dependencies.yaml",
    "dependencies-validated.yaml",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "tox.ini",
    "noxfile.py",
    "justfile",
}


def _relevant(path: str) -> bool:
    """Return whether a changed path can affect dependency rebuilding.

    :param path: Repository-relative path reported by Git.
    :returns: Whether the change should conservatively trigger the canary.
    """
    normalized = path.replace(os.sep, "/")
    name = normalized.rsplit("/", maxsplit=1)[-1]
    return (
        normalized.startswith((".github/", "tools/", "scripts/"))
        or normalized in RELEVANT_ROOT_FILES
        or (name.startswith("requirements") and name.endswith((".txt", ".in")))
    )


def _changed_paths(repository: Path, base: str, head: str) -> list[str]:
    """Read changed paths from the same base-to-head diff used by the PR.

    :param repository: Checkout containing the PR commits.
    :param base: Base commit SHA from the pull request event.
    :param head: Head commit SHA from the pull request event.
    :returns: Repository-relative changed paths.
    :raises subprocess.CalledProcessError: If Git cannot evaluate the diff.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{base}...{head}",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def _write_outputs(path: Path, values: dict[str, str]) -> None:
    """Append scalar decisions to the GitHub Actions output file.

    :param path: File named by ``GITHUB_OUTPUT``.
    :param values: Output keys and single-line values.
    """
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def _write_summary(
    path: Path,
    *,
    outcome: str,
    reason: str,
    revision: str,
    consistency_result: str,
) -> None:
    """Append a canary routing decision and its intended sample identity.

    :param path: File named by ``GITHUB_STEP_SUMMARY``.
    :param outcome: ``run``, ``skip``, or ``routing-error``.
    :param reason: Human-readable decision reason.
    :param revision: PR head revision for which the decision was made.
    :param consistency_result: Conclusion of the mandatory consistency job.
    """
    with path.open("a", encoding="utf-8") as summary:
        summary.write(
            "## PR rebuild canary admission\n\n"
            f"**Decision:** {outcome}\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Mandatory dependency consistency:** {consistency_result}\n\n"
            f"**Intended sample:** {revision}, {PLATFORM}/Python {PYTHON_VERSION}\n"
        )


def main(argv: list[str] | None = None) -> int:
    """Compute routing and consistency admission for the canary.

    :param argv: Optional command-line arguments; defaults to process arguments.
    :returns: Process status, failing only when routing could not be determined.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="PR base commit SHA")
    parser.add_argument("--head", required=True, help="PR head commit SHA")
    parser.add_argument(
        "--consistency-result",
        required=True,
        help="Conclusion of the mandatory consistency job",
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    identity = {
        "revision": args.head,
        "platform": PLATFORM,
        "python_version": PYTHON_VERSION,
    }
    try:
        relevant_paths = [
            path
            for path in _changed_paths(args.repository, args.base, args.head)
            if _relevant(path)
        ]
    except (OSError, subprocess.SubprocessError) as error:
        detail = str(error).replace("\n", " ").strip()
        reason = "Could not determine changed paths for this pull request."
        if detail:
            reason = f"{reason} Git reported: {detail}"
        outcome = "routing-error"
        should_run = "false"
        exit_code = 1
    else:
        exit_code = 0
        if args.consistency_result != "success":
            outcome = "skip"
            should_run = "false"
            reason = (
                "The mandatory dependency consistency job did not pass; "
                "the fresh rebuild was not admitted."
            )
        elif relevant_paths:
            outcome = "run"
            should_run = "true"
            reason = "Dependency, build, or CI configuration changes were detected."
        else:
            outcome = "skip"
            should_run = "false"
            reason = "No dependency, build, or CI configuration changes were detected."

    outputs = {
        "should_run": should_run,
        "outcome": outcome,
        "reason": reason,
        **identity,
    }
    if args.github_output:
        _write_outputs(args.github_output, outputs)
    if args.summary:
        _write_summary(
            args.summary,
            outcome=outcome,
            reason=reason,
            revision=args.head,
            consistency_result=args.consistency_result,
        )
    if exit_code:
        print(reason, file=sys.stderr)
        return exit_code
    print(f"{outcome}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
