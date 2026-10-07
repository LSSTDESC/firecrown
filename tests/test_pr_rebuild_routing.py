"""Test production PR rebuild canary routing and admission decisions."""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/route_pr_rebuild_canary.py"
)


def _git(repository: Path, *args: str) -> str:
    """Run Git in a temporary repository and return its standard output.

    :param repository: Temporary Git checkout.
    :param args: Arguments passed to Git after ``git``.
    :returns: Standard output from the successful command.
    """
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


@pytest.fixture
def pull_request(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a local base/head history for production diff routing.

    :param tmp_path: Pytest temporary directory.
    :returns: Repository path, base commit SHA, and head commit SHA.
    """
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Routing test")
    (repository / "README.md").write_text("base\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "--quiet", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")
    return repository, base, base


def _commit_change(repository: Path, path: str) -> str:
    """Add a path to the PR head and return the resulting commit SHA.

    :param repository: Temporary Git checkout.
    :param path: Repository-relative path to add.
    :returns: Commit SHA containing the new path.
    """
    changed = repository / path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("change\n", encoding="utf-8")
    _git(repository, "add", path)
    _git(repository, "commit", "--quiet", "-m", f"change {path}")
    return _git(repository, "rev-parse", "HEAD")


def _run_decision(
    repository: Path,
    base: str,
    head: str,
    consistency_result: str,
    tmp_path: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str], str]:
    """Invoke the production routing CLI and collect its Actions outputs.

    :param repository: Temporary Git checkout containing the PR commits.
    :param base: Base commit SHA.
    :param head: Head commit SHA.
    :param consistency_result: Controlled conclusion of the consistency gate.
    :param tmp_path: Pytest temporary directory for output artifacts.
    :returns: Process result, parsed scalar outputs, and written summary.
    """
    output_path = tmp_path / "github-output"
    summary_path = tmp_path / "summary.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository",
            str(repository),
            "--base",
            base,
            "--head",
            head,
            "--consistency-result",
            consistency_result,
            "--github-output",
            str(output_path),
            "--summary",
            str(summary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    outputs = dict(
        line.split("=", maxsplit=1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )
    return result, outputs, summary_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "path",
    [
        "dependencies.yaml",
        "dependencies-validated.yaml",
        "environment.yml",
        "pyproject.toml",
        "requirements-ci.txt",
        "Makefile",
        "pre-commit-check",
        "check-docs",
        "docs/Makefile",
        ".github/scripts/generate_conda_locks.sh",
        ".github/conda-lock/py3.12.conda-lock.yml",
        ".github/workflows/ci-reusable.yml",
        "tools/check_dependencies.py",
    ],
)
def test_relevant_pull_request_changes_are_admitted(
    pull_request: tuple[Path, str, str], path: str, tmp_path: Path
) -> None:
    """Relevant dependency, build, and CI edits admit the production canary.

    :param pull_request: Temporary base/head repository.
    :param path: Representative dependency-impacting path.
    :param tmp_path: Pytest temporary directory.
    """
    repository, base, _ = pull_request
    head = _commit_change(repository, path)

    result, outputs, summary = _run_decision(
        repository, base, head, "success", tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert outputs["should_run"] == "true"
    assert outputs["outcome"] == "run"
    assert outputs["revision"] == head
    assert outputs["platform"] == "Linux"
    assert outputs["python_version"] == "3.12"
    assert "Decision:** run" in summary


def test_irrelevant_change_is_an_explained_skip(
    pull_request: tuple[Path, str, str], tmp_path: Path
) -> None:
    """An unrelated edit skips only rebuilding and reports intended identity.

    :param pull_request: Temporary base/head repository.
    :param tmp_path: Pytest temporary directory.
    """
    repository, base, _ = pull_request
    head = _commit_change(repository, "README.md")

    result, outputs, summary = _run_decision(
        repository, base, head, "success", tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert outputs["should_run"] == "false"
    assert outputs["outcome"] == "skip"
    assert "No dependency, build, or CI configuration changes" in outputs["reason"]
    assert head in summary
    assert "Linux/Python 3.12" in summary


@pytest.mark.parametrize("path", ["dependencies.yaml", "README.md"])
def test_consistency_failure_skips_rebuild_with_explanation(
    pull_request: tuple[Path, str, str], path: str, tmp_path: Path
) -> None:
    """A failed mandatory gate prevents rebuild for relevant or irrelevant edits.

    :param pull_request: Temporary base/head repository.
    :param path: Relevant or irrelevant path changed by the PR.
    :param tmp_path: Pytest temporary directory.
    """
    repository, base, _ = pull_request
    head = _commit_change(repository, path)

    result, outputs, summary = _run_decision(
        repository, base, head, "failure", tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert outputs["should_run"] == "false"
    assert outputs["outcome"] == "skip"
    assert "mandatory dependency consistency job did not pass" in outputs["reason"]
    assert "Mandatory dependency consistency:** failure" in summary
    assert head in summary
    assert "Linux/Python 3.12" in summary


def test_routing_error_fails_even_when_consistency_also_failed(
    pull_request: tuple[Path, str, str], tmp_path: Path
) -> None:
    """A failed diff decision stays visible alongside a failed consistency gate.

    :param pull_request: Temporary base/head repository.
    :param tmp_path: Pytest temporary directory.
    """
    repository, base, _ = pull_request
    missing_head = "0" * 40

    result, outputs, summary = _run_decision(
        repository, base, missing_head, "failure", tmp_path
    )

    assert result.returncode == 1
    assert outputs["should_run"] == "false"
    assert outputs["outcome"] == "routing-error"
    assert "Could not determine changed paths" in outputs["reason"]
    assert "Mandatory dependency consistency:** failure" in summary
    assert missing_head in summary


def test_workflow_wires_production_routing_and_admission() -> None:
    """The workflow runs the tested decision CLI before admitting the canary."""
    workflow_path = SCRIPT.parents[1] / "workflows/ci.yml"
    workflow = yaml.load(
        workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    jobs = workflow["jobs"]
    decision = jobs["pr-canary-decision"]
    decision_step = next(
        step for step in decision["steps"] if step.get("id") == "decision"
    )
    canary = jobs["pr-rebuild-drift"]

    assert "always()" in decision["if"]
    assert "pr-dependency-consistency" in decision["needs"]
    assert "route_pr_rebuild_canary.py" in decision_step["run"]
    assert (
        "needs.pr-dependency-consistency.result"
        in decision_step["env"]["CONSISTENCY_RESULT"]
    )
    assert "pr-canary-decision.outputs.should_run == 'true'" in canary["if"]
    assert "pr-dependency-consistency.result == 'success'" in canary["if"]
    assert "pr-drift-paths" not in jobs
