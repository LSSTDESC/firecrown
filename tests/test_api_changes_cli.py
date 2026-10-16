"""CLI, report-output, and metadata tests using temporary API repositories."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from tools.api_changes import write_metadata


@pytest.fixture
def api_repo(tmp_path: Path) -> Path:
    """Create a Git repository with one committed Firecrown API."""
    package = tmp_path / "firecrown"
    package.mkdir()
    (package / "__init__.py").write_text("def public(): pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "firecrown"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Baseline API",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def run_api(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the public API reporting CLI from a temporary repository."""
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "api_changes.py"),
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def tag_api_baseline(repo: Path, source: str, private: str = "") -> None:
    """Publish a small API fixture as v1.0.0 in an existing repository."""
    package = repo / "firecrown"
    (package / "__init__.py").write_text(source, encoding="utf-8")
    if private:
        (package / "_internal.py").write_text(private, encoding="utf-8")
    subprocess.run(["git", "add", "firecrown"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Release API",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "tag", "v1.0.0"], cwd=repo, check=True)


def test_support_check_includes_uncommitted_public_api_changes(tmp_path: Path) -> None:
    package = tmp_path / "firecrown"
    package.mkdir()
    (package / "__init__.py").write_text(
        "def public(value):\n    return value\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "v1_0_support"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "add", "firecrown"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Release API",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "tag", "v1.0.0"], cwd=tmp_path, check=True)
    (package / "__init__.py").write_text(
        "def public(value, required):\n    return value\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "api_changes.py"),
            "support",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert "firecrown.public" in result.stdout
    assert "required" in result.stdout


def test_support_targeting_pr_writes_report_before_blocking(tmp_path: Path) -> None:
    package = tmp_path / "firecrown"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("def public(value): pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "feature"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "firecrown"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Baseline API",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/v1_0_support", "HEAD"],
        cwd=tmp_path,
        check=True,
    )
    source.write_text("def public(value, required): pass\n", encoding="utf-8")
    output = tmp_path / "reports" / "api.md"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "api_changes.py"),
            "pr",
            "--base",
            "refs/remotes/origin/v1_0_support",
            "--target",
            "v1_0_support",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert result.stdout == ""
    assert "firecrown.public" in output.read_text(encoding="utf-8")
    assert "required" in output.read_text(encoding="utf-8")


def test_non_support_pr_reports_break_without_blocking(tmp_path: Path) -> None:
    package = tmp_path / "firecrown"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("def public(value): pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "feature"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "firecrown"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Baseline API",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "branch", "master"], cwd=tmp_path, check=True)
    source.write_text("def public(value, required): pass\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "api_changes.py"),
            "pr",
            "--base",
            "master",
            "--target",
            "master",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "firecrown.public" in result.stdout
    assert "required" in result.stdout


@pytest.mark.parametrize(
    ("target", "status", "level", "verdict"),
    [
        ("master", 0, "notice", "No breaking API changes detected."),
        (
            "master",
            0,
            "warning",
            "Informational because the target is `master`. "
            "Reviewer assessment required.",
        ),
        (
            "v1_0_support",
            1,
            "error",
            "This fails support-line policy.",
        ),
    ],
)
def test_pr_github_actions_verdict(
    api_repo: Path,
    target: str,
    status: int,
    level: str,
    verdict: str,
) -> None:
    """Publish an explicit policy verdict without changing compatibility policy.

    :param api_repo: Repository containing a committed baseline API.
    :param target: PR target used for the compatibility policy.
    :param status: Expected CLI exit status.
    :param level: Expected GitHub annotation severity.
    :param verdict: Expected policy explanation.
    """
    if level != "notice":
        (api_repo / "firecrown" / "__init__.py").write_text("", encoding="utf-8")
    output = api_repo / "report.md"
    result = run_api(
        api_repo,
        "pr",
        "--base",
        "main",
        "--target",
        target,
        "--output",
        str(output),
        "--github-actions",
    )

    assert result.returncode == status, result.stderr
    assert f"::{level} title=Public API compatibility::" in result.stdout
    assert verdict in result.stdout
    report = output.read_text(encoding="utf-8")
    assert verdict in report
    assert f"Target: `{target}`" in report
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=api_repo, text=True
    ).strip()
    assert f"Revision: `{revision}`" in report
    assert "Static analysis" in report
    if level != "notice":
        assert "Breaking API changes detected" in report
        assert "`firecrown.public`: Public object removed" in report


@pytest.mark.parametrize("publish", [True, False])
def test_pr_workflow_publishes_verdict_and_reports_publication_failure(
    api_repo: Path, publish: bool
) -> None:
    """Exercise the actual workflow shell with a broken API and summary sink.

    :param api_repo: Repository containing a committed baseline API.
    :param publish: Whether the summary destination is writable as a file.
    """
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/api-changes.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["api-changes"]["steps"]
        if step.get("name") == "Report PR API changes"
    )
    command = step["run"].replace(
        "python tools/api_changes.py",
        shlex.join([sys.executable, str(root / "tools/api_changes.py")]),
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/master", "HEAD"],
        cwd=api_repo,
        check=True,
    )
    (api_repo / "firecrown" / "__init__.py").write_text("", encoding="utf-8")
    summary = api_repo / "summary.md" if publish else api_repo
    event = api_repo / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "number": 12,
                    "head": {"sha": "a" * 40},
                    "base": {"sha": "b" * 40, "ref": "master"},
                }
            }
        )
    )
    result = subprocess.run(
        ["bash", "-e", "-c", command],
        cwd=api_repo,
        env=os.environ
        | {
            "BASE_BRANCH": "master",
            "RUNNER_TEMP": str(api_repo),
            "GITHUB_STEP_SUMMARY": str(summary),
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_REPOSITORY": "LSSTDESC/firecrown",
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        "::warning title=Public API compatibility::Breaking API changes detected"
        in (result.stdout)
    )
    assert "`firecrown.public`: Public object removed" in result.stdout
    metadata = json.loads((api_repo / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == 123
    assert metadata["run_attempt"] == 2
    assert metadata["pr_number"] == 12
    assert metadata["verdict"] in result.stdout
    assert f"Baseline: `{metadata['baseline']}`" in result.stdout
    if publish:
        assert result.returncode == 0, result.stderr
        assert "**Breaking API changes detected" in summary.read_text()
    else:
        assert result.returncode == 2
        assert "::error title=Public API reporting failed::" in result.stdout


@pytest.mark.parametrize("base", ["missing", "main"])
def test_pr_metadata_records_analysis_failure(api_repo: Path, base: str) -> None:
    """Preserve failed comparison identity without inventing a policy verdict.

    :param api_repo: Repository containing a committed baseline API.
    :param base: Missing baseline or valid baseline with invalid Python source.
    """
    if base == "main":
        (api_repo / "firecrown/__init__.py").write_text("def invalid(\n")
    event = api_repo / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "number": 12,
                    "head": {"sha": "a" * 40},
                    "base": {"sha": "b" * 40, "ref": "master"},
                }
            }
        )
    )
    metadata = api_repo / "metadata.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools/api_changes.py"),
            "pr",
            "--base",
            base,
            "--metadata",
            str(metadata),
        ],
        cwd=api_repo,
        env=os.environ
        | {
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_REPOSITORY": "LSSTDESC/firecrown",
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    data = json.loads(metadata.read_text())
    assert data["status"] == "failed"
    assert data["verdict"] == data["baseline"] == ""


def test_write_metadata_requires_pr_mode(tmp_path: Path) -> None:
    """Reject metadata output for modes without PR event identity."""
    with pytest.raises(ValueError, match="--metadata requires pr mode"):
        write_metadata("support", str(tmp_path / "metadata.json"))


def test_pr_github_annotation_reports_analysis_failure(api_repo: Path) -> None:
    """Distinguish analysis failure from a completed compatibility verdict.

    :param api_repo: Repository containing a committed baseline API.
    """
    result = run_api(api_repo, "pr", "--base", "missing", "--github-actions")
    assert result.returncode == 2
    assert "::error title=Public API compatibility::API analysis/reporting failed." in (
        result.stdout
    )
    assert "Compatibility was not assessed" in result.stdout
    assert "No breaking API changes detected" not in result.stdout


def test_release_reports_against_latest_eligible_tag_without_blocking(
    tmp_path: Path,
) -> None:
    package = tmp_path / "firecrown"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("def public(value): pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "master"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "firecrown"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Baseline API",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "tag", "v1.0.2"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "v1.0.10"], cwd=tmp_path, check=True)
    source.write_text("def public(value, required): pass\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "api_changes.py"),
            "release",
            "--version",
            "1.0.11",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("# Public API changes: v1.0.10 -> HEAD\n")
    assert "firecrown.public" in result.stdout
    assert "required" in result.stdout


def test_support_check_reports_addition_without_blocking(tmp_path: Path) -> None:
    package = tmp_path / "firecrown"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("def public(): pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "init", "-q", "-b", "v1_0_support"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "add", "firecrown"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Baseline API",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "tag", "v1.0.0"], cwd=tmp_path, check=True)
    source.write_text("def public(): pass\ndef added(): pass\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "api_changes.py"),
            "support",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("# Public API changes: v1.0.0 -> HEAD\n")
    assert "No breaking API changes detected." in result.stdout
    assert "- `firecrown.added`" in result.stdout


def test_support_check_missing_release_tag_is_comparison_error(tmp_path: Path) -> None:
    package = tmp_path / "firecrown"
    package.mkdir()
    (package / "__init__.py").write_text("def public(): pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "init", "-q", "-b", "v1_0_support"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "add", "firecrown"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Baseline API",
        ],
        cwd=tmp_path,
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools" / "api_changes.py"),
            "support",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "No release tag found for v1_0_support" in result.stderr


def test_support_check_rejects_non_support_branch(api_repo: Path) -> None:
    result = run_api(api_repo, "support")

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Not a support branch: main" in result.stderr


@pytest.mark.parametrize(
    ("version", "message"),
    [
        ("1.0", "VERSION must have the form x.y.z"),
        ("1.0.1", "No previous release tag found for 1.0.1"),
    ],
)
def test_release_rejects_invalid_baseline(
    api_repo: Path, version: str, message: str
) -> None:
    result = run_api(api_repo, "release", "--version", version)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


@pytest.mark.parametrize(
    ("mode", "message"),
    [("release", "release requires --version"), ("pr", "pr requires --base")],
)
def test_cli_requires_comparison_arguments(
    api_repo: Path, mode: str, message: str
) -> None:
    result = run_api(api_repo, mode)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_support_check_rejects_tag_outside_branch_history(api_repo: Path) -> None:
    subprocess.run(["git", "tag", "v1.0.0"], cwd=api_repo, check=True)
    subprocess.run(
        ["git", "switch", "-q", "--orphan", "v1_0_support"],
        cwd=api_repo,
        check=True,
    )
    (api_repo / "firecrown").mkdir()
    (api_repo / "firecrown" / "__init__.py").write_text(
        "def public(): pass\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "firecrown"], cwd=api_repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=API test",
            "-c",
            "user.email=api@example.test",
            "commit",
            "-qm",
            "Unrelated support work",
        ],
        cwd=api_repo,
        check=True,
    )

    result = run_api(api_repo, "support")

    assert result.returncode == 2
    assert result.stdout == ""
    assert (
        "Latest release tag v1.0.0 is not on support branch v1_0_support"
        in result.stderr
    )


def test_release_rejects_unresolvable_public_exports(api_repo: Path) -> None:
    (api_repo / "firecrown" / "api.py").write_text(
        "def public(): pass\n", encoding="utf-8"
    )
    tag_api_baseline(api_repo, "from . import api\n")
    (api_repo / "firecrown" / "api.py").write_text(
        "__all__ = [missing]\n", encoding="utf-8"
    )

    result = run_api(api_repo, "release", "--version", "1.1.0")

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Cannot resolve __all__ in firecrown" in result.stderr


def test_release_keeps_unresolved_public_alias_without_aborting(api_repo: Path) -> None:
    (api_repo / "firecrown" / "api.py").write_text(
        "from nonexistent_pkg import Missing as Thing\n", encoding="utf-8"
    )
    tag_api_baseline(api_repo, "from . import api\n")

    result = run_api(api_repo, "release", "--version", "1.1.0")

    assert result.returncode == 0, result.stderr
    assert "No breaking API changes detected." in result.stdout
    assert "`firecrown.api.Thing`:" not in result.stdout


def test_release_excludes_private_child_of_public_module_alias(api_repo: Path) -> None:
    (api_repo / "firecrown" / "api.py").write_text(
        "from firecrown import _internal as Public\n", encoding="utf-8"
    )
    tag_api_baseline(
        api_repo,
        "from . import api\n",
        "__all__ = ['_hidden']\ndef _hidden(value): pass\n",
    )
    (api_repo / "firecrown" / "_internal.py").write_text(
        "__all__ = ['_hidden']\ndef _hidden(value, required): pass\n",
        encoding="utf-8",
    )

    result = run_api(api_repo, "release", "--version", "1.1.0")

    assert result.returncode == 0, result.stderr
    assert "No breaking API changes detected." in result.stdout
    assert "firecrown.api.Public._hidden" not in result.stdout


def test_release_reports_public_object_kind_change(api_repo: Path) -> None:
    tag_api_baseline(api_repo, "def public(value): pass\n")
    (api_repo / "firecrown" / "__init__.py").write_text(
        "public = 1\n", encoding="utf-8"
    )

    result = run_api(api_repo, "release", "--version", "1.1.0")

    assert result.returncode == 0, result.stderr
    assert (
        "`firecrown.public`: Public object points to a different kind of object"
        in result.stdout
    )
    assert "`function` -> `attribute`" in result.stdout
