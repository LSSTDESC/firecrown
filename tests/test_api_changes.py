"""Behavioral tests for the public API change report."""

from collections import Counter
from pathlib import Path
import subprocess
import sys
from typing import cast

import griffe
import pytest

from tools.api_changes import (
    Report,
    compare,
    latest_support_tag,
    previous_release_tag,
    should_fail,
)


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


def load_package(path: Path, source: str, private: str = "") -> griffe.Module:
    """Create and load a temporary package with the given API source."""
    package = path / "example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "api.py").write_text(source, encoding="utf-8")
    (package / "_internal.py").write_text(private, encoding="utf-8")
    return cast(
        griffe.Module,
        griffe.load(
            "example",
            search_paths=[path],
            resolve_aliases=True,
            resolve_implicit=True,
            resolve_external=False,
            allow_inspection=False,
        ),
    )


def test_imported_name_and_export_removal_are_reported(tmp_path: Path) -> None:
    old = load_package(
        tmp_path / "old",
        "from example._internal import helper\n__all__ = ['helper']\n",
        "def helper(x):\n    return x\n",
    )
    new = load_package(
        tmp_path / "new",
        "from example._internal import helper\n__all__ = []\n",
        "def helper(x):\n    return x\n",
    )
    report = compare(old, new)
    assert any("example.api.helper" in item for item in report.breaks)
    assert all("example._internal" not in item for item in report.breaks)


def test_unlisted_import_removal_is_public(tmp_path: Path) -> None:
    old = load_package(
        tmp_path / "old",
        "from example._internal import helper\n",
        "def helper(): pass\n",
    )
    new = load_package(tmp_path / "new", "", "def helper(): pass\n")
    report = compare(old, new)
    assert any("example.api.helper" in item for item in report.breaks)


def test_new_public_paths_exclude_private_paths(tmp_path: Path) -> None:
    old = load_package(tmp_path / "old", "")
    new = load_package(
        tmp_path / "new",
        "from example._internal import helper\n_new = 1\n",
        "def helper(): pass\n",
    )
    report = compare(old, new)
    assert "example.api.helper" in report.additions
    assert not any("._internal" in path or "._new" in path for path in report.additions)


def test_removing_public_module_reports_module_path(tmp_path: Path) -> None:
    old = load_package(tmp_path / "old", "def public(): pass\n")
    new_path = tmp_path / "new" / "example"
    new_path.mkdir(parents=True)
    (new_path / "__init__.py").write_text("", encoding="utf-8")
    new = cast(
        griffe.Module,
        griffe.load("example", search_paths=[tmp_path / "new"], allow_inspection=False),
    )
    report = compare(old, new)
    assert any("example.api" in item for item in report.breaks)


def test_signature_break_on_reexport_is_reported(tmp_path: Path) -> None:
    old = load_package(
        tmp_path / "old",
        "from example._internal import helper\n",
        "def helper(x): pass\n",
    )
    new = load_package(
        tmp_path / "new",
        "from example._internal import helper\n",
        "def helper(x, y): pass\n",
    )
    report = compare(old, new)
    assert any(
        "example.api.helper" in item and "required" in item.lower()
        for item in report.breaks
    )


def test_reexported_class_method_break_survives_target_move(tmp_path: Path) -> None:
    old = load_package(
        tmp_path / "old",
        "from example._internal import Thing\n",
        "class Thing:\n    def run(self, value): pass\n",
    )
    new_path = tmp_path / "new"
    load_package(
        new_path,
        "from example._other import Thing\n",
        "",
    )
    (new_path / "example" / "_other.py").write_text(
        "class Thing:\n    def run(self, value, required): pass\n",
        encoding="utf-8",
    )
    new = cast(
        griffe.Module,
        griffe.load(
            "example",
            search_paths=[new_path],
            resolve_aliases=True,
            resolve_implicit=True,
            resolve_external=False,
            allow_inspection=False,
        ),
    )
    report = compare(old, new)
    assert any(
        "example.api.Thing.run" in item and "required" in item.lower()
        for item in report.breaks
    )


def test_removing_public_class_method_is_reported(tmp_path: Path) -> None:
    old = load_package(tmp_path / "old", "class Thing:\n    def run(self): pass\n")
    new = load_package(tmp_path / "new", "class Thing: pass\n")
    report = compare(old, new)
    assert any("example.api.Thing.run" in item for item in report.breaks)


def test_multiple_removed_parameters_are_distinct(tmp_path: Path) -> None:
    old = load_package(tmp_path / "old", "def call(first, second): pass\n")
    new = load_package(tmp_path / "new", "def call(): pass\n")
    report = compare(old, new)
    assert sum("Parameter was removed" in item for item in report.breaks) == 2
    assert any("first" in item for item in report.breaks)
    assert any("second" in item for item in report.breaks)


def test_new_class_method_is_new_api(tmp_path: Path) -> None:
    old = load_package(tmp_path / "old", "class Thing: pass\n")
    new = load_package(tmp_path / "new", "class Thing:\n    def run(self): pass\n")
    assert "example.api.Thing.run" in compare(old, new).additions


def test_unchanged_exported_class_does_not_list_inherited_methods_as_new(
    tmp_path: Path,
) -> None:
    old = load_package(
        tmp_path / "old",
        "from example._internal import Thing\n",
        "class Thing:\n    def run(self): pass\n",
    )
    new = load_package(
        tmp_path / "new",
        "from example._internal import Thing\n",
        "class Thing:\n    def run(self): pass\n",
    )
    assert not compare(old, new).additions


def test_public_alias_to_private_module_detects_changed_function(
    tmp_path: Path,
) -> None:
    old = load_package(
        tmp_path / "old",
        "from example import _internal as Public\n",
        "def run(value): pass\n",
    )
    new = load_package(
        tmp_path / "new",
        "from example import _internal as Public\n",
        "def run(value, required): pass\n",
    )
    assert any("example.api.Public.run" in item for item in compare(old, new).breaks)


def test_support_tag_uses_numeric_patch_and_matching_line() -> None:
    assert (
        latest_support_tag(
            ["v1.15.2", "v1.15.10", "v1.16.4", "v1.15.0a0"], "v1_15_support"
        )
        == "v1.15.10"
    )


def test_previous_release_tag_skips_prereleases() -> None:
    assert (
        previous_release_tag(["v1.15.2", "v1.16.0", "v1.16.0a0"], "1.16.0") == "v1.15.2"
    )


def test_only_support_line_breaks_fail_ci() -> None:
    report = Report(["breaking"], set(), Counter())
    assert should_fail("pr", "v1_15_support", report)
    assert not should_fail("pr", "master", report)
    assert not should_fail("pr", "staging", report)


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
