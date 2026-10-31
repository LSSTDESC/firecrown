"""Tests for the Makefile's explicitly scoped cleanup targets."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

OUTPUTS = {
    "clean-coverage": (
        ".coverage",
        ".coverage.worker",
        "coverage.xml",
        "coverage.json",
        "coverage_other.json",
        "coverage_third.json",
        "htmlcov/index.html",
        "htmlcov_other/index.html",
        "htmlcov_third/index.html",
        "covdata/combined",
    ),
    "clean-docs": (
        "docs/_build/index.html",
        "docs/_static/index.html",
        "docs/autoapi/index.html",
        "docs/warnings.txt",
        "tutorial/.quarto/cache",
        "tutorial/_site/index.html",
    ),
    "clean-build": (
        "build/generated",
        "dist/package.tar.gz",
        "firecrown.egg-info/PKG-INFO",
    ),
    "clean-cache": (
        ".pytest_cache/cache",
        ".mypy_cache/cache",
        ".ruff_cache/cache",
        ".hypothesis/cache",
        "__pycache__/module.pyc",
        "firecrown/module/__pycache__/module.pyc",
        "tests/__pycache__/module.pyc",
        "tools/__pycache__/module.pyc",
        "pylint_plugins/__pycache__/module.pyc",
        "docs/__pycache__/module.pyc",
        "tutorial/__pycache__/module.pyc",
        ".github/scripts/__pycache__/module.pyc",
    ),
}

PRESERVED = (
    ".conda-env/bin/python",
    ".conda-env/lib/__pycache__/module.pyc",
    ".venv/bin/python",
    "env/bin/python",
    ".vscode/settings.json",
    "data.csv",
    "notes.txt",
    "other.egg-info/PKG-INFO",
    "tests/.hidden/__pycache__/module.pyc",
    "tests/venv/__pycache__/module.pyc",
    "tests/local-env/pyvenv.cfg",
    "tests/local-env/lib/__pycache__/module.pyc",
    "tests/local-conda/conda-meta/history",
    "tests/local-conda/lib/__pycache__/module.pyc",
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create outputs and user files without touching the actual checkout."""
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "Makefile", tmp_path / "Makefile")
    for path in (*PRESERVED, *(path for paths in OUTPUTS.values() for path in paths)):
        file_path = tmp_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("keep or remove as specified", encoding="utf-8")
    return tmp_path


def run_make(
    workspace: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run Make against the disposable workspace."""
    return subprocess.run(
        ["make", f"PYTHON={sys.executable}", *args],
        cwd=workspace,
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("target", [*OUTPUTS, "clean"])
def test_clean_targets_are_scoped_and_repeatable(workspace: Path, target: str) -> None:
    """Remove only the selected outputs, preserving unrelated files."""
    run_make(workspace, target)
    run_make(workspace, target)

    for group, paths in OUTPUTS.items():
        for path in paths:
            assert (workspace / path).exists() is (target not in (group, "clean"))
    for path in PRESERVED:
        assert (workspace / path).read_text(
            encoding="utf-8"
        ) == "keep or remove as specified"


def test_clean_coverage_ignores_configured_id(workspace: Path) -> None:
    """Remove reports for every ID without using the ID as a path."""
    run_make(workspace, "clean-coverage", "COVERAGE_ID=../outside")
    assert all(not (workspace / path).exists() for path in OUTPUTS["clean-coverage"])


def test_clean_cache_does_not_follow_symlinks(workspace: Path) -> None:
    """Do not traverse symlinks to directories outside the source tree."""
    external = workspace / "outside"
    cache = external / "__pycache__/module.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_text("keep", encoding="utf-8")
    (workspace / "tests/link").symlink_to(external, target_is_directory=True)

    run_make(workspace, "clean-cache")
    assert cache.read_text(encoding="utf-8") == "keep"


def test_docs_cleanup_precedes_cache_scan(workspace: Path) -> None:
    """Avoid scanning documentation outputs while they are being removed."""
    result = run_make(workspace, "-n", "-j4", "clean-cache", "clean-docs")
    assert result.stdout.index("rm -f -- docs/warnings.txt") < result.stdout.index(
        "for root in firecrown"
    )
    run_make(workspace, "-j4", "clean-cache", "clean-docs")


def test_clean_does_not_follow_output_symlinks(workspace: Path, tmp_path: Path) -> None:
    """Removing a generated-directory symlink must not remove its target."""
    external = tmp_path / "outside"
    external.mkdir()
    (external / "important.txt").write_text("keep", encoding="utf-8")
    shutil.rmtree(workspace / "docs/_static")
    (workspace / "docs/_static").symlink_to(external, target_is_directory=True)

    run_make(workspace, "clean-docs")
    assert (external / "important.txt").read_text(encoding="utf-8") == "keep"
    assert not (workspace / "docs/_static").is_symlink()


def test_clean_precedes_other_command_line_goals(workspace: Path) -> None:
    """Even in parallel, other requested goals run after cleanup."""
    with (workspace / "Makefile").open("a", encoding="utf-8") as makefile:
        makefile.write(
            "\n.PHONY: verify-clean\n"
            "verify-clean:\n"
            "\ttest ! -e build\n"
            "\ttest -e .conda-env/bin/python\n"
        )
    run_make(workspace, "-j4", "clean", "verify-clean")
