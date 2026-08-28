"""Tests for the Conda environment comparison utility."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(".github/scripts/compare_conda_environment.py")


def load_script() -> ModuleType:
    """Load the environment comparison script as a module."""
    spec = importlib.util.spec_from_file_location("compare_conda_environment", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_difference_report_ignores_editable_package(capsys) -> None:
    """The editable Firecrown package can be excluded from comparison."""
    module = load_script()
    locked = {
        "conda:python": {
            "manager": "conda",
            "name": "python",
            "version": "3.12.1",
            "url": "https://example/python-3.12.1-h123_0.conda",
        }
    }
    installed = {
        "conda:python": {
            "name": "python",
            "version": "3.12.1",
            "build_string": "h123_0",
            "dist_name": "python-3.12.1-h123_0",
        },
        "pip:firecrown": {
            "name": "firecrown",
            "version": "1.16.0",
            "build_string": "pypi_0",
        },
    }

    assert not module.print_difference_report(
        locked, installed, ignored={"pip:firecrown"}
    )
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "actual_path,version,expected_path,expected_version,expected_valid",
    [
        ("/repo/firecrown", "1.16.0", "/repo/firecrown", "1.16.0", True),
        ("/repo/other", "1.16.0", "/repo/firecrown", "1.16.0", False),
        ("/repo/firecrown", "1.16.1", "/repo/firecrown", "1.16.0", False),
    ],
)
def test_validate_editable_firecrown(
    monkeypatch: pytest.MonkeyPatch,
    actual_path: str,
    version: str,
    expected_path: str,
    expected_version: str,
    expected_valid: bool,
) -> None:
    """Editable validation checks both source path and reported version."""
    module = load_script()
    package: dict[str, Any] = {
        "metadata": {"name": "firecrown", "version": version},
        "direct_url": {
            "url": Path(actual_path).as_uri(),
            "dir_info": {"editable": True},
        },
    }
    monkeypatch.setattr(module, "editable_firecrown", lambda: package)

    valid, _ = module.validate_editable_firecrown(Path(expected_path), expected_version)

    assert valid is expected_valid


def test_validate_editable_firecrown_rejects_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editable validation fails when Firecrown is not installed."""
    module = load_script()
    monkeypatch.setattr(module, "editable_firecrown", lambda: None)

    valid, message = module.validate_editable_firecrown(Path.cwd(), None)

    assert not valid
    assert "not installed as an editable" in message


def test_comparison_allows_conda_firecrown_without_strict_mode() -> None:
    """Ordinary comparison reports a Conda Firecrown package normally."""
    module = load_script()
    installed = {"conda:firecrown": {"name": "firecrown"}}

    status, ignored = module.validate_firecrown_installation(installed, None, None)

    assert status == 0
    assert ignored == set()


def test_strict_mode_rejects_conda_firecrown() -> None:
    """Release validation rejects a Conda Firecrown package."""
    module = load_script()
    installed = {"conda:firecrown": {"name": "firecrown"}}

    status, ignored = module.validate_firecrown_installation(
        installed, Path.cwd(), None
    )

    assert status == module.INVALID_EDITABLE
    assert ignored == set()
