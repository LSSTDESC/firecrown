"""Tests for release-specific behavior in the dependency synchronization tool."""

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(".github/scripts/sync_deps.py")


def load_script() -> ModuleType:
    """Load the dependency synchronization script as a module."""
    spec = importlib.util.spec_from_file_location("sync_deps", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_version_uses_exact_release_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The feedstock version comes from the exact tag at HEAD."""
    module = load_script()
    completed = subprocess.CompletedProcess([], 0, stdout="v1.16.0\n", stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)

    assert module.source_version() == "1.16.0"


def test_source_version_rejects_nonrelease_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only final release tags are accepted for feedstock synchronization."""
    module = load_script()
    completed = subprocess.CompletedProcess([], 0, stdout="preview\n", stderr="")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(ValueError, match="not a final release tag"):
        module.source_version()


def test_source_version_requires_tag_at_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """An untagged checkout cannot provide release feedstock data."""
    module = load_script()

    def fail(*args, **kwargs) -> None:
        raise subprocess.CalledProcessError(128, ["git"])

    monkeypatch.setattr(module.subprocess, "run", fail)

    with pytest.raises(ValueError, match="exact release tag"):
        module.source_version()


def test_source_version_override_uses_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intentional mismatch override permits an untagged migration tree."""
    module = load_script()

    def fail(*args, **kwargs) -> None:
        raise subprocess.CalledProcessError(128, ["git"])

    monkeypatch.setattr(module.subprocess, "run", fail)
    monkeypatch.setattr(module.metadata, "version", lambda name: "1.16.0")

    assert module.source_version(allow_untagged=True) == "1.16.0"
