"""Static public-path comparison, baseline selection, and compatibility policy tests."""

from collections import Counter
from pathlib import Path
from typing import cast

import griffe

from tools.api_changes import (
    Report,
    compare,
    latest_support_tag,
    previous_release_tag,
    should_fail,
)


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
