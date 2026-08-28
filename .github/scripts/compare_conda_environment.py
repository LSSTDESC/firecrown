#!/usr/bin/env python3
"""Compare the active Conda environment with its platform lockfile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

NO_ACTIVE_ENVIRONMENT = 1
NO_FIRECROWN = 2
PACKAGE_DIFFERENCES = 3
CONFIGURATION_ERROR = 4
INVALID_EDITABLE = 5


def package_key(package: dict[str, Any]) -> str:
    """Return a normalized package name."""
    return package["name"].lower().replace("_", "-")


def run_conda_list() -> list[dict[str, Any]]:
    """Return package records from the active environment."""
    result = subprocess.run(
        ["conda", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(result.stdout)
    if not isinstance(packages, list):
        raise ValueError("conda list --json did not return a package list")
    return packages


def run_json_command(command: list[str]) -> Any:
    """Run a command and parse its JSON output."""
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def conda_platform() -> str:
    """Return the active Conda installation's target platform."""
    information = run_json_command(["conda", "info", "--json"])
    if not isinstance(information, dict) or not isinstance(
        information.get("platform"), str
    ):
        raise ValueError("conda info --json did not report a platform")
    return information["platform"]


def editable_firecrown() -> dict[str, Any] | None:
    """Return pip's Firecrown record when it is an editable installation."""
    inspection = run_json_command([sys.executable, "-m", "pip", "inspect", "--local"])
    if not isinstance(inspection, dict) or not isinstance(
        inspection.get("installed"), list
    ):
        raise ValueError("pip inspect did not return installed package records")

    for package in inspection["installed"]:
        metadata = package.get("metadata", {})
        if str(metadata.get("name", "")).lower() != "firecrown":
            continue
        direct_url = package.get("direct_url", {})
        if direct_url.get("dir_info", {}).get("editable") is True:
            return package
    return None


def editable_path(package: dict[str, Any]) -> Path:
    """Return the local source path from an editable pip record."""
    url = package["direct_url"]["url"]
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise ValueError(f"editable Firecrown URL is not local: {url}")
    return Path(unquote(parsed.path)).resolve()


def read_lockfile(path: Path, target_platform: str) -> dict[str, dict[str, Any]]:
    """Return the lockfile package records for one platform."""
    with path.open(encoding="utf-8") as lockfile:
        document = yaml.safe_load(lockfile)

    packages: dict[str, dict[str, Any]] = {}
    for package in document.get("package", []):
        if package.get("platform") not in {target_platform, "noarch"}:
            continue
        manager = package.get("manager")
        if manager not in {"conda", "pip"}:
            continue
        key = f"{manager}:{package_key(package)}"
        packages[key] = package
    return packages


def format_conda_record(package: dict[str, Any]) -> str:
    """Format an installed package for a difference report."""
    build = package.get("build_string", "<unknown build>")
    version = package.get("version", "<unknown version>")
    return f"{version} ({build})"


def format_lock_record(package: dict[str, Any]) -> str:
    """Format a locked package for a difference report."""
    if package["manager"] == "conda":
        version = package["version"]
        filename = package["url"].rsplit("/", 1)[-1]
        return f"{version} ({filename})"
    return str(package["version"])


def print_difference_report(
    locked: dict[str, dict[str, Any]],
    installed: dict[str, dict[str, Any]],
    ignored: set[str] | None = None,
) -> bool:
    """Print package differences and return whether any were found."""
    missing, mismatched, extra = environment_differences(locked, installed, ignored)

    if missing:
        print("Missing from environment:")
        for key in missing:
            print(f"  {key}: {format_lock_record(locked[key])}")
    if mismatched:
        print("Different version or build:")
        for key in mismatched:
            print(
                f"  {key}: lockfile {format_lock_record(locked[key])}; "
                f"environment {format_conda_record(installed[key])}"
            )
    if extra:
        print("In environment but not in lockfile:")
        for key in extra:
            print(f"  {key}: {format_conda_record(installed[key])}")
    return bool(missing or mismatched or extra)


def environment_differences(
    locked: dict[str, dict[str, Any]],
    installed: dict[str, dict[str, Any]],
    ignored: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return missing, mismatched, and extra environment package keys."""
    ignored = ignored or set()
    locked_keys = set(locked) - ignored
    installed_keys = set(installed) - ignored
    missing = sorted(locked_keys - installed_keys)
    extra = sorted(installed_keys - locked_keys)
    mismatched = sorted(
        key
        for key in locked_keys & installed_keys
        if (
            locked[key]["manager"] == "conda"
            and (
                locked[key]["version"] != installed[key]["version"]
                or locked[key]["url"]
                .rsplit("/", 1)[-1]
                .removesuffix(".conda")
                .removesuffix(".tar.bz2")
                != installed[key].get("dist_name")
            )
        )
        or (
            locked[key]["manager"] == "pip"
            and locked[key]["version"] != installed[key]["version"]
        )
    )
    return missing, mismatched, extra


def validate_editable_firecrown(
    project: Path, expected_version: str | None
) -> tuple[bool, str]:
    """Validate Firecrown's editable source path and optional version."""
    firecrown = editable_firecrown()
    if firecrown is None:
        return False, "Firecrown is not installed as an editable pip project."

    actual_path = editable_path(firecrown)
    expected_path = project.resolve()
    if actual_path != expected_path:
        return (
            False,
            f"Editable Firecrown points to {actual_path}, not {expected_path}.",
        )

    installed_version = str(firecrown["metadata"].get("version", ""))
    if expected_version and installed_version != expected_version:
        return (
            False,
            "Editable Firecrown reports version "
            f"{installed_version!r}, not {expected_version!r}.",
        )
    return True, f"Editable Firecrown: {actual_path} ({installed_version})"


def installed_records(
    packages: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index installed Conda and pip records by manager and package name."""
    installed: dict[str, dict[str, Any]] = {}
    for package in packages:
        manager = "pip" if package.get("channel") == "pypi" else "conda"
        installed[f"{manager}:{package_key(package)}"] = package
    return installed


def installed_python_series(packages: list[dict[str, Any]]) -> str:
    """Return the active environment's Python major and minor version."""
    try:
        version = next(
            package["version"]
            for package in packages
            if package_key(package) == "python"
        )
    except StopIteration as error:
        raise ValueError("active environment does not contain Python") from error
    return ".".join(version.split(".")[:2])


def check_editable(
    project: Path | None,
    expected_version: str | None,
) -> set[str]:
    """Validate an optional editable install and return ignored packages."""
    if project is None:
        if expected_version:
            raise ValueError("--expected-version requires --editable-project")
        return set()

    valid, message = validate_editable_firecrown(project, expected_version)
    if not valid:
        raise ValueError(message)
    print(message)
    return {"pip:firecrown"}


def inspect_environment(
    lockfile_arg: Path | None,
) -> tuple[list[dict[str, Any]], Path, str, dict[str, dict[str, Any]]]:
    """Inspect the active environment and load its selected lockfile."""
    installed_packages = run_conda_list()
    python_series = installed_python_series(installed_packages)
    lockfile = lockfile_arg or Path(
        f".github/conda-lock/py{python_series}.conda-lock.yml"
    )
    target_platform = conda_platform()
    locked = read_lockfile(lockfile, target_platform)
    return installed_packages, lockfile, target_platform, locked


def validate_firecrown_installation(
    installed: dict[str, dict[str, Any]],
    project: Path | None,
    expected_version: str | None,
) -> tuple[int, set[str]]:
    """Validate Firecrown's installation and return ignored lockfile keys."""
    if project is None:
        if expected_version:
            print("--expected-version requires --editable-project.", file=sys.stderr)
            return INVALID_EDITABLE, set()
        return 0, set()
    if "conda:firecrown" in installed:
        print(
            "Active environment contains a conda Firecrown package; "
            "install Firecrown only as an editable pip project.",
            file=sys.stderr,
        )
        return INVALID_EDITABLE, set()
    try:
        return 0, check_editable(project, expected_version)
    except (
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        print(f"Unable to inspect editable Firecrown: {error}", file=sys.stderr)
        return CONFIGURATION_ERROR, set()
    except (TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return INVALID_EDITABLE, set()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lockfile",
        type=Path,
        help="Lockfile to compare (defaults to the active Python version).",
    )
    parser.add_argument(
        "--editable-project",
        type=Path,
        help="Require Firecrown to be installed editable from this directory.",
    )
    parser.add_argument(
        "--expected-version",
        help="Also require the editable Firecrown version to equal this value.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the environment comparison."""
    args = parse_args()
    if not os.environ.get("CONDA_PREFIX") or not os.environ.get("CONDA_DEFAULT_ENV"):
        print("No active Conda environment.", file=sys.stderr)
        return NO_ACTIVE_ENVIRONMENT

    try:
        installed_packages, lockfile, target_platform, locked = inspect_environment(
            args.lockfile
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        yaml.YAMLError,
    ) as error:
        print(f"Unable to inspect environment or lockfile: {error}", file=sys.stderr)
        return CONFIGURATION_ERROR

    installed = installed_records(installed_packages)
    environment_name = os.environ["CONDA_DEFAULT_ENV"]
    print(f"Environment: {environment_name}")
    print(f"Lockfile: {lockfile} ({target_platform})")
    status, ignored = validate_firecrown_installation(
        installed, args.editable_project, args.expected_version
    )
    if status:
        return status

    if print_difference_report(locked, installed, ignored):
        return PACKAGE_DIFFERENCES
    print("Environment matches the lockfile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
