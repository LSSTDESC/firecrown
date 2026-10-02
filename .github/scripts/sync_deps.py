"""Generate every derived dependency list from ``dependencies.yaml``.

The manifest is the single source of truth.  This script writes:

* ``environment.yml``                      -- the developer conda environment
* ``pyproject.toml`` ``[project]`` deps and Python requirement -- pip metadata
* ``recipe/meta.yaml`` requirement blocks  -- the conda-forge feedstock

``dependencies-validated.yaml`` belongs to a later stage: it is derived from
the lockfiles, which are themselves solved from the generated
``environment.yml``.  Regenerating it therefore happens in ``make conda-lock``,
right after the lockfiles it reads.  It records the hash of the
``environment.yml`` it was derived alongside as historical metadata.  The
ordinary read-only ``--check`` path validates direct lock compatibility rather
than digest currency; compatible older selections remain valid.  Write mode
retains the legacy pin-digest warning.  Validated-constraint reproducibility is
handled separately by ``--pins --check``.

The feedstock blocks are delimited by ``BEGIN GENERATED``/``END GENERATED``
marker comments; everything outside them is left untouched.

Usage::

    python .github/scripts/sync_deps.py                 # write local files
    python .github/scripts/sync_deps.py --check         # verify, do not write
    python .github/scripts/sync_deps.py --feedstock DIR # also write the recipe

Inside a conda build, ``--check-installed`` compares the metapackage that was
just built against the manifest it claims to have been generated from, so that
a recipe left behind by a version bump fails the build rather than shipping.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "dependencies.yaml"
ENVIRONMENT_YML = REPO_ROOT / "environment.yml"
VALIDATED_YAML = REPO_ROOT / "dependencies-validated.yaml"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
LOCK_DIR = REPO_ROOT / ".github" / "conda-lock"
SUPPORTED_PYTHONS = ("3.12", "3.13", "3.14")
SUPPORTED_PLATFORMS = ("linux-64", "osx-arm64")

GENERATED_BY = "make deps-sync"
GENERATED_PINS_BY = "make conda-lock"
GROUPS = ("runtime", "workarounds", "devenv")
# The groups that make up firecrown-deps and the pip metadata.
REQUIRED_GROUPS = ("runtime", "workarounds")
MARKER_RE = re.compile(
    r"^(?P<indent>\s*)#\s*(?P<kind>BEGIN|END) GENERATED (?P<block>[\w.-]+)\b"
)
RECIPE_VERSION_RE = re.compile(r"""\{%\s*set version\s*=\s*["'](?P<version>[^"']+)""")
# The conda index records `run:` as `depends` and `run_constrained:` as
# `constrains`, so each metapackage is checked against a different field.
INSTALLED_FIELD = {
    "firecrown-deps": "depends",
    "firecrown-deps-validated": "constrains",
}


class Entry:
    """One dependency, as declared in the manifest."""

    def __init__(self, raw: dict[str, Any], group: str) -> None:
        self.name: str = raw["name"]
        self.group = group
        self.version: str | None = raw.get("version")
        self.note: str | None = raw.get("note")
        self.conda: str | None = self._resolve(raw.get("conda", self.name))
        # A workaround is, by definition, something firecrown does not import,
        # and its constraint is one conda enforces; an entry opts into the pip
        # metadata by naming the PyPI package.
        imported = group != "workarounds"
        self.module: str | None = self._resolve(
            raw.get("import", self.name if imported else False)
        )
        self.pip: str | None = self._resolve(
            raw.get("pip", self.name if imported else False)
        )

    def _resolve(self, value: Any) -> str | None:
        """Resolve a name override: false disables, true keeps ``name``."""
        if value is False:
            return None
        return self.name if value is True else str(value)

    def spec(self, name: str) -> str:
        """Return ``name`` with the manifest constraint appended."""
        return f"{name} {self.version}" if self.version else name


def source_version() -> str:
    """Return the version of the firecrown source tree this script lives in."""
    try:
        return metadata.version("firecrown")
    except metadata.PackageNotFoundError:
        return "unknown"


def recipe_version(recipe: str) -> str:
    """Return the version a conda recipe builds."""
    match = RECIPE_VERSION_RE.search(recipe)
    if not match:
        raise ValueError("no `{% set version %}` found in the recipe")
    return match.group("version")


def load_manifest() -> dict[str, Any]:
    """Read and lightly validate the manifest."""
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    seen: dict[str, str] = {}
    for group in GROUPS:
        for raw in data[group]:
            name = raw["name"]
            if name in seen:
                raise ValueError(f"{name} is in both {seen[name]} and {group}")
            seen[name] = group
    pip_only = [
        raw["name"]
        for group in GROUPS
        for raw in data[group]
        if raw.get("conda") is False and raw.get("pip") is not False
    ]
    if pip_only and "pip" not in seen:
        raise ValueError(
            "conda needs an explicit pip entry to install the pip section: "
            + ", ".join(sorted(pip_only))
        )
    return data


def entries(data: dict[str, Any], *groups: str) -> list[Entry]:
    """Return the manifest entries of ``groups``, sorted by name."""
    found = (Entry(raw, group) for group in groups for raw in data[group])
    return sorted(found, key=lambda e: e.name)


def required(data: dict[str, Any]) -> list[Entry]:
    """Return everything an installation of firecrown must satisfy."""
    return entries(data, *REQUIRED_GROUPS)


def python_entry(data: dict[str, Any]) -> Entry:
    """Return the interpreter itself as a manifest entry."""
    return Entry({"name": "python", "version": data["python"]}, "runtime")


# --------------------------------------------------------------------------
# environment.yml
# --------------------------------------------------------------------------
def render_environment(data: dict[str, Any]) -> str:
    """Render the developer environment file."""
    all_entries = entries(data, *GROUPS)
    all_entries.append(python_entry(data))
    conda_entries = sorted(
        (e for e in all_entries if e.conda), key=lambda e: e.conda or ""
    )
    pip_entries = sorted(
        (e for e in all_entries if not e.conda and e.pip), key=lambda e: e.pip or ""
    )

    lines = [
        f"# Generated from dependencies.yaml by `{GENERATED_BY}` -- do not edit.",
        "channels:",
        "  - conda-forge",
        "dependencies:",
    ]
    for entry in conda_entries:
        assert entry.conda is not None
        line = f"  - {entry.spec(entry.conda)}"
        if entry.note:
            line += f" # {entry.note}"
        lines.append(line)
        if entry.conda == "pip":
            lines.append("  - pip:")
            lines.extend(f"      - {e.spec(e.pip or '')}" for e in pip_entries)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# pyproject.toml
# --------------------------------------------------------------------------
def render_pyproject(data: dict[str, object], current: str) -> str:
    """Regenerate declarations within the project table, preserving other tables.

    :param data: Authoritative dependency manifest.
    :param current: Existing pyproject.toml contents.
    :returns: Metadata with project dependencies and Python requirement updated.
    """
    project = re.search(r"^[ \t]*\[project\][ \t]*(?:#[^\n]*)?$", current, re.M)
    if project is None:
        raise ValueError("no [project] table found in pyproject.toml")
    start = project.end()
    next_table = re.search(r"^[ \t]*\[", current[start:], re.M)
    end = start + next_table.start() if next_table else len(current)
    content = current[start:end]
    specs = sorted(e.spec(e.pip).replace(" ", "") for e in required(data) if e.pip)
    block = "dependencies = [\n"
    block += "".join(f'    "{spec}",\n' for spec in specs)
    block += "]"
    dependency_pattern = re.compile(
        r"^dependencies = \[.*?^\]", re.MULTILINE | re.DOTALL
    )
    if not dependency_pattern.search(content):
        raise ValueError("no [project] dependencies array found in pyproject.toml")
    content = dependency_pattern.sub(lambda _: block, content, count=1)
    python_pattern = re.compile(
        r'^[ \t]*requires-python[ \t]*=[ \t]*["\'][^"\']*["\']'
        r"(?P<suffix>[ \t]*(?:#[^\n]*)?)$",
        re.M,
    )
    if not python_pattern.search(content):
        raise ValueError(
            "no [project] requires-python declaration found in pyproject.toml"
        )
    python_line = f"requires-python = {json.dumps(data['python'])}"
    content = python_pattern.sub(
        lambda match: python_line + match.group("suffix"), content, count=1
    )
    return current[:start] + content + current[end:]


# --------------------------------------------------------------------------
# recipe/meta.yaml
# --------------------------------------------------------------------------
def locked_versions() -> dict[str, list[str]]:
    """Collect the conda versions recorded in the committed lockfiles."""
    found: dict[str, set[str]] = {}
    lockfiles = sorted(LOCK_DIR.glob("py3.*.conda-lock.yml"))
    if not lockfiles:
        raise FileNotFoundError(f"no lockfiles in {LOCK_DIR}")
    for lockfile in lockfiles:
        lock = yaml.safe_load(lockfile.read_text(encoding="utf-8"))
        for package in lock["package"]:
            if package["manager"] != "conda":
                continue
            found.setdefault(package["name"], set()).add(package["version"])
    return {name: sorted(versions, key=version_key) for name, versions in found.items()}


def pip_identity(name: str) -> str:
    """Normalize a PyPI distribution name as lock records may spell it.

    :param name: Manifest or locked distribution name.
    :returns: Canonical name for presence comparison.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def conda_version_matches(checks: list[tuple[str, str]]) -> list[bool | str]:
    """Evaluate version constraints with conda's own library, without solving.

    The developer environment need not install conda itself. Under ``conda run``
    or activation, ``CONDA_PYTHON_EXE`` identifies conda's tooling interpreter.
    Otherwise the invoking interpreter must provide conda.

    :param checks: Constraint and locked version pairs.
    :returns: Match verdicts or invalid-version diagnostics in input order.
    """
    if not checks:
        return []
    program = """
import json
import sys
from conda.exceptions import InvalidVersionSpec
from conda.models.version import VersionOrder, VersionSpec

results = []
for constraint, version in json.load(sys.stdin):
    try:
        VersionOrder(version)
        results.append(bool(VersionSpec(constraint).match(version)))
    except (InvalidVersionSpec, ValueError) as error:
        results.append(str(error))
json.dump(results, sys.stdout)
"""
    result = subprocess.run(
        [os.environ.get("CONDA_PYTHON_EXE", sys.executable), "-c", program],
        input=json.dumps(checks),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(
            "conda version evaluation failed; use an interpreter providing conda "
            f"or invoke through conda run:\n{result.stderr.strip()}"
        )
    verdicts = json.loads(result.stdout)
    if not isinstance(verdicts, list) or len(verdicts) != len(checks):
        raise RuntimeError("conda version evaluation returned invalid results")
    if not all(isinstance(verdict, (bool, str)) for verdict in verdicts):
        raise RuntimeError("conda version evaluation returned invalid verdicts")
    return verdicts


def pip_version_matches(constraint: str, version: str) -> bool:
    """Check a committed pip selection against PEP 440 requirement bounds.

    Prereleases already selected by conda-lock are eligible, as with installed
    versions; this is compatibility checking rather than candidate selection.
    PEP 440's exclusive bounds and local-version rules still apply.

    :param constraint: Manifest version constraint, or empty for no bounds.
    :param version: Committed pip distribution version.
    :returns: Whether the selection satisfies the requirement.
    """
    return SpecifierSet(constraint).contains(Version(version), prereleases=True)


def check_locked_compatibility(data: dict[str, object]) -> bool:
    """Check direct package selections in every supported locked environment.

    :param data: Authoritative dependency manifest.
    :returns: Whether all six selections satisfy their routed requirements.
    """
    requirements = [python_entry(data), *entries(data, *GROUPS)]
    complete = True
    conda_checks: list[tuple[str, str]] = []
    conda_diagnostics: list[str] = []
    for python in SUPPORTED_PYTHONS:
        lockfile = LOCK_DIR / f"py{python}.conda-lock.yml"
        if not lockfile.is_file():
            print(f"Python {python}: missing lockfile {lockfile}", file=sys.stderr)
            complete = False
            continue
        lock = yaml.safe_load(lockfile.read_text(encoding="utf-8"))
        platforms = set(lock.get("metadata", {}).get("platforms", []))
        for platform in SUPPORTED_PLATFORMS:
            selection = f"Python {python} / {platform}"
            if platform not in platforms:
                print(f"{selection}: missing supported environment", file=sys.stderr)
                complete = False
                continue
            present = {
                (package["manager"], package["name"]): package["version"]
                for package in lock["package"]
                if package["platform"] == platform
            }
            conda = {
                name: version
                for (manager, name), version in present.items()
                if manager == "conda"
            }
            pip = {
                pip_identity(name): version
                for (manager, name), version in present.items()
                if manager == "pip"
            }
            for entry in requirements:
                if entry.conda:
                    found = entry.conda in conda
                    version = conda.get(entry.conda)
                    route = f"conda:{entry.conda}"
                elif entry.pip:
                    found = pip_identity(entry.pip) in pip
                    version = pip.get(pip_identity(entry.pip))
                    route = f"pip:{entry.pip}"
                else:
                    continue
                if not found:
                    print(
                        f"{selection}: missing {entry.group} requirement "
                        f"{entry.name} ({route})",
                        file=sys.stderr,
                    )
                    complete = False
                    continue
                assert version is not None
                diagnostic = (
                    f"{selection}: incompatible {entry.group} requirement "
                    f"{entry.name} ({route}): locked {version}, "
                    f"requires {entry.version or '*'}"
                )
                if entry.conda:
                    conda_checks.append((entry.version or "*", version))
                    conda_diagnostics.append(diagnostic)
                    if entry.conda == "python":
                        conda_checks.append((f"{python}.*", version))
                        conda_diagnostics.append(
                            f"{selection}: incompatible runtime requirement "
                            f"python (conda:python): locked {version}, "
                            f"requires supported Python {python}.*"
                        )
                else:
                    try:
                        matches = pip_version_matches(entry.version or "", version)
                    except ValueError as error:
                        print(f"{diagnostic} ({error})", file=sys.stderr)
                        complete = False
                        continue
                    if not matches:
                        print(diagnostic, file=sys.stderr)
                        complete = False
    try:
        verdicts = conda_version_matches(conda_checks)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Lock compatibility could not be checked: {error}", file=sys.stderr)
        return False
    for diagnostic, verdict in zip(conda_diagnostics, verdicts, strict=True):
        if verdict is not True:
            detail = f" ({verdict})" if isinstance(verdict, str) else ""
            print(diagnostic + detail, file=sys.stderr)
            complete = False
    if complete:
        print(
            "Locked package presence and lock compatibility pass "
            "for all six supported environments."
        )
    return complete


def version_key(version: str) -> tuple[int, ...]:
    """Return a sortable key for a conda version string."""
    return tuple(int(part) for part in re.findall(r"\d+", version)) or (0,)


def validated_constraint(versions: Iterable[str]) -> str:
    """Return a conservative constraint covering the validated versions.

    The lower bound is the oldest version any supported python/platform
    combination was validated against; the upper bound excludes the next
    minor release after the newest one.
    """
    ordered = sorted(versions, key=version_key)
    lowest, highest = ordered[0], ordered[-1]
    parts = version_key(highest)
    major, minor = (list(parts) + [0, 0])[:2]
    return f">={lowest},<{major}.{minor + 1}"


def deps_items(data: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Return the (spec, note) pairs that make up firecrown-deps."""
    items = [python_entry(data)] + required(data)
    return [(e.spec(e.conda), e.note) for e in items if e.conda]


def validated_specs(data: dict[str, Any]) -> list[str]:
    """Return the constraints recorded in ``dependencies-validated.yaml``."""
    if not VALIDATED_YAML.exists():
        raise FileNotFoundError(f"{VALIDATED_YAML} is missing; run `make deps-sync`")
    pins = yaml.safe_load(VALIDATED_YAML.read_text(encoding="utf-8"))["constraints"]
    return [f"{name} {constraint}" for name, constraint in sorted(pins.items())]


def environment_digest() -> str:
    """Return the digest of the environment the lockfiles are solved from.

    Taken over the parsed specs rather than the file, so that editing a note
    does not claim the lockfiles are stale.
    """
    parsed = yaml.safe_load(ENVIRONMENT_YML.read_text(encoding="utf-8"))
    canonical = yaml.safe_dump(parsed, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def pins_are_current() -> bool:
    """Report whether the pins were derived from today's environment.yml."""
    if not VALIDATED_YAML.exists():
        print(f"{VALIDATED_YAML} is missing", file=sys.stderr)
        return False
    recorded = yaml.safe_load(VALIDATED_YAML.read_text(encoding="utf-8"))
    if recorded.get("environment-sha256") == environment_digest():
        return True
    print(
        f"{VALIDATED_YAML.name} was derived from a different environment.yml,"
        "\nso the lockfiles it comes from predate the current manifest."
        "\nRun `make conda-lock` to re-solve and regenerate the pins.",
        file=sys.stderr,
    )
    return False


def render_validated(data: dict[str, Any]) -> str:
    """Render the derived pins, so that they are reviewable and shippable."""
    locked = locked_versions()
    lines = [
        f"# Generated from the conda lockfiles by `{GENERATED_PINS_BY}`"
        " -- do not edit.",
        "#",
        "# The versions every supported python and platform was resolved to, as",
        "# conda constraints.  These become the run_constrained section of the",
        "# firecrown-deps-validated metapackage.",
        "#",
        "# The digest records which environment.yml the lockfiles were solved",
        "# from, so that pins left behind by a manifest edit are detected.",
        f'environment-sha256: "{environment_digest()}"',
        "constraints:",
    ]
    missing = []
    for entry in required(data):
        if not entry.conda:
            continue
        versions = locked.get(entry.conda)
        if not versions:
            missing.append(entry.conda)
            continue
        lines.append(f'  {entry.conda}: "{validated_constraint(versions)}"')
    for name in missing:
        lines.append(f"  # {name}: absent from the lockfiles")
    return "\n".join(lines) + "\n"


def render_block(block: str, data: dict[str, Any], indent: str) -> list[str]:
    """Render the generated lines of one recipe block."""
    if block == "firecrown-deps":
        return [
            f"{indent}- {spec}" + (f"  # {note}" if note else "")
            for spec, note in deps_items(data)
        ]
    if block == "firecrown-deps-validated":
        return [f"{indent}- {spec}" for spec in validated_specs(data)]
    if block == "firecrown-deps-imports":
        modules = sorted(e.module for e in required(data) if e.conda and e.module)
        return [f"""{indent}- python -c "import {', '.join(modules)}\""""]
    if block == "firecrown-devenv":
        return [
            f"{indent}- {e.spec(e.conda)}" for e in entries(data, "devenv") if e.conda
        ]
    raise ValueError(f"unknown generated block: {block}")


def render_recipe(data: dict[str, Any], current: str, version: str) -> str:
    """Return ``meta.yaml`` with every marked block regenerated.

    Each block records the firecrown version it was generated from, so that a
    recipe carrying another release's dependencies is visible in review.
    """
    lines = current.splitlines()
    out: list[str] = []
    index = 0
    seen: set[str] = set()
    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1
        match = MARKER_RE.match(line)
        if not match or match.group("kind") != "BEGIN":
            continue
        block = match.group("block")
        seen.add(block)
        indent = match.group("indent")
        out[-1] = f"{indent}# BEGIN GENERATED {block} (firecrown {version})"
        end = next(
            (
                candidate
                for candidate in range(index, len(lines))
                if (m := MARKER_RE.match(lines[candidate]))
                and m.group("kind") == "END"
                and m.group("block") == block
            ),
            None,
        )
        if end is None:
            raise ValueError(f"unterminated generated block: {block}")
        out.extend(render_block(block, data, indent))
        index = end
    if not seen:
        raise ValueError("no generated blocks found in the recipe")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def check_installed(data: dict[str, Any], name: str) -> bool:
    """Compare an installed metapackage against the manifest that built it."""
    if name not in INSTALLED_FIELD:
        raise ValueError(
            f"nothing known about {name}; expected one of "
            + ", ".join(sorted(INSTALLED_FIELD))
        )
    prefix = Path(os.environ.get("PREFIX", sys.prefix))
    records = [
        record
        for path in sorted((prefix / "conda-meta").glob(f"{name}-*.json"))
        if (record := json.loads(path.read_text(encoding="utf-8")))["name"] == name
    ]
    if not records:
        print(f"{name} is not installed in {prefix}", file=sys.stderr)
        return False

    field = INSTALLED_FIELD[name]
    if name == "firecrown-deps":
        expected = [spec for spec, _ in deps_items(data)]
    else:
        expected = validated_specs(data)
    wanted = normalize(expected)
    found = normalize(records[0].get(field, []))
    if wanted == found:
        print(f"{name} {field} matches the manifest ({len(wanted)} entries)")
        return True

    print(f"{name} {field} does not match the manifest:", file=sys.stderr)
    for spec in sorted(set(wanted) - set(found)):
        print(f"  missing from the package: {spec}", file=sys.stderr)
    for spec in sorted(set(found) - set(wanted)):
        print(f"  not in the manifest:      {spec}", file=sys.stderr)
    print(
        "\nThe recipe was generated from a different version of the manifest."
        "\nRegenerate it with `make feedstock-sync`.",
        file=sys.stderr,
    )
    return False


def normalize(specs: Iterable[str]) -> list[str]:
    """Return specs with their whitespace flattened, in a stable order."""
    return sorted(" ".join(spec.split()) for spec in specs)


def emit(path: Path, content: str, check: bool) -> bool:
    """Write ``content`` to ``path``, or report whether it is up to date."""
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return True
    if not check:
        path.write_text(content, encoding="utf-8")
        print(f"updated {path}")
        return True
    print(f"{path} is out of date:", file=sys.stderr)
    diff = difflib.unified_diff(
        current.splitlines(True),
        content.splitlines(True),
        fromfile=str(path),
        tofile="generated",
    )
    sys.stderr.writelines(diff)
    return False


def main(argv: list[str] | None = None) -> int:
    """Regenerate, or check, every derived dependency list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated declarations and direct lock compatibility without writing",
    )
    parser.add_argument(
        "--feedstock", type=Path, help="path to a firecrown-feedstock checkout"
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="write the recipe even though it builds another version; for"
        " introducing the generated blocks to a recipe, not for routine use",
    )
    parser.add_argument(
        "--pins",
        action="store_true",
        help="regenerate dependencies-validated.yaml from the lockfiles; run"
        " by `make conda-lock`, after the lockfiles have been re-solved",
    )
    parser.add_argument(
        "--check-installed",
        metavar="PACKAGE",
        help="verify an installed metapackage against the manifest",
    )
    args = parser.parse_args(argv)

    data = load_manifest()
    if args.check_installed:
        return 0 if check_installed(data, args.check_installed) else 1

    if args.pins:
        return 0 if emit(VALIDATED_YAML, render_validated(data), args.check) else 1

    targets_ok = emit(ENVIRONMENT_YML, render_environment(data), args.check)
    targets_ok &= emit(
        PYPROJECT_TOML,
        render_pyproject(data, PYPROJECT_TOML.read_text(encoding="utf-8")),
        args.check,
    )
    if args.feedstock:
        path = args.feedstock / "recipe" / "meta.yaml"
        recipe = path.read_text(encoding="utf-8")
        version, builds = source_version(), recipe_version(recipe)
        if version != builds and not args.allow_version_mismatch:
            print(
                f"This tree is firecrown {version}, but the recipe builds"
                f" {builds}.\nSyncing would put this tree's dependencies on a"
                " different release. Check out the matching tag, or pass"
                " --allow-version-mismatch if that is what you mean.",
                file=sys.stderr,
            )
            return 1
        targets_ok &= emit(path, render_recipe(data, recipe, version), args.check)
    if not targets_ok:
        print("\nRun `make deps-sync` and commit the result.", file=sys.stderr)
    if args.check:
        targets_ok &= check_locked_compatibility(data)
        return 0 if targets_ok else 1
    return 0 if targets_ok and pins_are_current() else 1


if __name__ == "__main__":
    sys.exit(main())
