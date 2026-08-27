"""Generate every derived dependency list from ``dependencies.yaml``.

The manifest is the single source of truth.  This script writes:

* ``environment.yml``                      -- the developer conda environment
* ``pyproject.toml`` ``[project]`` deps    -- the pip metadata
* ``recipe/meta.yaml`` requirement blocks  -- the conda-forge feedstock

``dependencies-validated.yaml`` belongs to a later stage: it is derived from
the lockfiles, which are themselves solved from the generated
``environment.yml``.  Regenerating it therefore happens in ``make conda-lock``,
right after the lockfiles it reads.  It records the hash of the
``environment.yml`` it was derived alongside, so that a manifest edit which has
not been re-locked is reported rather than silently producing pins for an
environment that no longer exists.

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
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "dependencies.yaml"
ENVIRONMENT_YML = REPO_ROOT / "environment.yml"
VALIDATED_YAML = REPO_ROOT / "dependencies-validated.yaml"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
LOCK_DIR = REPO_ROOT / ".github" / "conda-lock"

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
def render_pyproject(data: dict[str, Any], current: str) -> str:
    """Return ``pyproject.toml`` with a regenerated ``dependencies`` array."""
    specs = sorted(e.spec(e.pip).replace(" ", "") for e in required(data) if e.pip)
    block = "dependencies = [\n"
    block += "".join(f'    "{spec}",\n' for spec in specs)
    block += "]"
    pattern = re.compile(r"^dependencies = \[.*?^\]", re.MULTILINE | re.DOTALL)
    if not pattern.search(current):
        raise ValueError("no [project] dependencies array found in pyproject.toml")
    return pattern.sub(lambda _: block, current, count=1)


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
        "--check", action="store_true", help="verify the derived files are current"
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

    ok = emit(ENVIRONMENT_YML, render_environment(data), args.check)
    ok &= emit(
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
        ok &= emit(path, render_recipe(data, recipe, version), args.check)
    if not ok:
        print("\nRun `make deps-sync` and commit the result.", file=sys.stderr)
    return 0 if ok and pins_are_current() else 1


if __name__ == "__main__":
    sys.exit(main())
