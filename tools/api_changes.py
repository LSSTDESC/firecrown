"""Report changes to Firecrown's public Python API using Griffe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import cast

import griffe

TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
SUPPORT = re.compile(r"^v(\d+)_(\d+)_support$")


def release_tags(tags: list[str]) -> list[tuple[tuple[int, int, int], str]]:
    """Return numerically sorted stable release tags."""
    return sorted(
        ((int(match[1]), int(match[2]), int(match[3])), tag)
        for tag in tags
        if (match := TAG.fullmatch(tag))
    )


def latest_support_tag(tags: list[str], branch: str) -> str:
    """Find the highest patch release on the given support line."""
    match = SUPPORT.fullmatch(branch)
    if match is None:
        raise ValueError(f"Not a support branch: {branch}")
    line = tuple(map(int, match.groups()))
    matching = [tag for version, tag in release_tags(tags) if version[:2] == line]
    if not matching:
        raise ValueError(f"No release tag found for {branch}")
    return matching[-1]


def previous_release_tag(tags: list[str], version: str) -> str:
    """Find the preceding published release for a proposed release version."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError("VERSION must have the form x.y.z")
    current = tuple(map(int, match.groups()))
    eligible = [
        tag
        for parts, tag in release_tags(tags)
        if parts < current and (current[2] == 0 or parts[:2] == current[:2])
    ]
    if not eligible:
        raise ValueError(f"No previous release tag found for {version}")
    return eligible[-1]


@dataclass
class Report:
    """Public API differences between two snapshots."""

    breaks: list[str]
    additions: set[str]
    categories: Counter[str]


def public_paths(  # noqa: C901
    root: griffe.Module,
) -> tuple[dict[str, griffe.Object | griffe.Alias], dict[str, set[str]]]:
    """Collect importable public paths and explicit exports in public modules."""
    paths: dict[str, griffe.Object | griffe.Alias] = {}
    exports: dict[str, set[str]] = {}

    def walk(module: griffe.Module) -> None:
        exported = module.exports
        if exported is not None and any(not isinstance(name, str) for name in exported):
            raise ValueError(f"Cannot resolve __all__ in {module.path}")
        exports[module.path] = {
            name
            for name in exported or []
            if isinstance(name, str) and not name.startswith("_")
        }

        def include_member(member: griffe.Object | griffe.Alias) -> None:
            name = member.name
            public = not name.startswith("_")
            member.public = public
            if not public:
                return
            paths[member.path] = member
            if not member.is_alias and member.is_module:
                walk(cast(griffe.Module, member))
            else:
                try:
                    if member.is_class or (
                        member.is_module
                        and member.is_alias
                        and cast(griffe.Alias, member).target_path.startswith(
                            f"{root.path}."
                        )
                    ):
                        for child in member.members.values():
                            include_member(child)
                except griffe.AliasResolutionError:
                    pass  # The name is public, but an external target is unavailable.

        for member in module.members.values():
            include_member(member)

    walk(root)
    return paths, exports


def compare(  # noqa: C901  # pylint: disable=too-many-locals,too-many-branches
    old: griffe.Module, new: griffe.Module
) -> Report:
    """Compare public paths and exports, then ask Griffe about their signatures."""
    old_paths, old_exports = public_paths(old)
    new_paths, new_exports = public_paths(new)
    findings: set[tuple[str, str, str]] = set()

    def add(kind: str, path: str, detail: str = "") -> None:
        findings.add((kind, path, detail))

    for path in old_paths.keys() - new_paths.keys():
        add("Public object removed", path)
    for module, names in old_exports.items():
        if module in new_exports:
            for name in names - new_exports[module]:
                add("Removed from __all__", f"{module}.{name}")

    for path in sorted(old_paths.keys() & new_paths.keys()):
        old_member = old_paths[path]
        new_member = new_paths[path]
        try:
            if old_member.parent is not None and old_member.parent.is_class:
                # Imported modules are paths, not nested Firecrown APIs.
                continue
        except griffe.AliasResolutionError:  # pragma: no cover
            # A parent alias would raise here if checking its is_class property
            # could not resolve its target. This is defensive: public_paths()
            # obtains children of an alias through member.members, which first
            # resolves that same parent. An unresolved alias can be collected
            # as a path, but not with children that reach this parent check.
            pass
        try:
            source = (
                cast(griffe.Alias, old_member).target
                if old_member.is_alias
                else old_member
            )
            replacement = (
                cast(griffe.Alias, new_member).target
                if new_member.is_alias
                else new_member
            )
            old_parent = griffe.Module("previous")
            new_parent = griffe.Module("current")
            old_parent.set_member("api", griffe.Alias("api", source))
            new_parent.set_member("api", griffe.Alias("api", replacement))
            for breakage in griffe.find_breaking_changes(old_parent, new_parent):
                target = breakage.obj.path
                replacement_path = replacement.path
                if target != replacement_path and not target.startswith(
                    f"{replacement_path}."
                ):
                    continue
                public_path = path + target[len(replacement_path) :]  # noqa: E203
                if any(part.startswith("_") for part in public_path.split(".")[1:]):
                    continue
                if (
                    breakage.kind == griffe.BreakageKind.OBJECT_REMOVED
                    and public_path not in new_paths
                ):
                    continue
                explanation = breakage.explain(griffe.ExplanationStyle.MARKDOWN)
                detail = explanation.split("*", 2)[-1]
                parameter = breakage.new_value or breakage.old_value
                if breakage.kind.name.startswith("PARAMETER_"):
                    detail = f" ({parameter.name}){detail}"
                add(breakage.kind.value, public_path, detail)
        except griffe.AliasResolutionError:
            continue  # External imports have no locally available signature to compare.

    return Report(
        breaks=[f"`{path}`: {kind}{detail}" for kind, path, detail in sorted(findings)],
        additions=new_paths.keys() - old_paths.keys(),
        categories=Counter(kind for kind, _, _ in findings),
    )


def git(*args: str) -> str:
    """Run a read-only Git query and return its output."""
    return subprocess.check_output(["git", *args], text=True).strip()


def should_fail(mode: str, target: str, report: Report) -> bool:
    """Only enforce compatibility on support lines and support-targeting PRs."""
    return bool(report.breaks) and (
        mode == "support" or mode == "pr" and bool(SUPPORT.fullmatch(target))
    )


def load(ref: str) -> griffe.Module:
    """Load Firecrown from the checkout or a committed Git ref."""
    if ref == "HEAD":
        result = griffe.load(
            "firecrown",
            search_paths=[Path.cwd()],
            resolve_aliases=True,
            resolve_implicit=True,
            resolve_external=False,
            allow_inspection=False,
        )
    else:
        result = griffe.load_git(
            "firecrown",
            ref=ref,
            repo=Path.cwd(),
            resolve_aliases=True,
            resolve_implicit=True,
            resolve_external=False,
            allow_inspection=False,
        )
    return cast(griffe.Module, result)


def markdown(report: Report, old: str, new: str) -> str:
    """Render the complete API change report as Markdown."""
    lines = [f"# Public API changes: {old} -> {new}", "", "## Executive summary", ""]
    if report.breaks:
        lines.extend(
            f"- {count} {kind}" for kind, count in sorted(report.categories.items())
        )
    else:
        lines.append("No breaking API changes detected.")
    lines.extend(["", "## Breaking changes", ""])
    lines.extend(f"- {item}" for item in report.breaks)
    if not report.breaks:
        lines.append("None.")
    lines.extend(["", "## New public API", ""])
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(report.additions):
        grouped[path.rpartition(".")[0]].append(path)
    for module, paths in sorted(grouped.items()):
        lines.extend([f"### `{module}`", "", *(f"- `{path}`" for path in paths), ""])
    if not grouped:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:  # noqa: C901  # pylint: disable=inconsistent-return-statements
    """Select a comparison baseline and write the resulting report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["support", "release", "pr"])
    parser.add_argument("--version", help="Proposed release version x.y.z")
    parser.add_argument("--base", help="PR target branch (or its fetched ref)")
    parser.add_argument(
        "--target", help="PR target branch name, for support-line policy"
    )
    parser.add_argument(
        "--output", help="Write Markdown to this path instead of stdout"
    )
    args = parser.parse_args()
    try:
        tags = git("tag", "--list", "v*").splitlines()
        if args.mode == "support":
            branch = git("branch", "--show-current")
            old = latest_support_tag(tags, branch)
            new = "HEAD"
        elif args.mode == "release":
            if not args.version:
                parser.error("release requires --version")
            old = previous_release_tag(tags, args.version)
            new = "HEAD"
        else:
            if not args.base:
                parser.error("pr requires --base")
            old = git("merge-base", "HEAD", args.base)
            new = "HEAD"
        if (
            args.mode == "support"
            and subprocess.run(
                ["git", "merge-base", "--is-ancestor", old, "HEAD"], check=False
            ).returncode
        ):
            raise ValueError(
                f"Latest release tag {old} is not on support branch {branch}"
            )
        report = compare(load(old), load(new))
        text = markdown(report, old, new)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return int(should_fail(args.mode, args.target or args.base or "", report))
    except (ValueError, subprocess.CalledProcessError, griffe.GriffeError) as exc:
        parser.exit(2, f"API comparison failed: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
