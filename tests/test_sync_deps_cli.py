"""Exercise dependency declaration synchronization through the production CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/sync_deps.py"


def _run_cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the copied production script against the isolated repository.

    :param root: Repository containing the script and declaration fixtures.
    :param args: Arguments passed to the dependency CLI.
    :returns: Exit status and captured output from the CLI.
    """
    return subprocess.run(
        [sys.executable, str(root / ".github/scripts/sync_deps.py"), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _snapshot(root: Path) -> dict[Path, bytes]:
    """Record every fixture file to detect writes or new files during checking.

    :param root: Isolated repository and feedstock fixture directory.
    :returns: File paths and their exact contents.
    """
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create matching declarations and real pin metadata for the CLI.

    :param tmp_path: Temporary directory supplied by pytest.
    :returns: Root of the isolated repository containing the production script.
    """
    script = tmp_path / ".github/scripts/sync_deps.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    (tmp_path / "dependencies.yaml").write_text(
        'python: ">=3.12"\nruntime:\n'
        '  - {name: sample, version: ">=1", pip: true}\n'
        "workarounds: []\ndevenv:\n  - {name: dev-tool, pip: false}\n",
        encoding="utf-8",
    )
    environment = (
        "# Generated from dependencies.yaml by `make deps-sync` -- do not edit.\n"
        "channels:\n  - conda-forge\ndependencies:\n"
        "  - dev-tool\n  - python >=3.12\n  - sample >=1\n"
    )
    (tmp_path / "environment.yml").write_text(environment, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nrequires-python = ">=3.12"\n'
        'dependencies = [\n    "sample>=1",\n]\n',
        encoding="utf-8",
    )
    # This digest supplies the existing pin freshness stage's input; declaration
    # expectations above are literals independent of the production renderers.
    digest = hashlib.sha256(
        yaml.safe_dump(yaml.safe_load(environment), sort_keys=True).encode()
    ).hexdigest()
    (tmp_path / "dependencies-validated.yaml").write_text(
        f'environment-sha256: "{digest}"\nconstraints:\n  sample: ">=1,<2"\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize(
    "declaration",
    [
        'requires-python = ">=3.11"',
        'requires-python = ">=3.11"  # supported interpreter',
        'requires-python = ">=3.11"   ',
    ],
    ids=["other-table-first", "inline-comment", "trailing-spaces"],
)
def test_cli_sync_updates_only_project_python_requirement(
    repository: Path, declaration: str
) -> None:
    """Synchronize project Python metadata while preserving other TOML tables.

    :param repository: Isolated repository with matching declaration fixtures.
    :param declaration: Valid TOML spelling of a drifted project requirement.
    """
    pyproject = repository / "pyproject.toml"
    pyproject.write_text(
        '[tool.fixture]\nrequires-python = ">=3.9"\n'
        + pyproject.read_text(encoding="utf-8").replace(
            'requires-python = ">=3.12"', declaration
        ),
        encoding="utf-8",
    )
    before = _snapshot(repository)
    check = _run_cli(repository, "--check")
    assert check.returncode == 1
    assert "pyproject.toml is out of date" in check.stderr
    assert _snapshot(repository) == before

    sync = _run_cli(repository)
    assert sync.returncode == 0, sync.stderr
    parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert parsed["project"]["requires-python"] == ">=3.12"
    assert parsed["tool"]["fixture"]["requires-python"] == ">=3.9"
    if "# supported interpreter" in declaration:
        assert "# supported interpreter" in pyproject.read_text(encoding="utf-8")
    before = _snapshot(repository)
    assert _run_cli(repository, "--check").returncode == 0
    assert _snapshot(repository) == before


@pytest.fixture
def feedstock(repository: Path) -> Path:
    """Add a recipe containing each generated block to the fixture repository.

    :param repository: Isolated repository holding the dependency declarations.
    :returns: Feedstock directory with a synchronized recipe.
    """
    recipe = repository / "feedstock/recipe/meta.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        '{% set version = "fixture" %}\n'
        "requirements:\n  run:\n"
        "    # BEGIN GENERATED firecrown-deps\n"
        "    - python >=3.12\n    - sample >=1\n"
        "    # END GENERATED firecrown-deps\n"
        "  run_constrained:\n"
        "    # BEGIN GENERATED firecrown-deps-validated\n"
        "    - sample >=1,<2\n"
        "    # END GENERATED firecrown-deps-validated\n"
        "test:\n  commands:\n"
        "    # BEGIN GENERATED firecrown-deps-imports\n"
        '    - python -c "import sample"\n'
        "    # END GENERATED firecrown-deps-imports\n"
        "devenv:\n  requirements:\n"
        "    # BEGIN GENERATED firecrown-devenv\n"
        "    - dev-tool\n"
        "    # END GENERATED firecrown-devenv\n",
        encoding="utf-8",
    )
    # Synchronization adds version annotations to the markers. The dependency
    # lines are independently specified above and must survive unchanged.
    before = recipe.read_text(encoding="utf-8")
    result = _run_cli(
        repository, "--feedstock", str(recipe.parents[1]), "--allow-version-mismatch"
    )
    assert result.returncode == 0, result.stderr
    after = recipe.read_text(encoding="utf-8")
    expected_lines = [
        line for line in before.splitlines() if "BEGIN GENERATED" not in line
    ]
    actual_lines = [
        line for line in after.splitlines() if "BEGIN GENERATED" not in line
    ]
    assert actual_lines == expected_lines
    return recipe.parents[1]


@pytest.mark.parametrize("include_feedstock", [False, True], ids=["local", "feedstock"])
def test_cli_check_accepts_matching_declarations_without_writing(
    repository: Path, feedstock: Path, include_feedstock: bool
) -> None:
    """Matching declarations pass without changing any fixture file.

    :param repository: Repository holding the matching declaration fixtures.
    :param feedstock: Matching generated recipe fixture.
    :param include_feedstock: Whether to check the optional recipe too.
    """
    args = (
        ("--feedstock", str(feedstock), "--allow-version-mismatch")
        if include_feedstock
        else ()
    )
    before = _snapshot(repository)
    result = _run_cli(repository, "--check", *args)
    assert result.returncode == 0, result.stderr
    assert "is out of date" not in result.stderr
    assert _snapshot(repository) == before


@pytest.mark.parametrize(
    "relative_path, original, drifted",
    [
        ("environment.yml", "sample >=1", "sample >=2"),
        ("pyproject.toml", '"sample>=1"', '"sample>=2"'),
        ("pyproject.toml", 'requires-python = ">=3.12"', 'requires-python = ">=3.11"'),
        ("feedstock/recipe/meta.yaml", "- sample >=1\n", "- sample >=2\n"),
        ("feedstock/recipe/meta.yaml", "- sample >=1,<2", "- sample >=2,<3"),
        ("feedstock/recipe/meta.yaml", 'import sample"', 'import other"'),
        ("feedstock/recipe/meta.yaml", "- dev-tool", "- other-tool"),
    ],
    ids=[
        "environment",
        "project-dependencies",
        "python-requirement",
        "recipe-runtime",
        "recipe-validated",
        "recipe-imports",
        "recipe-development",
    ],
)
def test_cli_check_detects_declaration_drift_without_writing(
    repository: Path, feedstock: Path, relative_path: str, original: str, drifted: str
) -> None:
    """Each generated declaration independently fails with an attributable diff.

    :param repository: Isolated repository holding declaration fixtures.
    :param feedstock: Matching recipe included in the check.
    :param relative_path: File containing the declaration to change.
    :param original: Matching declaration text to replace.
    :param drifted: Inconsistent declaration substituted for the original.
    """
    path = repository / relative_path
    content = path.read_text(encoding="utf-8")
    assert original in content
    path.write_text(content.replace(original, drifted, 1), encoding="utf-8")
    before = _snapshot(repository)
    result = _run_cli(
        repository, "--check", "--feedstock", str(feedstock), "--allow-version-mismatch"
    )
    assert result.returncode == 1
    assert f"{path} is out of date" in result.stderr
    assert "--- " in result.stderr and "+++ generated" in result.stderr
    assert _snapshot(repository) == before
