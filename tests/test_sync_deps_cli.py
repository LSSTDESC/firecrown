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
ENVIRONMENTS = [
    (python, platform)
    for python in ("3.12", "3.13", "3.14")
    for platform in ("linux-64", "osx-arm64")
]


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


def _set_locked_version(
    root: Path,
    environment: tuple[str, str],
    name: str,
    manager: str,
    version: str,
) -> None:
    """Replace one environment's committed direct selection.

    :param root: Isolated repository fixture.
    :param environment: Python and platform identities of the locked selection.
    :param name: Locked package identity.
    :param manager: Installation route of the selection.
    :param version: Replacement locked version.
    """
    python, platform = environment
    lock_path = root / f".github/conda-lock/py{python}.conda-lock.yml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    for package in lock["package"]:
        if (
            package["name"] == name
            and package["manager"] == manager
            and package["platform"] == platform
        ):
            package["version"] = version
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")


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
        f'environment-sha256: "{digest}"\nconstraints:\n  sample: ">=1.0,<1.1"\n',
        encoding="utf-8",
    )
    lock_dir = tmp_path / ".github/conda-lock"
    lock_dir.mkdir()
    for version in ("3.12", "3.13", "3.14"):
        (lock_dir / f"py{version}.conda-lock.yml").write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "metadata": {"platforms": ["linux-64", "osx-arm64"]},
                    "package": [
                        {
                            "name": name,
                            "version": f"{version}.1" if name == "python" else "1.0",
                            "manager": "conda",
                            "platform": platform,
                        }
                        for platform in ("linux-64", "osx-arm64")
                        for name in ("python", "sample", "dev-tool")
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def locked_repository(repository: Path) -> Path:
    """Add direct requirements routed through conda and pip to every lock.

    :param repository: Isolated repository with six baseline lock selections.
    :returns: Repository with all manifest groups and installation routes.
    """
    manifest_path = repository / "dependencies.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"] = [
        {"name": "sample", "conda": "sample-conda", "pip": "sample-pypi"},
        {"name": "pip-runtime", "conda": False, "pip": "pip_runtime"},
        {"name": "pip", "pip": False},
    ]
    manifest["workarounds"] = [
        {"name": "workaround", "conda": "workaround-conda", "pip": "workaround-pypi"},
        {"name": "pip-workaround", "conda": False, "pip": "pip_workaround"},
    ]
    manifest["devenv"] = [
        {"name": "dev-tool", "conda": "dev-tool-conda", "pip": False},
        {"name": "pip-dev", "conda": False, "pip": "pip_dev"},
    ]
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    for lock_path in (repository / ".github/conda-lock").glob("*.yml"):
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        lock["package"] = [
            {
                "name": name,
                "version": (
                    lock_path.name.removeprefix("py").split(".conda-lock")[0] + ".1"
                    if name == "python"
                    else "1.0"
                ),
                "manager": manager,
                "platform": platform,
            }
            for platform in ("linux-64", "osx-arm64")
            for name, manager in (
                ("python", "conda"),
                ("pip", "conda"),
                ("sample-conda", "conda"),
                ("pip_runtime", "pip"),
                ("workaround-conda", "conda"),
                ("pip_workaround", "pip"),
                ("dev-tool-conda", "conda"),
                ("pip_dev", "pip"),
            )
        ]
        lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    sync = _run_cli(repository)
    assert (repository / "environment.yml").read_text(encoding="utf-8").count(
        "pip_runtime"
    ) == 1, sync.stderr
    environment = yaml.safe_load(
        (repository / "environment.yml").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(
        yaml.safe_dump(environment, sort_keys=True).encode()
    ).hexdigest()
    (repository / "dependencies-validated.yaml").write_text(
        f'environment-sha256: "{digest}"\n'
        'constraints:\n  pip: ">=1.0,<1.1"\n'
        '  sample-conda: ">=1.0,<1.1"\n'
        '  workaround-conda: ">=1.0,<1.1"\n',
        encoding="utf-8",
    )
    return repository


def test_cli_check_accepts_complete_lock_presence_without_writing(
    locked_repository: Path,
) -> None:
    """Each direct requirement is present in all six lock selections.

    :param locked_repository: Repository with complete conda and pip selections.
    """
    before = _snapshot(locked_repository)
    result = _run_cli(locked_repository, "--check")
    assert result.returncode == 0, result.stderr
    assert "presence" in result.stdout.lower()
    assert _snapshot(locked_repository) == before


@pytest.mark.parametrize("change_kind", ["relevant", "irrelevant"])
@pytest.mark.parametrize("consistent", [True, False])
def test_pr_consistency_gate_is_unconditional_for_path_and_checker_outcomes(
    repository: Path, change_kind: str, consistent: bool
) -> None:
    """Workflow wiring and production checker preserve the independent gate.

    :param repository: Isolated repository with matching declarations and locks.
    :param change_kind: Whether a PR edit is relevant to the rebuild canary.
    :param consistent: Whether committed dependency artifacts agree.
    """
    workflow = yaml.load(
        (SCRIPT.parents[2] / ".github/workflows/ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    jobs = workflow["jobs"]
    consistency = jobs["pr-dependency-consistency"]
    assert "if" not in consistency
    assert any(
        ".github/scripts/sync_deps.py --check" in step.get("run", "")
        for step in consistency["steps"]
    )
    canary = jobs["pr-rebuild-drift"]
    assert "pr-dependency-consistency" in canary["needs"]
    assert "needs.pr-drift-paths.outputs.should_run" in canary["if"]
    assert "needs.pr-dependency-consistency.result == 'success'" in canary["if"]

    changed_file = repository / (
        "dependencies.yaml" if change_kind == "relevant" else "README.md"
    )
    existing = changed_file.read_text(encoding="utf-8") if changed_file.exists() else ""
    changed_file.write_text(f"{existing}\n# PR edit\n", encoding="utf-8")
    if not consistent:
        _set_locked_version(repository, ("3.13", "osx-arm64"), "sample", "conda", "0.9")

    result = _run_cli(repository, "--check")

    assert result.returncode == (0 if consistent else 1), result.stderr


def test_cli_check_rejects_incompatible_lock_with_current_digest(
    repository: Path,
) -> None:
    """A current digest cannot make an incompatible direct selection pass.

    :param repository: Matching declarations and current historical pin digest.
    """
    lock_path = repository / ".github/conda-lock/py3.13.conda-lock.yml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    for package in lock["package"]:
        if package["name"] == "sample" and package["platform"] == "osx-arm64":
            package["version"] = "0.9"
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    before = _snapshot(repository)
    result = _run_cli(repository, "--check")
    assert result.returncode == 1
    assert "Python 3.13 / osx-arm64" in result.stderr
    assert "runtime requirement sample (conda:sample)" in result.stderr
    assert "0.9" in result.stderr and ">=1" in result.stderr
    assert _snapshot(repository) == before


@pytest.mark.parametrize(
    "constraint, version, compatible",
    [
        ("==1.0", "1.0+local", True),
        ("<1.0", "1.0rc1", False),
        (">=1,<2", "1.5rc1", True),
        ("~=1.4.2", "1.4.9", True),
        ("~=1.4.2", "1.5", False),
        ("==1.0.*", "1.0.9", True),
        (">=1,!=1.5", "1.5", False),
    ],
)
def test_cli_check_uses_pip_version_semantics(
    locked_repository: Path, constraint: str, version: str, compatible: bool
) -> None:
    """Pip selections obey PEP 440, including existing prerelease selections.

    :param locked_repository: Repository with mapped conda and pip requirements.
    :param constraint: PEP 440 requirement to apply to the pip runtime package.
    :param version: Locked version selected on every supported environment.
    :param compatible: Expected compatibility verdict from the stated examples.
    """
    manifest_path = locked_repository / "dependencies.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"][1]["version"] = constraint
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    _run_cli(locked_repository)
    for lock_path in (locked_repository / ".github/conda-lock").glob("*.yml"):
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        for package in lock["package"]:
            if package["manager"] == "pip" and package["name"] == "pip_runtime":
                package["version"] = version
                package["name"] = "PIP.Runtime"
        lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    # Keep the historical digest current to isolate the compatibility verdict.
    environment = yaml.safe_load(
        (locked_repository / "environment.yml").read_text(encoding="utf-8")
    )
    pins_path = locked_repository / "dependencies-validated.yaml"
    pins = yaml.safe_load(pins_path.read_text(encoding="utf-8"))
    pins["environment-sha256"] = hashlib.sha256(
        yaml.safe_dump(environment, sort_keys=True).encode()
    ).hexdigest()
    pins_path.write_text(yaml.safe_dump(pins, sort_keys=False), encoding="utf-8")
    before = _snapshot(locked_repository)
    result = _run_cli(locked_repository, "--check")
    assert result.returncode == (0 if compatible else 1), result.stderr
    if compatible:
        assert "lock compatibility pass" in result.stdout
    else:
        assert "pip-runtime (pip:pip_runtime)" in result.stderr
        assert constraint in result.stderr and version in result.stderr
    assert _snapshot(locked_repository) == before


def test_cli_check_accepts_compatible_older_locks_with_stale_digests(
    repository: Path,
) -> None:
    """Historical input changes alone do not require a fresh solve.

    :param repository: Matching declarations with older compatible selections.
    """
    pins_path = repository / "dependencies-validated.yaml"
    pins = yaml.safe_load(pins_path.read_text(encoding="utf-8"))
    pins["environment-sha256"] = "historical-environment"
    pins_path.write_text(yaml.safe_dump(pins, sort_keys=False), encoding="utf-8")
    for lock_path in (repository / ".github/conda-lock").glob("*.yml"):
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        lock["metadata"]["content_hash"] = {
            "linux-64": "historical-lock-input",
            "osx-arm64": "historical-lock-input",
        }
        lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    before = _snapshot(repository)
    result = _run_cli(repository, "--check")
    assert result.returncode == 0, result.stderr
    assert "lock compatibility pass" in result.stdout
    assert _snapshot(repository) == before
    pins_result = _run_cli(repository, "--pins", "--check")
    assert pins_result.returncode == 0, pins_result.stderr
    assert _snapshot(repository) == before


def test_cli_check_accepts_constraints_reproduced_from_committed_locks(
    repository: Path,
) -> None:
    """Validated constraints derive from changed committed selections.

    :param repository: Matching declarations and baseline committed locks.
    """
    for python, platform in ENVIRONMENTS:
        _set_locked_version(repository, (python, platform), "sample", "conda", "2.1")
    pins_path = repository / "dependencies-validated.yaml"
    pins = yaml.safe_load(pins_path.read_text(encoding="utf-8"))
    pins["constraints"]["sample"] = ">=2.1,<2.2"
    pins_path.write_text(yaml.safe_dump(pins, sort_keys=False), encoding="utf-8")
    before = _snapshot(repository)

    result = _run_cli(repository, "--check")

    assert result.returncode == 0, result.stderr
    assert "Validated constraints match committed locks." in result.stdout
    assert _snapshot(repository) == before


def test_cli_check_reports_validated_constraint_drift_from_committed_locks(
    repository: Path,
) -> None:
    """Changed locked selections make stale validated constraints fail clearly.

    :param repository: Matching declarations and baseline committed locks.
    """
    for python, platform in ENVIRONMENTS:
        _set_locked_version(repository, (python, platform), "sample", "conda", "2.1")
    before = _snapshot(repository)

    result = _run_cli(repository, "--check")

    assert result.returncode == 1
    assert "validated constraint sample differs from committed locks" in result.stderr
    assert "recorded >=1.0,<1.1; regenerated >=2.1,<2.2" in result.stderr
    assert _snapshot(repository) == before
    pins_result = _run_cli(repository, "--pins", "--check")
    assert pins_result.returncode == 1
    assert (
        "validated constraint sample differs from committed locks" in pins_result.stderr
    )
    assert _snapshot(repository) == before


@pytest.mark.parametrize("python, platform", ENVIRONMENTS)
def test_cli_check_rejects_wrong_python_identity(
    repository: Path, python: str, platform: str
) -> None:
    """A valid manifest interpreter cannot stand in for another supported line.

    :param repository: Repository with complete supported selections.
    :param python: Supported Python identity whose selection is changed.
    :param platform: Supported platform whose selection is changed.
    """
    _set_locked_version(repository, (python, platform), "python", "conda", "3.15.1")
    before = _snapshot(repository)
    result = _run_cli(repository, "--check")
    assert result.returncode == 1
    assert f"Python {python} / {platform}" in result.stderr
    assert "python (conda:python)" in result.stderr and "3.15.1" in result.stderr
    assert _snapshot(repository) == before


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize(
    "requirement",
    [
        ("runtime", "sample", "sample-conda", "conda"),
        ("runtime", "pip-runtime", "pip_runtime", "pip"),
        ("workarounds", "workaround", "workaround-conda", "conda"),
        ("workarounds", "pip-workaround", "pip_workaround", "pip"),
        ("devenv", "dev-tool", "dev-tool-conda", "conda"),
        ("devenv", "pip-dev", "pip_dev", "pip"),
    ],
)
def test_cli_check_validates_each_requirement_in_each_environment(
    locked_repository: Path,
    environment: tuple[str, str],
    requirement: tuple[str, str, str, str],
) -> None:
    """Compatible selections elsewhere cannot mask one incompatible selection.

    :param locked_repository: Repository with complete routed selections.
    :param environment: Python and platform identities of the selected environment.
    :param requirement: Group, logical identity, locked name and manager to constrain.
    """
    python, platform = environment
    group, identity, name, manager = requirement
    manifest_path = locked_repository / "dependencies.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest[group]:
        if entry["name"] == identity:
            entry["version"] = ">=1,<2"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    _run_cli(locked_repository)
    before = _snapshot(locked_repository)
    compatible = _run_cli(locked_repository, "--check")
    assert compatible.returncode == 0, compatible.stderr
    assert _snapshot(locked_repository) == before
    _set_locked_version(locked_repository, environment, name, manager, "0.9")
    before = _snapshot(locked_repository)
    incompatible = _run_cli(locked_repository, "--check")
    assert incompatible.returncode == 1
    assert f"Python {python} / {platform}" in incompatible.stderr
    assert f"{group} requirement {identity} ({manager}:{name})" in incompatible.stderr
    assert "locked 0.9, requires >=1,<2" in incompatible.stderr
    assert _snapshot(locked_repository) == before


@pytest.mark.parametrize(
    "constraint, version, compatible",
    [
        ("==1.0", "1.0+local", False),
        ("<1.0", "1.0rc1", True),
        ("=1.0", "1.0.9", True),
        ("1.0|2.0", "2.0", True),
        ("1.0|2.0", "3.0", False),
        ("~=1.4.2", "1.4.9", True),
        ("~=1.4.2", "1.5", False),
        (">=1,!=1.5", "1.5", False),
    ],
)
def test_cli_check_uses_conda_version_semantics(
    locked_repository: Path, constraint: str, version: str, compatible: bool
) -> None:
    """Conda versions use conda ordering, prefix, union and bound semantics.

    :param locked_repository: Repository with mapped direct requirements.
    :param constraint: Conda version constraint for the development requirement.
    :param version: Version selected in one supported environment.
    :param compatible: Expected verdict for the stated conda example.
    """
    manifest_path = locked_repository / "dependencies.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["devenv"][0]["version"] = constraint
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    _run_cli(locked_repository)
    # All environments must satisfy the new constraint before one is changed.
    baseline = (
        "1.4.2"
        if constraint == "~=1.4.2"
        else "1.0rc1" if constraint == "<1.0" else "1.0"
    )
    for environment in ENVIRONMENTS:
        _set_locked_version(
            locked_repository, environment, "dev-tool-conda", "conda", baseline
        )
    _set_locked_version(
        locked_repository, ("3.12", "linux-64"), "dev-tool-conda", "conda", version
    )
    before = _snapshot(locked_repository)
    result = _run_cli(locked_repository, "--check")
    assert result.returncode == (0 if compatible else 1), result.stderr
    if not compatible:
        assert "Python 3.12 / linux-64" in result.stderr
        assert "devenv requirement dev-tool (conda:dev-tool-conda)" in result.stderr
        assert constraint in result.stderr and version in result.stderr
    assert _snapshot(locked_repository) == before


def test_cli_check_rejects_missing_supported_environment_without_writing(
    locked_repository: Path,
) -> None:
    """A missing platform fails even when the other lockfiles are complete.

    :param locked_repository: Repository with complete conda and pip selections.
    """
    lock_path = locked_repository / ".github/conda-lock/py3.13.conda-lock.yml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lock["metadata"]["platforms"].remove("osx-arm64")
    lock["package"] = [
        package for package in lock["package"] if package["platform"] != "osx-arm64"
    ]
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    before = _snapshot(locked_repository)
    result = _run_cli(locked_repository, "--check")
    assert result.returncode == 1
    assert "Python 3.13" in result.stderr
    assert "osx-arm64" in result.stderr
    assert _snapshot(locked_repository) == before


def test_cli_check_rejects_missing_lockfile_without_writing(
    locked_repository: Path,
) -> None:
    """Expected Python coverage is independent of the files on disk.

    :param locked_repository: Repository with complete conda and pip selections.
    """
    lock_path = locked_repository / ".github/conda-lock/py3.12.conda-lock.yml"
    lock_path.unlink()
    before = _snapshot(locked_repository)
    result = _run_cli(locked_repository, "--check")
    assert result.returncode == 1
    assert "Python 3.12" in result.stderr
    assert "py3.12.conda-lock.yml" in result.stderr
    assert _snapshot(locked_repository) == before


@pytest.mark.parametrize(
    "name, manager, identity",
    [
        ("sample-conda", "conda", "sample"),
        ("pip_runtime", "pip", "pip-runtime"),
        ("workaround-conda", "conda", "workaround"),
        ("dev-tool-conda", "conda", "dev-tool"),
        ("pip_dev", "pip", "pip-dev"),
    ],
)
def test_cli_check_rejects_package_missing_in_one_environment(
    locked_repository: Path, name: str, manager: str, identity: str
) -> None:
    """A selection elsewhere cannot satisfy a missing direct requirement.

    :param locked_repository: Repository with complete conda and pip selections.
    :param name: Locked package name to remove from one environment.
    :param manager: Installation manager for the locked package.
    :param identity: Manifest requirement identity used in diagnostics.
    """
    lock_path = locked_repository / ".github/conda-lock/py3.14.conda-lock.yml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lock["package"] = [
        package
        for package in lock["package"]
        if not (
            package["platform"] == "linux-64"
            and package["name"] == name
            and package["manager"] == manager
        )
    ]
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    before = _snapshot(locked_repository)
    result = _run_cli(locked_repository, "--check")
    assert result.returncode == 1
    assert "Python 3.14" in result.stderr
    assert "linux-64" in result.stderr
    assert identity in result.stderr
    assert _snapshot(locked_repository) == before


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
        "    - sample >=1.0,<1.1\n"
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
        (
            "feedstock/recipe/meta.yaml",
            "- sample >=1.0,<1.1",
            "- sample >=2,<3",
        ),
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
