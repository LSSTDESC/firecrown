# Contributing to Firecrown

Thank you for your interest in contributing to Firecrown! This document provides guidelines and workflows to help you get started.

## Development Environment Setup

We recommend using `conda` (or `mamba`/`miniforge`) to manage your development environment.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/LSSTDESC/firecrown.git
   cd firecrown
   ```

2. **Create and activate the environment:**

   ```bash
   conda env create --name firecrown_developer --file environment.yml
   conda activate firecrown_developer
   ```

3. **Install Firecrown in development mode:**

   ```bash
   make install
   ```

## Recommended Developer Workflow

To maintain high code quality and consistency, we use several automated tools. We recommend following this workflow during development:

| Target | Description | When to run |
| :--- | :--- | :--- |
| `make format` | Automatically format all code using `black` | Frequently during development |
| `make lint` | Run all linters (`black`, `flake8`, `mypy`, `pylint`) in parallel | Before every commit |
| `make test` | Run fast unit tests in parallel | Regularly during development |
| `make unit-tests` | Run all unit tests with 100% per-component coverage check | Before pushing |
| `make test-ci` | Run the full test suite exactly as the CI system does | Final check before pushing |
| `make docs` | Build and verify all documentation (tutorials + API) | When changing tutorials or docstrings |
| `make pre-commit` | A comprehensive check: format, lint, docs-verify, and full tests | Recommended pre-push check |

### Dependencies

`environment.yml` and the `dependencies` list in `pyproject.toml` are generated
from [`dependencies.yaml`](dependencies.yaml), which is the single source of
truth for every dependency of the project. Add or change a dependency there and
regenerate:

```bash
make deps-sync
```

`make deps-check`, which `make pre-commit` runs for you, fails if the generated
files no longer match the manifest.

The manifest has three groups. `runtime` is what firecrown imports. `devenv` is
what developing, testing, documenting and running the examples needs — the link
checker's `bs4` and `requests` live here, since `make docs-linkcheck` is the
only thing that runs it. `workarounds` is neither: packages firecrown does not
import but has to constrain anyway, either to route around an upstream bug or
because whatever pulls them in does not bound them itself. Give every
`workarounds` entry a `note` saying what it works around, so the constraint can
be retired when the upstream fix lands instead of outliving the bug.

`deps-sync` also writes `dependencies-validated.yaml`, the version ranges the
committed lockfiles resolve to. It is generated, but committed and reviewed:
changes to the pins should be legible in a pull request rather than buried in a
lockfile regeneration.

The same manifest generates the requirement lists of the [conda-forge
recipe](https://github.com/conda-forge/firecrown-feedstock), which builds the
`firecrown`, `firecrown-deps`, `firecrown-deps-validated`, `firecrown-validated`
and `firecrown-devenv` packages. With a feedstock checkout available:

```bash
make feedstock-sync FEEDSTOCK=../firecrown-feedstock
```

This refuses to run unless the checked-out tree is the release the recipe
builds, so that a development manifest cannot be written onto a released
recipe; check out the release tag first. `ALLOW_VERSION_MISMATCH=1` overrides
it, which is for introducing the generated blocks to a recipe, not for routine
use. Either way each block records the version it came from:

```yaml
# BEGIN GENERATED firecrown-deps (firecrown 1.15.2)
```

The manifest, the validated pins and `sync_deps.py` ship in the sdist, and the
recipe's test sections run `sync_deps.py --check-installed` against the
metapackage that was just built. A version bump that leaves the generated
blocks behind therefore fails in conda-forge CI instead of silently shipping
the previous release's dependencies.

### Conda Lockfiles

After changing dependencies, regenerate the lockfiles:

```bash
make conda-lock
```

Before pushing, verify the lockfiles are up to date:

```bash
make conda-lock-check
```

This ensures CI can install the environment from the committed lockfiles. The lockfiles are stored in `.github/conda-lock/` as unified format files (`py{version}.conda-lock.yml`) that support Python versions 3.12-3.14 on both Linux and macOS.

> [!TIP]
> The `Makefile` automatically runs targets in parallel and detects the number of available CPUs. Use `make -j1 <target>` to run serially (useful for debugging), or `JOBS=N make <target>` to override the number of jobs.

For detailed diagrams of how `Makefile` targets relate to each other, how parallelism
works, and how the CI pipeline is structured, see
[CONTRIBUTING_ADVANCED.md](CONTRIBUTING_ADVANCED.md).

To regenerate lockfiles and manage the lockfile generation process, see
[CONTRIBUTING_ADVANCED.md#conda-lock](CONTRIBUTING_ADVANCED.md#conda-lock).

## Pull Request Process

1. **Create a Branch**: Always work on a new branch for your feature or bug fix.
2. **Write Tests**: Ensure your changes are covered by unit tests. We aim for 100% coverage on new code.
3. **Verify Locally**: Run `make pre-commit` to ensure everything is in order. If you modified `dependencies.yaml`, also run `make deps-sync`, then `make conda-lock`, and commit the regenerated files.
4. **Submit PR**: Once your tests pass locally, submit a Pull Request to the `master` branch.
5. **CI Pipeline**: Our CI system will run the full test matrix on Ubuntu and macOS with various Python versions. Your PR must pass all CI checks before it can be merged.

## Coding Style

- Use `black` for formatting.
- Follow PEP 8 guidelines (enforced by `flake8`).
- Use type hints wherever possible (checked by `mypy`).
- Ensure `pylint` passes without warnings in the relevant packages.

## Conda Lockfiles

Developers must have `conda-lock` installed to regenerate lockfiles. Install it with:

```bash
pip install conda-lock==4.0.2
```

The `CONTRIBUTING_ADVANCED.md` file contains detailed information about lockfile management and the CI system.
