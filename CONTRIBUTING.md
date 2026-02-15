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
   conda env create -f environment.yml
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

> [!TIP]
> The `Makefile` automatically runs targets in parallel and detects the number of available CPUs. Use `make -j1 <target>` to run serially (useful for debugging), or `JOBS=N make <target>` to override the number of jobs.

### Target Relationships

The following diagram shows how the key `Makefile` targets depend on each other:

```mermaid
graph TD
    classDef main fill:#f9f,stroke:#333,stroke-width:2px;
    
    pre-commit["make pre-commit"]:::main
    test-ci["make test-ci"]:::main
    docs["make docs"]:::main
    unit-tests["make unit-tests"]:::main
    
    %% Pre-commit dependencies
    pre-commit --> format["make format"]
    pre-commit --> lint["make lint"]
    pre-commit --> docs-verify["make docs-verify"]
    pre-commit --> test-ci
    
    %% Test-CI dependencies
    test-ci --> unit-tests-pre["unit-tests-pre"]
    test-ci --> test-all-coverage["make test-all-coverage"]
    test-ci --> test-slow["make test-slow"]
    test-ci --> test-integration["make test-integration"]
    test-ci --> test-example["make test-example"]
    
    %% Test coverage dependencies
    test-all-coverage --> unit-tests-core["unit-tests-core"]
    test-all-coverage --> unit-tests-post["unit-tests-post"]
    
    %% Unit tests dependencies
    unit-tests --> unit-tests-post
    
    %% Unit tests post dependencies
    unit-tests-post --> test-updatable["test-updatable"]
    unit-tests-post --> test-utils["test-utils"]
    unit-tests-post --> test-parameters["test-parameters"]
    unit-tests-post --> test-modeling-tools["test-modeling-tools"]
    unit-tests-post --> test-models-cluster["test-models-cluster"]
    unit-tests-post --> test-models-two-point["test-models-two-point"]
    
    %% Lint dependencies
    lint --> lint-black["lint-black"]
    lint --> lint-flake8["lint-flake8"]
    lint --> lint-mypy["lint-mypy"]
    lint --> lint-pylint["lint-pylint"]
    
    %% Documentation dependencies
    docs --> docs-build["make docs-build"]
    docs --> docs-verify
    
    docs-build --> tutorials["make tutorials"]
    docs-build --> api-docs["make api-docs"]
    
    docs-verify --> docs-generate-symbol-map["docs-generate-symbol-map"]
    docs-verify --> docs-code-check["docs-code-check"]
    docs-verify --> docs-symbol-check["docs-symbol-check"]
    docs-verify --> docs-linkcheck["make docs-linkcheck"]
    
    tutorials --> docs-generate-symbol-map
    api-docs --> tutorials
    docs-code-check --> tutorials
    docs-symbol-check --> tutorials
    docs-symbol-check --> docs-generate-symbol-map
    docs-linkcheck --> docs-build
```

## Pull Request Process

1. **Create a Branch**: Always work on a new branch for your feature or bug fix.
2. **Write Tests**: Ensure your changes are covered by unit tests. We aim for 100% coverage on new code.
3. **Verify Locally**: Run `make pre-commit` to ensure everything is in order.
4. **Submit PR**: Once your tests pass locally, submit a Pull Request to the `master` branch.
5. **CI Pipeline**: Our CI system will run the full test matrix on Ubuntu and macOS with various Python versions. Your PR must pass all CI checks before it can be merged.

## Coding Style

- Use `black` for formatting.
- Follow PEP 8 guidelines (enforced by `flake8`).
- Use type hints wherever possible (checked by `mypy`).
- Ensure `pylint` passes without warnings in the relevant packages.
