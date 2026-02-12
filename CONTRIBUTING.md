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
| `make lint -j` | Run all linters (`black`, `flake8`, `mypy`, `pylint`) in parallel | Before every commit |
| `make test -j` | Run fast unit tests in parallel | Regularly during development |
| `make unit-tests -j` | Run all unit tests with 100% per-component coverage check | Before pushing |
| `make test-ci` | Run the full test suite exactly as the CI system does | Final check before pushing |
| `make docs -j` | Build and verify all documentation (tutorials + API) | When changing tutorials or docstrings |
| `make pre-commit` | A comprehensive check: format, lint, docs-verify, and full tests | Recommended pre-push check |

> [!TIP]
> Always use the `-j` flag (e.g., `make -j lint`) to take full advantage of parallel execution. The `Makefile` automatically detects the number of available CPUs.

### Target Relationships

The following diagram shows how the key `Makefile` targets depend on each other:

```mermaid
graph TD
    classDef main fill:#f9f,stroke:#333,stroke-width:2px;
    
    pre-commit["make pre-commit"]:::main
    test-ci["make test-ci"]:::main
    docs-verify["make docs-verify"]
    unit-tests["make unit-tests"]
    
    pre-commit --> format["make format"]
    pre-commit --> lint["make lint"]
    pre-commit --> docs-verify
    pre-commit --> test-ci
    
    test-ci --> test-all-coverage["make test-all-coverage"]
    test-ci --> test-slow["make test-slow"]
    test-ci --> test-integration["make test-integration"]
    test-ci --> test-example["make test-example"]
    
    test-all-coverage --> unit-tests-post["unit-tests-post"]
    unit-tests --> unit-tests-post
    
    unit-tests-post --> test-updatable["test-updatable"]
    unit-tests-post --> test-utils["test-utils"]
    unit-tests-post --> test-parameters["test-parameters"]
    unit-tests-post --> test-modeling-tools["test-modeling-tools"]
    unit-tests-post --> test-models-cluster["test-models-cluster"]
    unit-tests-post --> test-models-two-point["test-models-two-point"]
    
    docs-verify --> tutorials["make tutorials"]
    docs-verify --> docs-linkcheck["make docs-linkcheck"]
    docs-verify --> docs-code-check["docs-code-check"]
    docs-verify --> docs-symbol-check["docs-symbol-check"]
    
    api-docs["make api-docs"] --> tutorials
    docs-linkcheck --> api-docs
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
