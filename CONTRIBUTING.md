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
| `make test` | Run unit tests in parallel with coverage reporting | Regularly during development |
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
    test-all["make test-all"]:::main
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
    
    %% Test-all dependencies
    test-all --> test["make test\n(+coverage)"]
    test-all --> test-slow
    test-all --> test-integration
    test-all --> test-example
    
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

### Parallelism Architecture

The Makefile supports two levels of parallelism:

- **Make-level**: Multiple independent targets run simultaneously (enabled by default with `-j`)
- **Pytest-level**: Test suites use `pytest -n auto` to parallelize individual test execution

The following diagram shows which targets support parallel execution of their subtargets:

```mermaid
graph TD
    classDef parallel fill:#c7f5c7,stroke:#2d8e2d,stroke-width:2px;
    classDef sequential fill:#ffd9b3,stroke:#cc6600,stroke-width:2px;
    classDef pytest fill:#d4e6f1,stroke:#2874a6,stroke-width:2px;
    
    pre-commit["make pre-commit<br/>(parallel)"]:::parallel
    
    %% Pre-commit parallel branches
    pre-commit --> format["make format"]:::sequential
    pre-commit --> lint["make lint<br/>(parallel)"]:::parallel
    pre-commit --> docs-verify["make docs-verify<br/>(parallel)"]:::parallel
    pre-commit --> test-ci["make test-ci<br/>(parallel)"]:::parallel
    
    %% Lint parallelism
    lint --> lint-black["lint-black"]:::sequential
    lint --> lint-flake8["lint-flake8"]:::sequential
    lint --> lint-mypy["lint-mypy"]:::sequential
    lint --> lint-pylint["lint-pylint<br/>(parallel)"]:::parallel
    
    lint-pylint --> pylint-fc["pylint-firecrown"]:::sequential
    lint-pylint --> pylint-plug["pylint-plugins"]:::sequential
    lint-pylint --> pylint-test["pylint-tests"]:::sequential
    lint-pylint --> pylint-ex["pylint-examples"]:::sequential
    
    %% Test-CI parallelism (after unit-tests-pre barrier)
    test-ci --> barrier["🚧 unit-tests-pre<br/>(runs first)"]:::sequential
    barrier -.->|then parallel| test-all-cov["test-all-coverage<br/>(pytest -n auto)"]:::pytest
    barrier -.->|then parallel| test-slow["test-slow<br/>(pytest -n auto)"]:::pytest
    barrier -.->|then parallel| test-int["test-integration"]:::sequential
    barrier -.->|then parallel| test-ex["test-example"]:::sequential
    
    %% Test-all parallelism
    test-all["make test-all<br/>(parallel)"]:::parallel
    test-all --> test-run["test<br/>(pytest -n auto<br/>+coverage)"]:::pytest
    test-all --> test-slow
    test-all --> test-int
    test-all --> test-ex
    
    %% Unit tests post parallelism
    test-all-cov --> unit-post["unit-tests-post<br/>(parallel)"]:::parallel
    
    unit-post --> t-upd["test-updatable<br/>(100% cov)"]:::sequential
    unit-post --> t-util["test-utils<br/>(100% cov)"]:::sequential
    unit-post --> t-param["test-parameters<br/>(100% cov)"]:::sequential
    unit-post --> t-model["test-modeling-tools<br/>(100% cov)"]:::sequential
    unit-post --> t-clus["test-models-cluster<br/>(100% cov)"]:::sequential
    unit-post --> t-two["test-models-two-point<br/>(100% cov)"]:::sequential
    
    %% Docs verify parallelism
    docs-verify --> dg-map["docs-generate-symbol-map"]:::sequential
    docs-verify --> d-code["docs-code-check"]:::sequential
    docs-verify --> d-sym["docs-symbol-check"]:::sequential
    docs-verify --> d-link["docs-linkcheck"]:::sequential
    
    %% Legend
    subgraph legend["Legend"]
        leg-par["Parallel: subtargets run simultaneously"]:::parallel
        leg-seq["Sequential: single-threaded"]:::sequential
        leg-pytest["Pytest: uses pytest -n auto internally"]:::pytest
    end
```

**Key Points:**

- **Green boxes**: Run subtargets in parallel
- **Orange boxes**: Run sequentially (no internal parallelism)
- **Blue boxes**: Use pytest's `-n auto` for parallel test execution
- **Barrier (🚧)**: `unit-tests-pre` runs first, then its dependent targets run in parallel
- **Tutorials**: Specifically run sequentially due to Quarto's shared asset limitations

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
