# Firecrown Build System and CI Internals

This document describes the internal structure of the `Makefile` build system
and the CI pipeline.
It is intended for maintainers and contributors who need to understand or
modify these systems.
For general contribution guidance, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Target Relationships

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

## Parallelism Architecture

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

## CI System Architecture

Firecrown's CI is split across three files in `.github/`.
This structure keeps all job definitions in one place
and the list of supported long-lived branches in exactly one place,
while allowing both PR and nightly scheduled runs
to target multiple branches without any duplication.

| File | Purpose |
| :--- | :--- |
| `ci-branches.json` | **The single source of truth** for which long-lived branches are tested nightly. Edit only this file to add or remove a branch. |
| `workflows/ci-reusable.yml` | The single source of truth for all CI job definitions (all three stages). Called by the other two workflows. Never triggered directly by GitHub events. |
| `workflows/ci.yml` | Triggered on every `pull_request` event. Calls `ci-reusable.yml` unconditionally. There is no push trigger: all commits to long-lived branches arrive via merged PRs (which have already run CI), and the nightly workflow covers ongoing health checks. |
| `workflows/nightly.yml` | Triggered by the daily cron schedule. Reads `ci-branches.json` at runtime to build the branch matrix, then calls `ci-reusable.yml` once per branch. |

### How it works

`ci-reusable.yml` accepts an optional `ref` string input.
When `ref` is empty (the default), `actions/checkout` falls back to the commit
that triggered the calling workflow.
When `ref` is a branch name (as supplied by `nightly.yml`),
all checkout steps test that specific branch's code.

Because GitHub always executes scheduled workflows from the repository's
**default branch** (`master`), `nightly.yml` must explicitly supply `ref`
to test any branch other than `master`.
The reusable workflow definition used is always the one on `master`,
but the source code and `environment.yml` checked out during testing
come from the branch named in `ref`.

### Adding or removing a supported branch

Edit `.github/ci-branches.json` only — for example, to add `v1.15`:

```json
["master", "v1.14", "v1.15"]
```

No changes to any workflow file are needed.
