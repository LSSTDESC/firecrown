# Firecrown Makefile
# ===================
# Useful targets for testing, formatting, linting, and building documentation.
# Run 'make help' for a list of available targets.

SHELL := /bin/bash

.PHONY: help format lint typecheck test test-coverage test-example test-integration test-slow \
	test-all clean clean-docs clean-coverage docs tutorials api-docs docs-build \
	lint-black lint-flake8 lint-pylint lint-pylint-firecrown lint-pylint-plugins \
	lint-pylint-tests lint-pylint-examples lint-mypy pre-commit install all-checks \
	test-updatable test-utils test-parameters test-modeling-tools \
	test-models-cluster test-models-two-point unit-tests test-ci test-all-coverage \
	unit-tests-pre unit-tests-post unit-tests-core docs-generate-symbol-map \
	release-build-check release-gh-check conda-lock conda-lock-check \
	release-check release-tag release-sdist release-verify-sdist release-push \
	release-github \
	release-conda-forge \
	docs-verify docs-code-check docs-symbol-check docs-linkcheck

# Default target
.DEFAULT_GOAL := help

# Parallel execution configuration
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MAKEFLAGS += -j$(JOBS) --output-sync=target

# Ensure 'clean' targets run before any other targets on the same command line
# to avoid race conditions (e.g., 'make clean test -j').
ifneq ($(filter clean%,$(MAKECMDGOALS)),)
    $(filter-out clean%,$(MAKECMDGOALS)): | $(filter clean%,$(MAKECMDGOALS))
endif

# Tools
PYTHON ?= python
export FIRECROWN_VERSION := $(shell $(PYTHON) -c "import importlib.metadata; print(importlib.metadata.version('firecrown'))" 2>/dev/null || echo "dev")
PYTEST := pytest
RM := rm -f
BASH := bash
GH := gh
GH_HOST := github.com
BUILD := $(PYTHON) -m build

# Project directories
FIRECROWN_PKG_DIR := firecrown
TESTS_DIR := tests
EXAMPLES_DIR := examples
PYLINT_PLUGINS_DIR := pylint_plugins
DOCS_DIR := docs
TUTORIAL_DIR := tutorial

# Output configuration
COVERAGE_ID ?=
COVERAGE_JSON := coverage$(if $(COVERAGE_ID),_$(COVERAGE_ID),).json
HTMLCOV_DIR := htmlcov$(if $(COVERAGE_ID),_$(COVERAGE_ID),)
DOCS_BUILD_DIR := $(DOCS_DIR)/_build
UNIT_COVERAGE_DIR := covdata
UNIT_COVERAGE_UPDATABLE := $(UNIT_COVERAGE_DIR)/updatable
UNIT_COVERAGE_UTILS := $(UNIT_COVERAGE_DIR)/utils
UNIT_COVERAGE_PARAMETERS := $(UNIT_COVERAGE_DIR)/parameters
UNIT_COVERAGE_MODELING_TOOLS := $(UNIT_COVERAGE_DIR)/modeling-tools
UNIT_COVERAGE_MODELS_CLUSTER := $(UNIT_COVERAGE_DIR)/models-cluster
UNIT_COVERAGE_MODELS_TWO_POINT := $(UNIT_COVERAGE_DIR)/models-two-point
UNIT_COVERAGE_COMBINED := $(UNIT_COVERAGE_DIR)/combined
UNIT_COVERAGE_FILES := $(UNIT_COVERAGE_UPDATABLE) \
	$(UNIT_COVERAGE_UTILS) \
	$(UNIT_COVERAGE_PARAMETERS) \
	$(UNIT_COVERAGE_MODELING_TOOLS) \
	$(UNIT_COVERAGE_MODELS_CLUSTER) \
	$(UNIT_COVERAGE_MODELS_TWO_POINT)

# Patterns to preserve during 'make clean'
CLEAN_EXCLUDES := --exclude=.venv \
                  --exclude=venv \
                  --exclude=env \
                  --exclude=.env \
                  --exclude=.vscode \
                  --exclude=.agent \
                  --exclude=.amazonq
AUTOAPI_BUILD_DIR := $(DOCS_DIR)/autoapi
# Tutorial configuration
TUTORIAL_OUTPUT_DIR := $(DOCS_DIR)/_static
CONDA_LOCK_DIR := .github/conda-lock
CONDA_LOCK_SCRIPT := .github/scripts/generate_conda_locks.sh
GITHUB_RELEASE_REPO := LSSTDESC/firecrown
CONDA_FORGE_FEEDSTOCK_REPO := conda-forge/firecrown-feedstock
RELEASE_DIST_DIR := dist
RELEASE_SDIST := $(RELEASE_DIST_DIR)/firecrown-$(VERSION).tar.gz

# Test configuration
PYTEST_PARALLEL := $(PYTEST) -n auto
PYTEST_DURATIONS := --durations 10
PYTEST_COV_FLAGS := --cov $(FIRECROWN_PKG_DIR) --cov-report json:$(COVERAGE_JSON) --cov-report html:$(HTMLCOV_DIR) --cov-report term-missing --cov-branch

# These targets create shared temporary files and should always run serially.
.NOTPARALLEL: conda-lock conda-lock-check release-sdist release-verify-sdist

help:  ## Show common developer targets
	@echo "Firecrown Developer Quick Reference"
	@echo "===================================="
	@echo ""
	@echo "During development:"
	@echo "  make format          - Auto-format code (run frequently)"
	@echo "  make lint            - Check code quality (before commit)"
	@echo "  make test            - Run fast tests (during development)"
	@echo ""
	@echo "Before committing:"
	@echo "  make unit-tests      - Verify 100% coverage on changed modules"
	@echo "  make docs            - Build docs if you changed tutorials/docstrings"
	@echo "  make clean-docs      - Remove all generated tutorials and API docs"
	@echo ""
	@echo "Before pushing:"
	@echo "  make pre-commit      - Comprehensive check (format, lint, docs, full tests)"
	@echo "  make test-ci         - Run exactly what CI will run"
	@echo ""
	@echo "Release process:"
	@echo "  make release-check VERSION=x.y.z      - Validate release state"
	@echo "  make release-tag VERSION=x.y.z        - Create local release tag and .0 support branch"
	@echo "  make release-sdist VERSION=x.y.z      - Build the release sdist"
	@echo "  make release-verify-sdist VERSION=x.y.z - Verify the release sdist"
	@echo "  make release-push VERSION=x.y.z       - Push the verified tag and support branch"
	@echo "  make release-github VERSION=x.y.z     - Publish GitHub release and upload sdist"
	@echo "  make release-conda-forge VERSION=x.y.z - Start feedstock handoff"
	@echo ""
	@echo "Other useful targets:"
	@echo "  make help-all        - Show all available targets"
	@echo "  make clean           - Remove all generated files"
	@echo ""

help-all:  ## Show this help message
	@echo "Firecrown Makefile targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common workflows:"
	@echo "  make format          - Format all code with black"
	@echo "  make lint            - Run all linting tools (parallel by default)"
	@echo "  make test            - Run fast tests (parallel by default)"
	@echo "  make unit-tests      - Run all unit tests with 100% coverage check"
	@echo "  make test-ci         - Run the full CI suite (all tests, slow, examples)"
	@echo "  make docs            - Build and verify all documentation (tutorials + API)"
	@echo "  make pre-commit      - Comprehensive pre-push check (format, lint, docs, test-ci)"
	@echo "  make release-check VERSION=x.y.z      - Validate feature or maintenance release state"
	@echo "  make release-tag VERSION=x.y.z        - Create the local tag and any required support branch"
	@echo "  make release-sdist VERSION=x.y.z      - Build dist/firecrown-x.y.z.tar.gz from the tagged checkout"
	@echo "  make release-verify-sdist VERSION=x.y.z - Install the sdist into a temp target and verify version metadata"
	@echo "  make release-push VERSION=x.y.z       - Push the verified tag and any required support branch"
	@echo "  make release-github VERSION=x.y.z     - Create a GitHub release and upload the verified sdist"
	@echo "  make release-conda-forge VERSION=x.y.z - Create the conda-forge handoff issue"
	@echo ""
	@echo "Parallel execution:"
	@echo "  Parallel execution is ENABLED by default using $(JOBS) jobs."
	@echo "  Use 'make -j1 <target>' to run serially (e.g., for debugging)."
	@echo "  Use 'JOBS=N make <target>' to override the number of jobs."
	@echo ""

##@ Formatting

format:  ## Format code with black
	black $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/

format-check:  ## Check code formatting without modifying files
	black --check $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/

##@ Conda Lockfiles

conda-lock:  ## Generate committed conda lockfiles for CI matrix
	@$(BASH) $(CONDA_LOCK_SCRIPT)

conda-lock-check:  ## Verify generated lockfiles are up to date
	@$(BASH) $(CONDA_LOCK_SCRIPT)
	@git diff --exit-code -- $(CONDA_LOCK_DIR)

##@ Linting

lint: lint-black lint-flake8 lint-mypy lint-pylint  ## Run all linting tools
	@echo "✅ All linters passed!"

lint-black:  ## Check code formatting with black
	@echo "Running black..."
	@black --check $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ || (echo "❌ black failed" && exit 1)
	@echo "✅ black passed"

lint-flake8:  ## Run flake8 linter
	@echo "Running flake8..."
	@flake8 $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ || (echo "❌ flake8 failed" && exit 1)
	@echo "✅ flake8 passed"

lint-mypy:  ## Run mypy type checker
	@echo "Running mypy..."
	@mypy -p $(FIRECROWN_PKG_DIR) -p $(EXAMPLES_DIR) -p $(TESTS_DIR) || (echo "❌ mypy failed" && exit 1)
	@echo "✅ mypy passed"

lint-pylint: lint-pylint-firecrown lint-pylint-plugins lint-pylint-tests lint-pylint-examples ## Run all pylint checks
	@echo "✅ All pylint checks passed!"

lint-pylint-firecrown:  ## Run pylint on firecrown package
	@echo "Running pylint on firecrown..."
	@pylint $(FIRECROWN_PKG_DIR) || (echo "❌ pylint failed for firecrown" && exit 1)
	@echo "✅ pylint passed for firecrown"

lint-pylint-plugins:  ## Run pylint on pylint_plugins
	@echo "Running pylint on pylint_plugins..."
	@pylint $(PYLINT_PLUGINS_DIR) || (echo "❌ pylint failed for pylint_plugins" && exit 1)
	@echo "✅ pylint passed for pylint_plugins"

lint-pylint-tests:  ## Run pylint on tests
	@echo "Running pylint on tests..."
	@pylint --rcfile $(TESTS_DIR)/pylintrc $(TESTS_DIR) || (echo "❌ pylint failed for tests" && exit 1)
	@echo "✅ pylint passed for tests"

lint-pylint-examples:  ## Run pylint on examples
	@echo "Running pylint on examples..."
	@pylint --rcfile $(EXAMPLES_DIR)/pylintrc $(EXAMPLES_DIR) || (echo "❌ pylint failed for examples" && exit 1)
	@echo "✅ pylint passed for examples"

typecheck: lint-mypy  ## Alias for mypy type checking

##@ Testing

test:  ## Run tests in parallel (fast, no --runslow)
	$(PYTEST_PARALLEL) $(PYTEST_DURATIONS)

test-coverage:  ## Run tests with coverage reporting
	$(RM) $(COVERAGE_JSON)
	$(RM) -r $(HTMLCOV_DIR)
	$(PYTEST_PARALLEL) $(PYTEST_DURATIONS) $(PYTEST_COV_FLAGS)
	@echo ""
	@echo "Coverage reports generated:"
	@echo "  - JSON: coverage.json"
	@echo "  - HTML: $(HTMLCOV_DIR)/index.html"
	@echo "  - Terminal output above"

test-slow:  ## Run only slow tests (with --runslow)
	$(PYTEST_PARALLEL) $(PYTEST_DURATIONS) -m slow --runslow $(TESTS_DIR)

test-example:  ## Run example tests only
	@tmpfile=$$(mktemp /tmp/test-example.XXXXXX); \
	set -o pipefail; \
	if $(PYTEST) -v --example -m example tests/example 2>&1 | tee "$$tmpfile"; then \
		rm -f "$$tmpfile"; \
	else \
		echo ""; \
		echo "❌ test-example failed. Output saved to: $$tmpfile"; \
		exit 1; \
	fi

test-integration:  ## Run integration tests only
	$(PYTEST) -v -s -m integration tests/integration

test-all: test-slow test-example test-integration test  ## Run all tests (slow + example + integration)

unit-tests: unit-tests-post ## Run all unit tests in parallel
	@echo "✅ All unit tests passed!"

unit-tests-pre:
	@$(RM) .coverage.* .coverage
	@$(RM) -r $(UNIT_COVERAGE_DIR)
	@mkdir -p $(UNIT_COVERAGE_DIR)

# Order-only prerequisite ensures unit-tests-pre runs before any test target
# but doesn't force a rebuild if it's already "complete".
test-updatable test-utils test-parameters test-modeling-tools test-models-cluster \
test-models-two-point unit-tests-core test-slow test-example test-integration: | unit-tests-pre

unit-tests-post: test-updatable test-utils test-parameters test-modeling-tools test-models-cluster test-models-two-point
	@echo "Combining coverage data..."
	@COVERAGE_FILE=$(UNIT_COVERAGE_COMBINED) coverage combine $(UNIT_COVERAGE_FILES)
	@COVERAGE_FILE=$(UNIT_COVERAGE_COMBINED) coverage report

test-updatable:  ## Run tests for firecrown.updatable module with coverage
	@COVERAGE_FILE=$(UNIT_COVERAGE_UPDATABLE) $(PYTEST) tests/test_updatable.py \
		tests/test_assert_updatable_interface.py \
		tests/test_updatable_parameters.py \
		--cov=firecrown.updatable \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-updatable failed" && exit 1)
	@echo "✅ test-updatable passed"

test-utils:  ## Run tests for firecrown.utils module with coverage
	@COVERAGE_FILE=$(UNIT_COVERAGE_UTILS) $(PYTEST) tests/test_utils.py \
		--cov=firecrown.utils \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-utils failed" && exit 1)
	@echo "✅ test-utils passed"

test-parameters:  ## Run tests for firecrown.parameters module with coverage
	@COVERAGE_FILE=$(UNIT_COVERAGE_PARAMETERS) $(PYTEST) tests/test_parameters_deprecated.py \
		--cov=firecrown.parameters \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-parameters failed" && exit 1)
	@echo "✅ test-parameters passed"

test-modeling-tools:  ## Run tests for firecrown.modeling_tools module with coverage
	@COVERAGE_FILE=$(UNIT_COVERAGE_MODELING_TOOLS) $(PYTEST) tests/test_modeling_tools.py \
		tests/test_modeling_tools_ccl_factory.py \
		--cov=firecrown.modeling_tools \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-modeling-tools failed" && exit 1)
	@echo "✅ test-modeling-tools passed"

test-models-cluster:  ## Run unit tests for firecrown.models.cluster package with coverage
	@COVERAGE_FILE=$(UNIT_COVERAGE_MODELS_CLUSTER) $(PYTEST) tests/models/cluster/ \
		--cov=firecrown.models.cluster \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-models-cluster failed" && exit 1)
	@echo "✅ test-models-cluster passed"

test-models-two-point:  ## Run unit tests for firecrown.models.two_point package with coverage
	@COVERAGE_FILE=$(UNIT_COVERAGE_MODELS_TWO_POINT) $(PYTEST) tests/models/two_point/ \
		--cov=firecrown.models.two_point \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-models-two-point failed" && exit 1)
	@echo "✅ test-models-two-point passed"

##@ Documentation

docs-generate-symbol-map:  ## Generate the firecrown symbol-to-URL map for documentation
	@mkdir -p $(TUTORIAL_OUTPUT_DIR)
	@$(PYTHON) $(FIRECROWN_PKG_DIR)/fctools/generate_symbol_map.py > $(TUTORIAL_OUTPUT_DIR)/symbol_map.json

# Note: Building tutorials in parallel using 'make -j' with individual Rendering targets
# is unsafe because multiple Quarto processes compete for shared assets in 'site_libs',
# leading to race conditions and "No such file or directory" errors.
# We build the entire project in a single Quarto process for safety and reliability.
tutorials: docs-generate-symbol-map ## Render all tutorials with quarto (safe sequential build)
	quarto render $(TUTORIAL_DIR) --output-dir=$(CURDIR)/$(TUTORIAL_OUTPUT_DIR) --to html --metadata "firecrown-version=$(FIRECROWN_VERSION)" --metadata "quarto-filters=[$(TUTORIAL_DIR)/version_filter.lua,$(TUTORIAL_DIR)/link_symbols.lua]"
	@echo "✅ All tutorials rendered"

api-docs: tutorials ## Build API documentation with Sphinx
	@$(MAKE) -C $(DOCS_DIR) html

docs-build: api-docs  ## Build tutorials and API docs

docs: docs-verify ## Build and check all documentation

docs-verify: docs-code-check docs-symbol-check docs-linkcheck ## Run all documentation verification checks

docs-code-check: tutorials ## Check Python code blocks in .qmd files
	@echo "Checking tutorial code blocks for syntax errors..."
	@$(PYTHON) $(FIRECROWN_PKG_DIR)/fctools/code_block_checker.py $(TUTORIAL_DIR) || (echo "❌ docs-code-check failed" && exit 1)
	@echo "✅ docs-code-check passed"

docs-symbol-check: tutorials docs-generate-symbol-map ## Validate symbol references in .qmd files
	@echo "Validating Firecrown symbol references in tutorials..."
	@$(PYTHON) $(FIRECROWN_PKG_DIR)/fctools/symbol_reference_checker.py $(TUTORIAL_DIR) $(TUTORIAL_OUTPUT_DIR)/symbol_map.json --external-symbols-file $(TUTORIAL_DIR)/external_symbols.txt || (echo "❌ docs-symbol-check failed" && exit 1)
	@echo "✅ docs-symbol-check passed"

docs-linkcheck: docs-build ## Check documentation for broken links
	@echo "Checking for broken links..."
	@firecrown-link-checker $(DOCS_BUILD_DIR)/html -v || (echo "❌ docs-linkcheck failed" && exit 1)
	@echo "✅ docs-linkcheck passed"

##@ Cleaning

clean-coverage:  ## Remove coverage reports
	git clean -fdX $(CLEAN_EXCLUDES) -- coverage.json coverage.xml .coverage .coverage.* $(HTMLCOV_DIR)

clean-docs:  ## Remove built documentation
	git clean -fdX $(CLEAN_EXCLUDES) -- $(DOCS_BUILD_DIR) $(TUTORIAL_OUTPUT_DIR) $(AUTOAPI_BUILD_DIR)

clean-build:  ## Remove build artifacts
	git clean -fdX $(CLEAN_EXCLUDES) -- build/ dist/ *.egg-info/ firecrown/fctools/__pycache__ tests/__pycache__

clean:  ## Remove all generated files (using .gitignore as truth)
	git clean -fdX $(CLEAN_EXCLUDES)

##@ Pre-commit

pre-commit: format lint docs-verify test-ci ## Run all pre-commit checks
	@echo ""
	@echo "✅ All pre-commit checks passed!"

all-checks: pre-commit test-slow test-integration ## Run everything

install:  ## Install firecrown in development mode
	pip uninstall -y firecrown || true
	pip install --no-deps -e .

##@ Release

release-build-check:  ## Verify that the Python build frontend is installed
	@set -euo pipefail; \
	if ! $(BUILD) --version >/dev/null 2>&1; then \
		echo "Python package 'build' is required for release artifact targets."; \
		echo "Install it in the active environment with: python -m pip install build"; \
		exit 1; \
	fi

release-gh-check:  ## Verify that GitHub CLI is installed and authenticated
	@set -euo pipefail; \
	if ! command -v $(GH) >/dev/null 2>&1; then \
		echo "GitHub CLI 'gh' is required for release targets."; \
		echo "Install it first, for example with: brew install gh"; \
		echo "Then log in with: gh auth login --hostname $(GH_HOST) --web"; \
		exit 1; \
	fi; \
	if ! $(GH) auth status --hostname $(GH_HOST) >/dev/null 2>&1; then \
		echo "GitHub CLI is installed but not authenticated for $(GH_HOST)."; \
		echo "Log in with: gh auth login --hostname $(GH_HOST) --web"; \
		echo "Then verify with: gh auth status --hostname $(GH_HOST)"; \
		exit 1; \
	fi

release-check: release-build-check release-gh-check ## Validate the checkout for release VERSION=x.y.z
	@set -euo pipefail; \
	if [[ -z "$(VERSION)" ]]; then \
		echo "VERSION is required. Use: make $@ VERSION=x.y.z"; \
		exit 1; \
	fi; \
	if [[ ! "$(VERSION)" =~ ^[0-9]+\.[0-9]+\.[0-9]+$$ ]]; then \
		echo "VERSION must have the form x.y.z"; \
		exit 1; \
	fi; \
	if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "Release checkout must be clean."; \
		exit 1; \
	fi; \
	if git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then \
		echo "Tag v$(VERSION) already exists locally."; \
		exit 1; \
	fi; \
	if git remote get-url origin >/dev/null 2>&1 && \
		git ls-remote --exit-code --tags origin "refs/tags/v$(VERSION)" >/dev/null 2>&1; then \
		echo "Tag v$(VERSION) already exists on origin."; \
		exit 1; \
	fi; \
	IFS=. read -r major minor patch <<< "$(VERSION)"; \
	support_branch="v$${major}_$${minor}_support"; \
	if [[ "$$patch" == "0" ]]; then \
		if git rev-parse -q --verify "refs/heads/$$support_branch" >/dev/null; then \
			echo "Support branch $$support_branch already exists locally."; \
			exit 1; \
		fi; \
		if git remote get-url origin >/dev/null 2>&1 && \
			git ls-remote --exit-code --heads origin "$$support_branch" >/dev/null 2>&1; then \
			echo "Support branch $$support_branch already exists on origin."; \
			exit 1; \
		fi; \
	else \
		current_branch="$$(git branch --show-current)"; \
		if [[ "$$current_branch" != "$$support_branch" ]]; then \
			echo "Maintenance releases for v$(VERSION) must be created from $$support_branch."; \
			exit 1; \
		fi; \
		if git remote get-url origin >/dev/null 2>&1 && \
			! git ls-remote --exit-code --heads origin "$$support_branch" >/dev/null 2>&1; then \
			echo "Support branch $$support_branch was not found on origin."; \
			exit 1; \
		fi; \
	fi; \
	$(MAKE) pre-commit conda-lock-check; \
	echo "✅ Release checks passed for v$(VERSION)"

	release-tag:  ## Create local tag, plus .0 support branch VERSION=x.y.z
	@set -euo pipefail; \
	$(MAKE) release-check VERSION=$(VERSION); \
	IFS=. read -r major minor patch <<< "$(VERSION)"; \
	if [[ "$$patch" == "0" ]]; then \
		support_branch="v$${major}_$${minor}_support"; \
	fi; \
	git tag -a "v$(VERSION)" -m "Release $(VERSION)"; \
	if [[ "$$patch" == "0" ]]; then \
		git branch "$$support_branch" HEAD; \
		echo "✅ Created local v$(VERSION) and $$support_branch"; \
	else \
		echo "✅ Created local v$(VERSION)"; \
	fi

release-sdist: release-build-check ## Build the release sdist VERSION=x.y.z
	@set -euo pipefail; \
	if [[ -z "$(VERSION)" ]]; then \
		echo "VERSION is required. Use: make $@ VERSION=x.y.z"; \
		exit 1; \
	fi; \
	if ! git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then \
		echo "Local tag v$(VERSION) was not found. Run: make release-tag VERSION=$(VERSION)"; \
		exit 1; \
	fi; \
	tag_commit="$$(git rev-list -n 1 "v$(VERSION)")"; \
	head_commit="$$(git rev-parse HEAD)"; \
	if [[ "$$head_commit" != "$$tag_commit" ]]; then \
		echo "HEAD does not match local tag v$(VERSION). Check out the tagged release commit before building the sdist."; \
		exit 1; \
	fi; \
	rm -rf "$(RELEASE_DIST_DIR)"; \
	$(BUILD) --sdist --outdir "$(RELEASE_DIST_DIR)"; \
	if [[ ! -f "$(RELEASE_SDIST)" ]]; then \
		echo "Expected sdist was not created: $(RELEASE_SDIST)"; \
		exit 1; \
	fi; \
	echo "✅ Built $(RELEASE_SDIST)"

release-verify-sdist: release-sdist ## Verify the release sdist VERSION=x.y.z
	@set -euo pipefail; \
	if [[ -z "$(VERSION)" ]]; then \
		echo "VERSION is required. Use: make $@ VERSION=x.y.z"; \
		exit 1; \
	fi; \
	tmpdir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmpdir"' EXIT; \
	target_dir="$$tmpdir/site"; \
	mkdir -p "$$target_dir"; \
	$(PYTHON) -m pip install --no-deps --target "$$target_dir" "$(RELEASE_SDIST)" >/dev/null; \
	(
		cd "$$tmpdir"; \
		PYTHONPATH="$$target_dir" $(PYTHON) -c "import importlib.metadata; import firecrown; expected='$(VERSION)'; assert importlib.metadata.version('firecrown') == expected, importlib.metadata.version('firecrown'); assert firecrown.__version__ == expected, firecrown.__version__"
	); \
	echo "✅ Verified $(RELEASE_SDIST)"

release-push:  ## Push the verified tag, plus .0 support branch VERSION=x.y.z
	@set -euo pipefail; \
	$(MAKE) release-verify-sdist VERSION=$(VERSION); \
	if [[ -z "$(VERSION)" ]]; then \
		echo "VERSION is required. Use: make $@ VERSION=x.y.z"; \
		exit 1; \
	fi; \
	if ! git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then \
		echo "Local tag v$(VERSION) was not found. Run: make release-tag VERSION=$(VERSION)"; \
		exit 1; \
	fi; \
	IFS=. read -r major minor patch <<< "$(VERSION)"; \
	if [[ "$$patch" == "0" ]]; then \
		support_branch="v$${major}_$${minor}_support"; \
		if ! git rev-parse -q --verify "refs/heads/$$support_branch" >/dev/null; then \
			echo "Local support branch $$support_branch was not found. Run: make release-tag VERSION=$(VERSION)"; \
			exit 1; \
		fi; \
		git push origin "v$(VERSION)" "$$support_branch"; \
		echo "✅ Pushed v$(VERSION) and $$support_branch"; \
	else \
		git push origin "v$(VERSION)"; \
		echo "✅ Pushed v$(VERSION)"; \
	fi

release-github: release-gh-check ## Create GitHub release VERSION=x.y.z
	@set -euo pipefail; \
	if [[ -z "$(VERSION)" ]]; then \
		echo "VERSION is required. Use: make $@ VERSION=x.y.z"; \
		exit 1; \
	fi; \
	if [[ ! -f "$(RELEASE_SDIST)" ]]; then \
		echo "Release sdist was not found: $(RELEASE_SDIST)"; \
		echo "Run: make release-verify-sdist VERSION=$(VERSION)"; \
		exit 1; \
	fi; \
	if ! git remote get-url origin >/dev/null 2>&1 || \
		! git ls-remote --exit-code --tags origin "refs/tags/v$(VERSION)" >/dev/null 2>&1; then \
		echo "Remote tag v$(VERSION) was not found on origin."; \
		echo "Run: make release-push VERSION=$(VERSION)"; \
		exit 1; \
	fi; \
	latest_version="$$( { git tag -l 'v[0-9]*.[0-9]*.[0-9]*' | sed 's/^v//'; echo '$(VERSION)'; } | sort -uV | tail -n 1 )"; \
	if [[ "$$latest_version" == "$(VERSION)" ]]; then \
		latest_flag="--latest"; \
	else \
		latest_flag="--latest=false"; \
	fi; \
	$(GH) release create "v$(VERSION)" --repo "$(GITHUB_RELEASE_REPO)" --verify-tag --generate-notes $$latest_flag "$(RELEASE_SDIST)"

release-conda-forge: release-gh-check ## Create conda-forge handoff issue for VERSION=x.y.z
	@set -euo pipefail; \
	if [[ -z "$(VERSION)" ]]; then \
		echo "VERSION is required. Use: make $@ VERSION=x.y.z"; \
		exit 1; \
	fi; \
	$(GH) issue create --repo "$(CONDA_FORGE_FEEDSTOCK_REPO)" \
		--title "@conda-forge-admin, please update version" \
		--body "Please update firecrown to v$(VERSION)."

##@ Advanced

test-verbose:  ## Run tests with verbose output
	$(PYTEST) -vv -n auto

test-serial:  ## Run tests serially (no parallelization, useful for debugging)
	$(PYTEST) -vv

test-failfast:  ## Run tests and stop at first failure
	$(PYTEST) -x -n auto

test-ci: test-all-coverage test-slow test-integration test-example ## Run exactly what CI runs

test-all-coverage: unit-tests-core unit-tests-post ## Run core tests with coverage (fast)

unit-tests-core:  ## Internal target for core tests with coverage
	$(PYTEST) -vv --cov firecrown --cov-report xml --cov-branch -n auto
