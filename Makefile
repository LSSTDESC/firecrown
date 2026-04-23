# Firecrown Makefile
# ===================
# Useful targets for testing, formatting, linting, and building documentation.
# Run 'make help' for a list of available targets.

SHELL := /bin/bash

.PHONY: help format lint typecheck test test-coverage test-example test-integration test-slow \
	test-all clean clean-docs clean-coverage docs tutorials api-docs docs-build \
	lint-black lint-flake8 lint-pylint lint-pylint-firecrown lint-pylint-plugins \
	lint-pylint-tests lint-pylint-examples lint-mypy pre-commit install all-checks \
	test-updatable test-utils test-parameters test-modeling-tools test-models \
	test-models-cluster test-models-two-point unit-tests test-ci test-all-coverage \
	unit-tests-pre unit-tests-post unit-tests-core docs-generate-symbol-map \
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
PYTHON := python3
PYTEST := pytest
RM := rm -f
find := find

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

# Test configuration
PYTEST_PARALLEL := $(PYTEST) -n auto
PYTEST_DURATIONS := --durations 10
PYTEST_COV_FLAGS := --cov $(FIRECROWN_PKG_DIR) --cov-report json:$(COVERAGE_JSON) --cov-report html:$(HTMLCOV_DIR) --cov-report term-missing --cov-branch

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

test-example:  ## No example tests on v1.14 (no-op)
	@echo "ℹ️  No example tests on v1.14 branch — skipping."

test-integration:  ## Run integration tests only
	$(PYTEST) -v -s -m integration tests/integration

test-all: test-slow test-example test-integration test  ## Run all tests (slow + example + integration)

unit-tests: unit-tests-post ## Run all unit tests in parallel
	@echo "✅ All unit tests passed!"

unit-tests-pre:
	@$(RM) .coverage.* .coverage

# Order-only prerequisite ensures unit-tests-pre runs before any test target
# but doesn't force a rebuild if it's already "complete".
unit-tests-core test-slow test-example test-integration: | unit-tests-pre

unit-tests-post:  ## No-op on v1.14 (coverage handled by unit-tests-core)
	@echo "ℹ️  unit-tests-post: no per-module coverage targets on v1.14."

##@ Documentation

docs-generate-symbol-map:  ## Generate the firecrown symbol-to-URL map for documentation
	@mkdir -p $(TUTORIAL_OUTPUT_DIR)
	@$(PYTHON) $(FIRECROWN_PKG_DIR)/fctools/generate_symbol_map.py > $(TUTORIAL_OUTPUT_DIR)/symbol_map.json

# Note: Building tutorials in parallel using 'make -j' with individual Rendering targets
# is unsafe because multiple Quarto processes compete for shared assets in 'site_libs',
# leading to race conditions and "No such file or directory" errors.
# We build the entire project in a single Quarto process for safety and reliability.
tutorials: docs-generate-symbol-map ## Render all tutorials with quarto (safe sequential build)
	quarto render $(TUTORIAL_DIR) --output-dir=$(CURDIR)/$(TUTORIAL_OUTPUT_DIR) --to html --metadata "quarto-filters=[$(TUTORIAL_DIR)/link_symbols.lua]"
	@echo "✅ All tutorials rendered"

api-docs: tutorials ## Build API documentation with Sphinx
	@$(MAKE) -C $(DOCS_DIR) html

docs-build: tutorials api-docs  ## Build tutorials and API docs

docs: docs-build docs-verify ## Build and check all documentation

docs-verify: docs-generate-symbol-map docs-code-check docs-symbol-check docs-linkcheck ## Run all documentation verification checks

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

##@ Advanced

test-verbose:  ## Run tests with verbose output
	$(PYTEST) -vv -n auto

test-serial:  ## Run tests serially (no parallelization, useful for debugging)
	$(PYTEST) -vv

test-failfast:  ## Run tests and stop at first failure
	$(PYTEST) -x -n auto

test-ci: unit-tests-pre test-all-coverage test-slow test-integration test-example ## Run exactly what CI runs

test-all-coverage: unit-tests-core unit-tests-post ## Run core tests with coverage (fast)

unit-tests-core:  ## Internal target for core tests with coverage
	$(PYTEST) -vv --cov firecrown --cov-report xml --cov-branch -n auto
