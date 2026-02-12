# Firecrown Makefile
# ===================
# Useful targets for testing, formatting, linting, and building documentation.
# Run 'make help' for a list of available targets.

.PHONY: help format lint typecheck test test-coverage test-integration test-slow \
	test-all clean clean-docs clean-coverage docs tutorials api-docs docs-build \
	lint-black lint-flake8 lint-pylint lint-pylint-firecrown lint-pylint-plugins \
	lint-pylint-tests lint-pylint-examples lint-mypy pre-commit install all-checks \
	test-updatable test-utils test-parameters test-modeling-tools test-models \
	test-models-cluster test-models-two-point unit-tests

# Default target
.DEFAULT_GOAL := help

# Parallel execution configuration
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MAKEFLAGS += -j$(JOBS)

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

# Output directories
HTMLCOV_DIR := htmlcov
DOCS_BUILD_DIR := $(DOCS_DIR)/_build
AUTOAPI_BUILD_DIR := $(DOCS_DIR)/autoapi
# Tutorial configuration
TUTORIAL_OUTPUT_DIR := $(DOCS_DIR)/_static

# Test configuration
PYTEST_PARALLEL := $(PYTEST) -n auto
PYTEST_DURATIONS := --durations 10
PYTEST_COV_FLAGS := --cov $(FIRECROWN_PKG_DIR) --cov-report json --cov-report html --cov-report term-missing --cov-branch

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
	rm -f coverage.json
	rm -rf $(HTMLCOV_DIR)
	$(PYTEST_PARALLEL) $(PYTEST_DURATIONS) $(PYTEST_COV_FLAGS)
	@echo ""
	@echo "Coverage reports generated:"
	@echo "  - JSON: coverage.json"
	@echo "  - HTML: $(HTMLCOV_DIR)/index.html"
	@echo "  - Terminal output above"

test-slow:  ## Run only slow tests (with --runslow)
	$(PYTEST_PARALLEL) $(PYTEST_DURATIONS) -m slow --runslow $(TESTS_DIR)

test-example:  ## Run example tests only
	$(PYTEST) -vv -s --example -m example tests/example

test-integration:  ## Run integration tests only
	$(PYTEST) -vv -s -m integration tests/integration

test-all: test-slow test-example test-integration test  ## Run all tests (slow + example + integration)

unit-tests: unit-tests-post ## Run all unit tests in parallel
	@echo "✅ All unit tests passed!"

unit-tests-pre:
	@$(RM) .coverage.* .coverage

# Order-only prerequisite ensures unit-tests-pre runs before any test target
# but doesn't force a rebuild if it's already "complete".
test-updatable test-utils test-parameters test-modeling-tools test-models-cluster test-models-two-point: | unit-tests-pre

unit-tests-post: test-updatable test-utils test-parameters test-modeling-tools test-models-cluster test-models-two-point
	@echo "Combining coverage data..."
	@coverage combine
	@coverage report

test-updatable:  ## Run tests for firecrown.updatable module with coverage
	@COVERAGE_FILE=.coverage.updatable $(PYTEST) tests/test_updatable.py \
		tests/test_assert_updatable_interface.py \
		tests/test_updatable_parameters.py \
		--cov=firecrown.updatable \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-updatable failed" && exit 1)
	@echo "✅ test-updatable passed"

test-utils:  ## Run tests for firecrown.utils module with coverage
	@COVERAGE_FILE=.coverage.utils $(PYTEST) tests/test_utils.py \
		--cov=firecrown.utils \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-utils failed" && exit 1)
	@echo "✅ test-utils passed"

test-parameters:  ## Run tests for firecrown.parameters module with coverage
	@COVERAGE_FILE=.coverage.parameters $(PYTEST) tests/test_parameters_deprecated.py \
		--cov=firecrown.parameters \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-parameters failed" && exit 1)
	@echo "✅ test-parameters passed"

test-modeling-tools:  ## Run tests for firecrown.modeling_tools module with coverage
	@COVERAGE_FILE=.coverage.modeling-tools $(PYTEST) tests/test_modeling_tools.py \
		tests/test_modeling_tools_ccl_factory.py \
		--cov=firecrown.modeling_tools \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-modeling-tools failed" && exit 1)
	@echo "✅ test-modeling-tools passed"

test-models-cluster:  ## Run unit tests for firecrown.models.cluster package with coverage
	@COVERAGE_FILE=.coverage.models-cluster $(PYTEST) tests/models/cluster/ \
		--cov=firecrown.models.cluster \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100 || (echo "❌ test-models-cluster failed" && exit 1)
	@echo "✅ test-models-cluster passed"

test-models-two-point:  ## Run unit tests for firecrown.models.two_point package with coverage
	@COVERAGE_FILE=.coverage.models-two-point $(PYTEST) tests/models/two_point/ \
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
	$(RM) coverage.json coverage.xml .coverage .coverage.*
	$(RM) -r $(HTMLCOV_DIR)

clean-docs:  ## Remove built documentation
	$(RM) -r $(DOCS_BUILD_DIR)
	$(RM) -r $(TUTORIAL_OUTPUT_DIR)
	$(RM) -r $(AUTOAPI_BUILD_DIR)

clean-build:  ## Remove build artifacts
	$(RM) -r build/ dist/ *.egg-info/
	$(find) . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	$(find) . -type f -name "*.pyc" -delete
	$(find) . -type f -name "*.pyo" -delete

clean: clean-coverage clean-docs clean-build  ## Remove all generated files

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

test-ci: test-all-coverage test-slow test-integration test-example ## Run exactly what CI runs

test-all-coverage: unit-tests-pre unit-tests-core unit-tests-post ## Run core tests with coverage (fast)

unit-tests-core:  ## Internal target for core tests with coverage
	$(PYTEST) -vv --cov firecrown --cov-report xml --cov-branch -n auto
