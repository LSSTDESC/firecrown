# Firecrown Makefile
# ===================
# Useful targets for testing, formatting, linting, and building documentation.
# Run 'make help' for a list of available targets.

.PHONY: help format lint typecheck test test-coverage test-integration test-slow \
	test-all clean clean-docs clean-coverage docs tutorials api-docs docs-build \
	lint-black lint-flake8 lint-pylint lint-pylint-firecrown lint-pylint-plugins \
	lint-pylint-tests lint-pylint-examples lint-mypy pre-commit install all-checks \
	test-updatable test-utils test-parameters test-models test-models-cluster \
	test-models-two-point unit-tests

# Default target
.DEFAULT_GOAL := help

# Parallel execution configuration
JOBS ?= auto
ifeq ($(JOBS),auto)
	NPROCS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
	NPROCS := $(JOBS)
endif

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
TUTORIAL_OUTPUT_DIR := $(DOCS_DIR)/_static

# Test configuration
PYTEST := pytest
PYTEST_PARALLEL := $(PYTEST) -n auto
PYTEST_DURATIONS := --durations 10
PYTEST_COV_FLAGS := --cov $(FIRECROWN_PKG_DIR) --cov-report json --cov-report html --cov-report term-missing --cov-branch

help:  ## Show this help message
	@echo "Firecrown Makefile targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common workflows:"
	@echo "  make format          - Format all code with black"
	@echo "  make lint            - Run all linting tools in parallel"
	@echo "  make test            - Run tests in parallel (fast)"
	@echo "  make test-all        - Run all tests including slow tests"
	@echo "  make pre-commit      - Run all pre-commit checks (format, lint, test)"
	@echo "  make docs            - Build all documentation (tutorials + API)"
	@echo ""
	@echo "Parallel execution:"
	@echo "  make -j4 lint        - Run linting with 4 parallel jobs"
	@echo "  JOBS=8 make lint     - Set parallel jobs via environment variable"
	@echo "  Detected CPUs: $(NPROCS)"

##@ Formatting

format:  ## Format code with black
	black $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/

format-check:  ## Check code formatting without modifying files
	black --check $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/

##@ Linting

lint:  ## Run all linting tools in parallel
	@echo "Running all linters in parallel..."
	@(black --check $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ && echo "✅ black passed" || (echo "❌ black failed" && exit 1)) & \
	(flake8 $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ && echo "✅ flake8 passed" || (echo "❌ flake8 failed" && exit 1)) & \
	(mypy -p $(FIRECROWN_PKG_DIR) -p $(EXAMPLES_DIR) -p $(TESTS_DIR) && echo "✅ mypy passed" || (echo "❌ mypy failed" && exit 1)) & \
	($(MAKE) lint-pylint && echo "✅ pylint passed" || (echo "❌ pylint failed" && exit 1)) & \
	wait

lint-black:  ## Check code formatting with black
	@echo "Running black..."
	@black --check $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ || (echo "❌ black failed" && exit 1)

lint-flake8:  ## Run flake8 linter
	@echo "Running flake8..."
	@flake8 $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ || (echo "❌ flake8 failed" && exit 1)

lint-pylint:  ## Run pylint on all packages in parallel
	@echo "Running pylint on all packages in parallel..."
	@(pylint $(FIRECROWN_PKG_DIR) && echo "✅ pylint passed for firecrown" || (echo "❌ pylint failed for firecrown" && exit 1)) & \
	(pylint $(PYLINT_PLUGINS_DIR) && echo "✅ pylint passed for pylint_plugins" || (echo "❌ pylint failed for pylint_plugins" && exit 1)) & \
	(pylint --rcfile $(TESTS_DIR)/pylintrc $(TESTS_DIR) && echo "✅ pylint passed for tests" || (echo "❌ pylint failed for tests" && exit 1)) & \
	(pylint --rcfile $(EXAMPLES_DIR)/pylintrc $(EXAMPLES_DIR) && echo "✅ pylint passed for examples" || (echo "❌ pylint failed for examples" && exit 1)) & \
	wait

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

lint-mypy:  ## Run mypy type checker
	@echo "Running mypy..."
	@mypy -p $(FIRECROWN_PKG_DIR) -p $(EXAMPLES_DIR) -p $(TESTS_DIR) || (echo "❌ mypy failed" && exit 1)

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

test-updatable:  ## Run tests for firecrown.updatable module with coverage
	$(PYTEST) tests/test_updatable.py tests/test_assert_updatable_interface.py \
		--cov=firecrown.updatable \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100

test-utils:  ## Run tests for firecrown.utils module with coverage
	$(PYTEST) tests/test_utils.py \
		--cov=firecrown.utils \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100

test-parameters:  ## Run tests for firecrown.parameters module with coverage
	$(PYTEST) tests/test_parameters.py \
		--cov=firecrown.parameters \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100

test-models-cluster:  ## Run unit tests for firecrown.models.cluster package with coverage
	$(PYTEST) tests/models/cluster/ \
		--cov=firecrown.models.cluster \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100

test-models-two-point:  ## Run unit tests for firecrown.models.two_point package with coverage
	$(PYTEST) tests/models/two_point/ \
		--cov=firecrown.models.two_point \
		--cov-report=term-missing \
		--cov-branch \
		--cov-fail-under=100

unit-tests:  ## Run all unit tests in parallel
	@echo "Running unit tests in parallel..."
	@rm -f .coverage.* .coverage
	@(COVERAGE_FILE=.coverage.updatable $(MAKE) test-updatable && echo "✅ test-updatable passed" || (echo "❌ test-updatable failed" && exit 1)) & \
	PID1=$$! ; \
	(COVERAGE_FILE=.coverage.utils $(MAKE) test-utils && echo "✅ test-utils passed" || (echo "❌ test-utils failed" && exit 1)) & \
	PID2=$$! ; \
	(COVERAGE_FILE=.coverage.parameters $(MAKE) test-parameters && echo "✅ test-parameters passed" || (echo "❌ test-parameters failed" && exit 1)) & \
	PID3=$$! ; \
	(COVERAGE_FILE=.coverage.models-cluster $(MAKE) test-models-cluster && echo "✅ test-models-cluster passed" || (echo "❌ test-models-cluster failed" && exit 1)) & \
	PID4=$$! ; \
	(COVERAGE_FILE=.coverage.models-two-point $(MAKE) test-models-two-point && echo "✅ test-models-two-point passed" || (echo "❌ test-models-two-point failed" && exit 1)) & \
	PID5=$$! ; \
	FAILED=0 ; \
	wait $$PID1 || FAILED=1 ; \
	wait $$PID2 || FAILED=1 ; \
	wait $$PID3 || FAILED=1 ; \
	wait $$PID4 || FAILED=1 ; \
	wait $$PID5 || FAILED=1 ; \
	if [ $$FAILED -eq 1 ]; then \
		echo "❌ One or more unit test targets failed" ; \
		exit 1 ; \
	fi
	@echo "Combining coverage data..."
	@coverage combine
	@coverage report
	@echo "✅ All unit tests passed!"

##@ Documentation


##@ Documentation

tutorials:  ## Render tutorials with quarto
	quarto render $(TUTORIAL_DIR) --output-dir=../$(TUTORIAL_OUTPUT_DIR)

api-docs:  ## Build API documentation with Sphinx
	$(MAKE) -C $(DOCS_DIR) html

docs-build: tutorials api-docs  ## Build tutorials and API docs (can run in parallel with make -j2)

docs: docs-build docs-linkcheck ## Build and check all documentation (tutorials + API docs)
	@echo ""
	@echo "Documentation built successfully:"
	@echo "  - Tutorials: $(TUTORIAL_OUTPUT_DIR)/"
	@echo "  - API docs: $(DOCS_BUILD_DIR)/html/index.html"

docs-linkcheck:  ## Check documentation for broken links
	firecrown-link-checker $(DOCS_BUILD_DIR)/html -v

##@ Cleaning

clean-coverage:  ## Remove coverage reports
	rm -f coverage.json coverage.xml .coverage
	rm -rf $(HTMLCOV_DIR)

clean-docs:  ## Remove built documentation
	rm -rf $(DOCS_BUILD_DIR)
	rm -rf $(TUTORIAL_OUTPUT_DIR)
	rm -rf $(AUTOAPI_BUILD_DIR)

clean-build:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean: clean-coverage clean-docs clean-build  ## Remove all generated files (can run with make -j3 clean)

##@ Pre-commit

pre-commit: format lint test-coverage docs  ## Run all pre-commit checks (format, lint, test with coverage, docs)
	@echo ""
	@echo "✅ All pre-commit checks passed!"

all-checks: format lint test-coverage test-slow test-integration docs  ## Run everything: format, lint, all tests, and docs with optimal parallelism
	@echo ""
	@echo "✅ All checks passed! (format, lint, coverage, slow tests, integration tests, docs)"

install:  ## Install firecrown in development mode
	pip uninstall -y firecrown || true
	pip install --no-deps -e .

##@ Advanced

test-verbose:  ## Run tests with verbose output
	$(PYTEST_PARALLEL) -vv

test-serial:  ## Run tests serially (no parallelization, useful for debugging)
	$(PYTEST) -vv

test-failfast:  ## Run tests and stop at first failure
	$(PYTEST_PARALLEL) -x
