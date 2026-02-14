# Firecrown Makefile
# ===================
# Useful targets for testing, formatting, linting, and building documentation.
# Run 'make help' for a list of available targets.

.PHONY: help format lint typecheck test test-coverage test-integration test-slow \
	test-all clean clean-docs clean-coverage docs tutorials api-docs docs-build \
	lint-black lint-flake8 lint-pylint lint-pylint-firecrown lint-pylint-plugins \
	lint-pylint-tests lint-pylint-examples lint-mypy pre-commit install all-checks \
	check-env check-deps

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

# UI Colors and Prefixes
BOLD          := $(shell tput bold 2>/dev/null || echo "")
CYAN          := $(shell tput setaf 6 2>/dev/null || echo "")
GREEN         := $(shell tput setaf 2 2>/dev/null || echo "")
RED           := $(shell tput setaf 1 2>/dev/null || echo "")
RESET         := $(shell tput sgr0 2>/dev/null || echo "")

INFO_MSG      := @echo "${CYAN}${BOLD}[INFO]${RESET}"
OK_MSG        := @echo "${GREEN}${BOLD}[OK]  ${RESET}"
FAIL_MSG      := @echo "${RED}${BOLD}[FAIL]${RESET}"

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
	@echo "  make docs            - Build docs if you changed tutorials/docstrings"
	@echo ""
	@echo "Before pushing:"
	@echo "  make pre-commit      - Comprehensive check (format, lint, docs, full tests)"
	@echo "  make test-ci         - Run everything CI runs"
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
	@echo "  make test-ci         - Run the full CI suite (all tests, slow, examples)"
	@echo "  make docs            - Build and verify all documentation (tutorials + API)"
	@echo "  make pre-commit      - Comprehensive pre-push check (format, lint, docs, test-ci)"
	@echo ""
	@echo "Parallel execution:"
	@echo "  Parallel execution is ENABLED by default using $(JOBS) jobs."
	@echo "  Use 'make -j1 <target>' to run serially (e.g., for debugging)."
	@echo "  Use 'JOBS=N make <target>' to override the number of jobs."
	@echo ""

##@ Environment

check-env:  ## Verify that the environment is correct for development
	@if [ -n "$$SPACK_ENV" ]; then \
		echo "${RED}${BOLD}Error: A Spack environment is active ($$SPACK_ENV).${RESET}"; \
		echo "Firecrown development targets are not supported inside Spack environments"; \
		echo "due to potential conflicts with Conda or local dependencies."; \
		echo "Please deactivate the Spack environment and try again."; \
		exit 1; \
	fi
	$(OK_MSG) "Environment check passed (no Spack environment active)"

check-deps:  ## Verify that all required dependencies are installed correctly
	@echo "Verifying project dependencies..."
	@$(PYTHON) -m pip check || (echo "${RED}${BOLD}Error: Dependency check failed.${RESET}" && exit 1)
	$(OK_MSG) "Dependency check passed"

##@ Formatting

format:  ## Format code with black
	black $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/

format-check:  ## Check code formatting without modifying files
	black --check $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/

##@ Linting

lint: check-env check-deps lint-black lint-flake8 lint-mypy lint-pylint  ## Run all linting tools
	$(OK_MSG) "All linters passed!"

lint-black:  ## Check code formatting with black
	@echo "Running black..."
	@black --check $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ || (echo "${RED}${BOLD}❌ black failed${RESET}" && exit 1)
	$(OK_MSG) "black passed"

lint-flake8:  ## Run flake8 linter
	@echo "Running flake8..."
	@flake8 $(FIRECROWN_PKG_DIR)/ $(EXAMPLES_DIR)/ $(TESTS_DIR)/ || (echo "${RED}${BOLD}❌ flake8 failed${RESET}" && exit 1)
	$(OK_MSG) "flake8 passed"

lint-mypy:  ## Run mypy type checker
	@echo "Running mypy..."
	@mypy -p $(FIRECROWN_PKG_DIR) -p $(EXAMPLES_DIR) -p $(TESTS_DIR) || (echo "${RED}${BOLD}❌ mypy failed${RESET}" && exit 1)
	$(OK_MSG) "mypy passed"

lint-pylint: lint-pylint-firecrown lint-pylint-plugins lint-pylint-tests lint-pylint-examples ## Run all pylint checks
	$(OK_MSG) "All pylint checks passed!"

lint-pylint-firecrown:  ## Run pylint on firecrown package
	@echo "Running pylint on firecrown..."
	@pylint $(FIRECROWN_PKG_DIR) || (echo "${RED}${BOLD}❌ pylint failed for firecrown${RESET}" && exit 1)
	$(OK_MSG) "pylint passed for firecrown"

lint-pylint-plugins:  ## Run pylint on pylint_plugins
	@echo "Running pylint on pylint_plugins..."
	@pylint $(PYLINT_PLUGINS_DIR) || (echo "${RED}${BOLD}❌ pylint failed for pylint_plugins${RESET}" && exit 1)
	$(OK_MSG) "pylint passed for pylint_plugins"

lint-pylint-tests:  ## Run pylint on tests
	@echo "Running pylint on tests..."
	@pylint --rcfile $(TESTS_DIR)/pylintrc $(TESTS_DIR) || (echo "${RED}${BOLD}❌ pylint failed for tests${RESET}" && exit 1)
	$(OK_MSG) "pylint passed for tests"

lint-pylint-examples:  ## Run pylint on examples
	@echo "Running pylint on examples..."
	@pylint --rcfile $(EXAMPLES_DIR)/pylintrc $(EXAMPLES_DIR) || (echo "${RED}${BOLD}❌ pylint failed for examples${RESET}" && exit 1)
	$(OK_MSG) "pylint passed for examples"

typecheck: lint-mypy  ## Alias for mypy type checking

##@ Testing

test: check-env check-deps  ## Run tests in parallel (fast, no --runslow)
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

test-integration:  ## Run integration tests only
	$(PYTEST) -vv -s --integration -m integration tests/integration

test-all: test-slow test-integration test  ## Run all tests (slow + integration)


##@ Documentation

tutorials: ## Render all tutorials with quarto
	@mkdir -p $(TUTORIAL_OUTPUT_DIR)
	quarto render $(TUTORIAL_DIR) --output-dir=$(CURDIR)/$(TUTORIAL_OUTPUT_DIR) --to html --metadata "quarto-filters=[$(TUTORIAL_DIR)/linkgen.lua]"
	$(OK_MSG) "All tutorials rendered"

api-docs: ## Build API documentation with Sphinx
	@$(MAKE) -C $(DOCS_DIR) html

docs-build: tutorials api-docs  ## Build tutorials and API docs

docs: docs-build  ## Build and check all documentation


##@ Cleaning

clean-coverage:  ## Remove coverage reports
	$(RM) coverage.json coverage.xml .coverage .coverage.*
	$(RM) -r $(HTMLCOV_DIR)

clean-docs:  ## Remove built documentation
	$(RM) -r $(DOCS_BUILD_DIR)
	$(RM) -r $(TUTORIAL_OUTPUT_DIR)

clean-build:  ## Remove build artifacts
	$(RM) -r build/ dist/ *.egg-info/
	$(find) . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	$(find) . -type f -name "*.pyc" -delete
	$(find) . -type f -name "*.pyo" -delete

clean: clean-coverage clean-docs clean-build  ## Remove all generated files

##@ Pre-commit

pre-commit: check-env check-deps format lint test-ci ## Run all pre-commit checks
	@echo ""
	$(OK_MSG) "All pre-commit checks passed!"

all-checks: pre-commit test-slow test-integration ## Run everything

install: check-env check-deps  ## Install firecrown in development mode
	pip uninstall -y firecrown || true
	pip install --no-deps -e .

##@ Advanced

test-verbose:  ## Run tests with verbose output
	$(PYTEST) -vv -n auto

test-serial:  ## Run tests serially (no parallelization, useful for debugging)
	$(PYTEST) -vv

test-failfast:  ## Run tests and stop at first failure
	$(PYTEST) -x -n auto

test-ci: test test-slow test-integration ## Run exactly what CI runs
