# Python Code Style

These rules apply to Python code and tests in this repository. The second-brain
tool configuration implements the automated checks for that package. Pyright
remains configured for editor feedback.

## Required checks

- **Tool environment:** Run Python checks from within an active, isolated project
  environment (e.g., `conda`, `venv`, or an agent sandbox). Verify that `python`,
  `mypy`, and `black` all resolve to this same environment before running the Make
  targets.
- **Types:** Annotate functions and methods. Avoid `Any` in signatures. Use
  `make typecheck` for type checking; it runs the project's mypy target.
- **Format and lint:** Format modified Python files with `black` from the
  project environment, then run `make lint` before finishing.
- **Docstrings:** Give every public module, class, function, method, and test a
  descriptive docstring. Use Sphinx-style `:param:` entries to describe each
  parameter and `:returns:` when a function returns a value. Omit
  `:returns: None` for procedures and tests. Give a private helper a docstring
  when its name and types do not explain its behavior, especially when it has side
  effects, relies on an invariant, or handles a subtle edge case. Keep useful
  existing private docstrings. Ruff checks public docstring presence; review
  checks private helpers and Sphinx fields.

## Testing

- **Test environment:** Run `pytest` and Make test targets within your active
  environment. NumCosmo tests write FFTW wisdom under `~/.numcosmo`, and download
  tests use `~/.firecrown`; ensure these directories are writable before running
  those tests in a sandbox. Treat a denied cache write as a sandbox failure, then
  rerun with scoped access before judging the test result.
- **Avoid monkeypatching:** Prefer use of fixtures, parameterized tests, and
  dependency injection to the use of monkeypatching. Use monkeypatching only
  when other techniques would lead to more complicated code. Any use of
  monkeypatching must contain a comment explaining why monkeypatching is the
  best choice for that test.

## Review guidance

- **File length:** Review files over 500 lines for separable responsibilities.
  A cohesive module may remain longer.
- **Comprehensions:** Prefer a comprehension for a simple transformation or
  filter that reads clearly in one expression. Use a loop for branching, state
  changes, early exits, or clearer diagnostics.
