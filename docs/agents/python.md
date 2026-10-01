# Python Code Style

These rules apply to Python code and tests in this repository. The second-brain
tool configuration implements the automated checks for that package. Pyright
remains configured for editor feedback.

## Required checks

- **Types:** Annotate functions and methods. Avoid `Any` in signatures. Run mypy
  on modified Python code.
- **Format and lint:** Run `black format` on modified Python files and `make lint`
  before finishing.
- **Docstrings:** Give every public module, class, function, method, and test a
  descriptive docstring. Use Sphinx-style `:param:` entries to describe each
  parameter and `:returns:` when a function returns a value. Omit
  `:returns: None` for procedures and tests. Give a private helper a docstring
  when its name and types do not explain its behavior, especially when it has side
  effects, relies on an invariant, or handles a subtle edge case. Keep useful
  existing private docstrings. Ruff checks public docstring presence; review
  checks private helpers and Sphinx fields.

## Testing

- **Avoid monkeypatching**: Prefer use of fixtures, parameterized tests, and
  dependency injection to the use of monkeypatching. Use monkeypatching only
  when other techniques would lead to more complicated code.  Any use of
  monkeypatching must contain a comment explaining why monkeypatching it the
  best choice for that test.

## Review guidance

- **File length:** Review files over 500 lines for separable responsibilities.
  A cohesive module may remain longer.
- **Comprehensions:** Prefer a comprehension for a simple transformation or
  filter that reads clearly in one expression. Use a loop for branching, state
  changes, early exits, or clearer diagnostics.