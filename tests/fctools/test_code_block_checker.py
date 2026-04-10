"""Tests for the code_block_checker module.

Test imports must be inside functions to avoid circular dependencies.
"""

from pathlib import Path
import tempfile

from typer.testing import CliRunner

from firecrown.fctools.code_block_checker import check_qmd_file, app

# Create CliRunner instance for testing
runner = CliRunner()


# Test fixtures - sample QMD content


def create_temp_qmd_file(content: str) -> Path:
    """Create a temporary .qmd file with the given content.

    :param content: The content to write to the file
    :return: Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".qmd", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(content)
        return Path(temp_file.name)


# Tests for check_qmd_file()


def test_check_qmd_file_no_code_blocks():
    """Test with a QMD file containing no code blocks."""
    content = """# Tutorial Title

This is just some markdown text.

No code blocks here!
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_valid_python_block():
    """Test with valid Python code block."""
    content = """# Tutorial

Here's some valid Python:

```{python}
x = 42
y = x + 1
print(y)
```

Done!
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_invalid_python_block():
    """Test with invalid Python code block."""
    content = """# Tutorial

This has a syntax error:

```{python}
x = 42
y = x +
print(y)
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 1
        assert "Block 1" in errors[0]
        assert "SyntaxError" in errors[0]
    finally:
        qmd_file.unlink()


def test_check_qmd_file_multiple_valid_blocks():
    """Test with multiple valid Python blocks."""
    content = """# Tutorial

First block:

```{python}
x = 1
```

Second block:

```{python}
y = 2
z = x + y
```

Third block:

```{python}
def foo():
    return 42
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_multiple_blocks_with_errors():
    """Test with multiple Python blocks where some have errors."""
    content = """# Tutorial

Valid block:

```{python}
x = 1
```

Invalid block:

```{python}
y = 2 +
```

Another valid block:

```{python}
z = 3
```

Another invalid block:

```{python}
def foo(
    return 42
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 2
        assert "Block 2" in errors[0]
        assert "Block 4" in errors[1]
    finally:
        qmd_file.unlink()


def test_check_qmd_file_non_python_code_blocks():
    """Test that non-Python code blocks are ignored."""
    content = """# Tutorial

This is R code (should be ignored):

```{r}
x <- 42
y <- x + 1
print(y)
```

This is bash (should be ignored):

```{bash}
echo "Hello"
invalid bash syntax here )(
```

This is Python (should be checked):

```{python}
x = 42
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_python_block_with_options():
    """Test Python block with execution options."""
    content = """# Tutorial

Block with options:

```{python, echo=false}
x = 42
y = x + 1
```

Another with multiple options:

```{python, eval=true, echo=false}
z = 100
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_empty_python_block():
    """Test with empty Python code block."""
    content = """# Tutorial

Empty block:

```{python}
```

More text.
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_python_block_with_comments():
    """Test Python block with comments."""
    content = """# Tutorial

```{python}
# This is a comment
x = 42  # inline comment
# Another comment
y = x + 1
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_display_only_python_block():
    """Test display-only Python block with .python format."""
    content = """# Tutorial

Display-only block (should still be checked):

```{.python}
x = 42
y = x + 1
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


def test_check_qmd_file_display_only_with_syntax_error():
    """Test display-only Python block with syntax error."""
    content = """# Tutorial

Display block with error:

```{.python}
x = 42
if x > 10
    print(x)
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 1
        assert "Block 1" in errors[0]
        assert "SyntaxError" in errors[0]
    finally:
        qmd_file.unlink()


def test_check_qmd_file_mixed_executable_and_display():
    """Test mix of executable and display-only blocks."""
    content = """# Tutorial

Executable block:

```{python}
x = 1
```

Display block:

```{.python}
y = 2
```

Another executable with error:

```{python}
z = 3 +
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 1
        assert "Block 3" in errors[0]
    finally:
        qmd_file.unlink()


def test_check_qmd_file_syntax_error_details():
    """Test that syntax error details are captured correctly."""
    content = """# Tutorial

```{python}
x = 42
if x > 10
    print(x)
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 1
        assert "Block 1" in errors[0]
        assert "SyntaxError" in errors[0]
        assert "line" in errors[0]
    finally:
        qmd_file.unlink()


def test_check_qmd_file_multiline_statements():
    """Test Python block with multiline statements."""
    content = """# Tutorial

```{python}
x = (
    1 + 2 + 3 +
    4 + 5 + 6
)

def long_function(
    arg1,
    arg2,
    arg3
):
    return arg1 + arg2 + arg3
```
"""
    qmd_file = create_temp_qmd_file(content)
    try:
        errors = check_qmd_file(qmd_file)
        assert len(errors) == 0
    finally:
        qmd_file.unlink()


# Tests for main()


def test_main_with_valid_directory():
    """Test main function with a directory containing valid QMD files."""
    # Create temporary directory with valid QMD file
    with tempfile.TemporaryDirectory() as temp_dir:
        qmd_path = Path(temp_dir) / "test.qmd"
        qmd_path.write_text(
            """# Test
```{python}
x = 42
```
""",
            encoding="utf-8",
        )

        # Run main via CliRunner
        result = runner.invoke(app, [temp_dir])

        assert result.exit_code == 0
        assert "No syntax errors" in result.stderr


def test_main_with_invalid_directory():
    """Test main function with non-existent directory."""
    result = runner.invoke(app, ["/nonexistent/directory"])

    assert result.exit_code != 0


def test_main_with_syntax_errors():
    """Test main function with QMD files containing syntax errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        qmd_path = Path(temp_dir) / "test.qmd"
        qmd_path.write_text(
            """# Test
```{python}
x = 42 +
```
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, [temp_dir])

        assert result.exit_code == 1
        assert "error" in result.stderr.lower()
        assert "Block 1" in result.stderr


def test_main_with_multiple_files():
    """Test main function with multiple QMD files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Valid file
        (Path(temp_dir) / "valid.qmd").write_text(
            """```{python}
x = 42
```""",
            encoding="utf-8",
        )

        # Invalid file
        (Path(temp_dir) / "invalid.qmd").write_text(
            """```{python}
y = 1 +
```""",
            encoding="utf-8",
        )

        result = runner.invoke(app, [temp_dir])

        assert result.exit_code == 1
        assert "invalid.qmd" in result.stderr
        assert "Total syntax errors" in result.stderr


def test_main_with_empty_directory():
    """Test main function with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, [temp_dir])

        assert result.exit_code == 0
        assert "Checking 0 .qmd files" in result.stderr
        assert "No syntax errors" in result.stderr


def test_main_quiet_mode_no_errors():
    """Test --quiet flag suppresses success messages."""
    with tempfile.TemporaryDirectory() as temp_dir:
        qmd_path = Path(temp_dir) / "valid.qmd"
        qmd_path.write_text(
            """```{python}
x = 42
```""",
            encoding="utf-8",
        )

        result = runner.invoke(app, [temp_dir, "--quiet"])

        assert result.exit_code == 0
        # Should have no output when quiet and no errors
        assert result.stderr.strip() == ""


def test_main_quiet_mode_with_errors():
    """Test --quiet flag still reports errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        qmd_path = Path(temp_dir) / "error.qmd"
        qmd_path.write_text(
            """```{python}
x = 1 +
```""",
            encoding="utf-8",
        )

        result = runner.invoke(app, [temp_dir, "--quiet"])

        assert result.exit_code == 1
        # Errors should still be shown even in quiet mode
        assert "error" in result.stderr.lower()
