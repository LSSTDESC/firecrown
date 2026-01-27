"""Check Python code blocks in Quarto (.qmd) files for syntax errors."""

import ast
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()


def check_qmd_file(file_path: Path) -> list[str]:
    """Parse a .qmd file, extract Python code blocks, and check their syntax.

    :param file_path: Path to the Quarto (.qmd) file to check
    :return: List of error messages for code blocks with syntax errors
    """
    content: str = file_path.read_text(encoding="utf-8")
    code_blocks: list[str] = []

    extract_code_blocks(content, code_blocks)

    errors: list[str] = []
    for i, block in enumerate(code_blocks, 1):
        try:
            ast.parse(block)
        except SyntaxError as e:
            error_text = e.text.strip() if e.text else "(no text available)"
            errors.append(
                f"  - Block {i}: SyntaxError on line {e.lineno}: {error_text}"
            )

    return errors


def extract_code_blocks(content: str, code_blocks: list[str]) -> None:
    """Extract Python code blocks from Quarto markdown content.

    Parses content line by line to find code blocks delimited by {python}
    or {.python} (display-only), including blocks with execution options
    like {python, echo=false}. Appends the code from each Python block
    to the code_blocks list.

    :param content: The Quarto markdown content to parse
    :param code_blocks: List to append extracted code blocks to (modified in-place)
    """
    in_code_block: bool = False
    current_block: list[str] = []
    language: str = ""

    for line in content.splitlines():
        if line.strip().startswith("```{"):
            in_code_block = True
            # Extract language from ```{python} or ```{.python}
            # or ```{python, echo=false}
            # Strip the opening ```{ and closing }
            language_spec: str = line.strip()[4:-1].strip()

            # Handle optional dot prefix (.python for display-only blocks)
            if language_spec.startswith("."):
                language_spec = language_spec[1:]

            # Handle optional execution options after comma
            if "," in language_spec:
                language = language_spec.split(",")[0].strip()
            else:
                language = language_spec

            current_block = []
        elif line.strip() == "```" and in_code_block:
            in_code_block = False
            if language == "python" and current_block:
                code_blocks.append("\n".join(current_block))
            language = ""
        elif in_code_block:
            current_block.append(line)


@app.command()
def main(
    directory: Path = typer.Argument(
        ..., help="The directory containing .qmd files to check", exists=True
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only report errors, suppress success messages"
    ),
) -> None:
    """Check Python code blocks in Quarto (.qmd) files for syntax errors.

    Scans all .qmd files in the specified directory, extracts Python code blocks,
    and checks them for syntax errors using Python's ast.parse().

    Usage:
        code_block_checker.py tutorial/

    Exit codes:
        0: No errors found
        1: Syntax errors found or directory invalid
    """
    console = Console(stderr=True)
    tutorial_dir: Path = directory

    qmd_files: list[Path] = list(tutorial_dir.glob("*.qmd"))
    total_errors: int = 0

    if not quiet:
        console.print(
            f"[bold cyan]Checking {len(qmd_files)} .qmd files in '{tutorial_dir}'...[/]"
        )

    for file in qmd_files:
        errors: list[str] = check_qmd_file(file)
        if errors:
            console.print(f"\n[bold red]Found {len(errors)} error(s) in '{file}':[/]")
            for error in errors:
                console.print(f"  [red]{error}[/]")
            total_errors += len(errors)

    if total_errors > 0:
        console.print(f"\n[bold red]Total syntax errors: {total_errors}[/]")
        sys.exit(1)
    elif not quiet:
        console.print(
            "\n[bold green]✓ No syntax errors in any tutorial code blocks.[/]"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
