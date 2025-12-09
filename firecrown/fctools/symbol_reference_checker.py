"""Validate Firecrown symbol references in Quarto (.qmd) files.

This tool extracts all inline code spans (backticks) from tutorial files and
validates that any Firecrown symbol references actually exist in the codebase.
It checks:
- Fully-qualified names (e.g., firecrown.likelihood.SourceSystematic)
- Partial module paths (e.g., firecrown.likelihood)
- Unqualified class/function names (e.g., Updatable)
"""

import re
import sys
from pathlib import Path

import typer
from rich.console import Console

from firecrown.fctools.docs_helpers import (
    load_json_file,
    print_error,
    print_success,
)


app = typer.Typer()


def extract_code_spans(content: str) -> list[tuple[int, str]]:
    """Extract all inline code spans from markdown content.

    :param content: The markdown content to parse
    :return: List of (line_number, code_text) tuples
    """
    results: list[tuple[int, str]] = []
    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        # Find all backtick-enclosed code spans
        # Match single backticks that are not part of triple backticks
        # Use a simple approach: find `...` patterns
        pattern = r"`([^`]+)`"
        matches = re.finditer(pattern, line)

        for match in matches:
            code_text = match.group(1)
            results.append((line_num, code_text))

    return results


def _load_external_symbols(external_symbols_file: Path | None) -> set[str]:
    """Load external symbols from a text file.

    :param external_symbols_file: Path to file with external symbols (one per line)
    :return: Set of external symbol names
    """
    if not external_symbols_file or not external_symbols_file.is_file():
        return set()

    external_symbols: set[str] = set()
    try:
        content = external_symbols_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            # Strip whitespace
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                external_symbols.add(line)
    except (OSError, UnicodeDecodeError) as e:
        print(
            f"Warning: Could not read external symbols file: {e}",
            file=sys.stderr,
        )

    return external_symbols


def _build_symbol_sets(
    symbol_map: dict[str, str], external_symbols: set[str]
) -> tuple[set[str], set[str]]:
    """Build sets of fully-qualified and unqualified symbols.

    :param symbol_map: Dictionary mapping valid symbols to their URLs
    :param external_symbols: Set of allowed external symbols
    :return: Tuple of (fully_qualified_symbols, unqualified_symbols)
    """
    fully_qualified_symbols = set(symbol_map.keys())
    unqualified_symbols = set()

    for symbol in fully_qualified_symbols:
        if "." in symbol:
            unqualified_name = symbol.split(".")[-1]
            unqualified_symbols.add(unqualified_name)

    # Add external symbols to unqualified set
    unqualified_symbols.update(external_symbols)

    return fully_qualified_symbols, unqualified_symbols


def _compile_exclude_pattern(exclude_pattern: str | None) -> re.Pattern | None:
    """Compile the exclude pattern regex if provided.

    :param exclude_pattern: Optional regex pattern string
    :return: Compiled regex pattern or None
    """
    if not exclude_pattern:
        return None

    try:
        return re.compile(exclude_pattern)
    except re.error as e:
        print(
            f"Warning: Invalid exclude pattern '{exclude_pattern}': {e}",
            file=sys.stderr,
        )
        return None


def _check_symbol(
    code_text: str,
    fully_qualified_symbols: set[str],
    unqualified_symbols: set[str],
    line_num: int,
) -> str | None:
    """Check if a code span is a valid symbol reference.

    :param code_text: The text from the code span
    :param fully_qualified_symbols: Set of valid fully-qualified symbols
    :param unqualified_symbols: Set of valid unqualified symbols
    :param line_num: Line number where the code span appears
    :return: Error message if invalid, None if valid
    """
    if code_text.startswith("firecrown."):
        if code_text not in fully_qualified_symbols:
            return (
                f"  Line {line_num}: Unknown symbol `{code_text}` "
                f"(not found in symbol map)"
            )
    elif "." not in code_text and code_text and code_text[0].isupper():
        if code_text not in unqualified_symbols:
            if code_text not in fully_qualified_symbols:
                return (
                    f"  Line {line_num}: Unknown symbol `{code_text}` "
                    f"(not found in symbol map)"
                )
    return None


def check_qmd_file(
    file_path: Path,
    symbol_map: dict[str, str],
    external_symbols: set[str] | None = None,
    exclude_pattern: str | None = None,
) -> list[str]:
    """Check a .qmd file for invalid Firecrown symbol references.

    :param file_path: Path to the Quarto (.qmd) file to check
    :param symbol_map: Dictionary mapping valid symbols to their URLs
    :param external_symbols: Set of allowed external symbols
    :param exclude_pattern: Optional regex pattern for symbols to exclude from checking
    :return: List of error messages for invalid symbol references
    """
    content: str = file_path.read_text(encoding="utf-8")
    code_spans: list[tuple[int, str]] = extract_code_spans(content)
    errors: list[str] = []

    if external_symbols is None:
        external_symbols = set()

    fully_qualified_symbols, unqualified_symbols = _build_symbol_sets(
        symbol_map, external_symbols
    )
    exclude_regex = _compile_exclude_pattern(exclude_pattern)

    for line_num, code_text in code_spans:
        if exclude_regex and exclude_regex.search(code_text):
            continue

        error = _check_symbol(
            code_text, fully_qualified_symbols, unqualified_symbols, line_num
        )
        if error:
            errors.append(error)

    return errors


def _print_summary(
    tutorial_dir: Path,
    symbol_map_path: Path,
    symbol_map: dict[str, str],
    external_symbols: set[str],
    exclude_pattern: str | None,
) -> None:
    """Print summary of what will be checked.

    :param tutorial_dir: Directory containing tutorial files
    :param symbol_map_path: Path to symbol map file
    :param symbol_map: The loaded symbol map
    :param external_symbols: Set of external symbols
    :param exclude_pattern: Optional exclude pattern
    """
    qmd_count = len(list(tutorial_dir.glob("*.qmd")))
    print(f"Checking {qmd_count} .qmd files in '{tutorial_dir}'...")
    print(f"Using symbol map from '{symbol_map_path}'")
    print(f"Symbol map contains {len(symbol_map)} symbols")

    if external_symbols:
        print(f"Loaded {len(external_symbols)} external symbols")

    if exclude_pattern:
        print(f"Excluding symbols matching: {exclude_pattern}")

    print()


@app.command()
def main(
    directory: Path = typer.Argument(
        ..., help="The directory containing .qmd files to check", exists=True
    ),
    symbol_map_file: Path = typer.Argument(
        ..., help="Path to the JSON file containing the symbol map", exists=True
    ),
    exclude_pattern: str | None = typer.Option(
        None, help="Regex pattern for symbols to exclude from checking"
    ),
    external_symbols_file: Path | None = typer.Option(
        None,
        help="Path to text file containing allowed external symbols (one per line)",
    ),
) -> None:
    """Validate Firecrown symbol references in Quarto (.qmd) files.

    Extracts all inline code spans (backticks) from tutorial files and
    validates that any Firecrown symbol references actually exist in the codebase.

    Checks:
    - Fully-qualified names (e.g., firecrown.likelihood.SourceSystematic)
    - Partial module paths (e.g., firecrown.likelihood)
    - Unqualified class/function names (e.g., Updatable)

    Exit codes:
        0: No invalid symbol references found
        1: Invalid symbols found or input error
    """
    console = Console(stderr=True)

    # Load the symbol map (typer validates exists=True)
    symbol_map: dict[str, str] = load_json_file(symbol_map_file)

    # Load external symbols if provided
    external_symbols = _load_external_symbols(external_symbols_file)

    # Find all .qmd files
    tutorial_dir = directory
    qmd_files: list[Path] = list(tutorial_dir.glob("*.qmd"))
    total_errors = 0
    files_with_errors = 0

    _print_summary(
        tutorial_dir,
        symbol_map_file,
        symbol_map,
        external_symbols,
        exclude_pattern,
    )

    for file in sorted(qmd_files):
        errors: list[str] = check_qmd_file(
            file, symbol_map, external_symbols, exclude_pattern
        )
        if errors:
            console.print(
                f"[bold red]Found {len(errors)} invalid symbol(s) in '{file.name}':[/]"
            )
            for error in errors:
                console.print(f"  [red]{error}[/]")
            console.print()
            total_errors += len(errors)
            files_with_errors += 1

    if total_errors > 0:
        print_error(
            f"Total: {total_errors} invalid symbol(s) in {files_with_errors} file(s)"
        )
        sys.exit(1)
    else:
        print_success("No invalid symbol references found in any tutorial files.")


if __name__ == "__main__":  # pragma: no cover
    app()
