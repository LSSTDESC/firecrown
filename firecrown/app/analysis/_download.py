"""Utilities for analysis data and file operations.

This is an internal module. Use the public API from firecrown.app.analysis.
"""

import shutil
import urllib.request
from pathlib import Path
from types import ModuleType
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console


def download_from_url(
    url: str, output_file: Path, console: Console, description: str = "Downloading..."
) -> None:
    """Download file from URL with progress indicator.

    :param url: URL to download from
    :param output_file: Path where file will be saved
    :param console: Rich console for output
    :param description: Progress description text
    :raises: Exception if download fails
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)

        try:
            urllib.request.urlretrieve(url, output_file)
            progress.update(task, completed=True)
            console.print(f"[dim]Downloaded from {url}[/dim]")
        except Exception as e:
            console.print(f"[red]Download failed:[/red] {e}")
            raise


def copy_template(template_module: ModuleType, output_file: Path) -> None:
    """Copy template module file to output location.

    :param template_module: Python module to copy
    :param output_file: Destination path
    """
    assert template_module.__file__ is not None
    template = Path(template_module.__file__)
    shutil.copyfile(template, output_file)
