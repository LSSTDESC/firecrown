"""Utilities for analysis data and file operations.

This is an internal module. Use the public API from firecrown.app.analysis.

File Download Caching
=====================

The download_from_url function implements an automatic caching mechanism to improve
performance and enable offline usage:

**Cache Location:**
    $HOME/.firecrown/sacc_files/

**Behavior:**
    1. First checks if the file exists in the cache directory
    2. If cached, copies from cache (no network request)
    3. If not cached, downloads from URL and saves to both cache and output location
    4. If download fails, provides instructions for manual file placement

**Benefits:**
    - Significantly faster test execution (avoids redundant downloads)
    - Enables offline work once files are cached
    - Reduces network load during development

**Manual Cache Management:**
    Users can manually place files in the cache directory to avoid downloads, or
    clear the cache by removing files from $HOME/.firecrown/sacc_files/
"""

import shutil
import urllib.request
from pathlib import Path
from types import ModuleType
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console


def get_cache_dir() -> Path:
    """Get the firecrown cache directory for SACC files.

    Creates $HOME/.firecrown/sacc_files if it doesn't exist.

    :return: Path to the cache directory
    """
    cache_dir = Path.home() / ".firecrown" / "sacc_files"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_filename(url: str) -> str:
    """Generate a cache filename from URL.

    Uses the last component of the URL path as the filename. Validates that the URL
    contains a proper filename component, not a bare domain, directory path, or POST
    endpoint.

    :param url: URL to generate filename from (e.g., 'https://example.com/data.fits')
    :return: Cache filename
    :raises AssertionError: If URL does not contain a valid filename
    """
    filename = url.rstrip("/").split("/")[-1]

    # Validate that we extracted a reasonable filename
    assert filename, (
        f"URL does not contain a valid filename: {url}\n"
        "URL must end with a filename component (e.g., 'data.fits'), "
        "not a bare domain or path ending with '/'"
    )

    # Check for suspicious patterns that might indicate URL parsing issues
    assert not filename.startswith(("http:", "https:")), (
        f"Invalid filename extracted from URL: {filename}\n"
        f"Original URL: {url}\n"
        "URL format may be malformed"
    )

    # Check that filename has an extension and doesn't look like a domain
    # Domain names typically have TLDs like .com, .org, but data files have
    # extensions like .fits, .hdf5, .yaml, .tar.gz
    parts = filename.split(".")
    if len(parts) >= 2:
        # Get the extension (everything after the first dot for simplicity)
        extension = parts[-1].lower()

        # Common TLDs that suggest this is a domain, not a file
        domain_tlds = {
            "com",
            "org",
            "net",
            "edu",
            "gov",
            "mil",
            "int",
            "io",
            "dev",
            "app",
            "cloud",
            "ai",
            "tech",
            "co",
            "uk",
        }

        assert extension not in domain_tlds, (
            f"URL appears to be a bare domain without a filename: {url}\n"
            f"Extracted: {filename}\n"
            "Expected URLs like 'https://example.com/data.fits', "
            "not 'https://example.com'"
        )
    else:
        # No extension at all
        raise AssertionError(
            f"Filename appears to lack an extension: {filename}\n"
            f"Original URL: {url}\n"
            "Expected URLs like 'https://example.com/data.fits', "
            "not bare paths or POST endpoints"
        )

    return filename


def download_from_url(
    url: str, output_file: Path, console: Console, description: str = "Downloading..."
) -> None:
    """Download file from URL with caching and progress indicator.

    This function implements a caching mechanism to avoid redundant downloads:

    1. Checks if the file exists in $HOME/.firecrown/sacc_files cache
    2. If cached, copies from cache to output location
    3. If not cached, attempts to download from URL
    4. On successful download, saves to both cache and output location
    5. If download fails, provides helpful instructions for manual placement

    This significantly improves performance in testing scenarios and handles
    offline usage gracefully.

    :param url: URL to download from
    :param output_file: Path where file will be saved
    :param console: Rich console for output
    :param description: Progress description text
    :raises SystemExit: If download fails and file is not in cache
    """
    cache_dir = get_cache_dir()
    cache_filename = get_cached_filename(url)
    cached_file = cache_dir / cache_filename

    # Check if file exists in cache
    if cached_file.exists():
        console.print(
            f"[dim]Using cached file: {cached_file.relative_to(Path.home())}[/dim]"
        )
        shutil.copyfile(cached_file, output_file)
        return

    # File not in cache, attempt download
    console.print(f"[dim]File not in cache, downloading from {url}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)

        try:
            # Download to cache first
            urllib.request.urlretrieve(url, cached_file)
            progress.update(task, completed=True)
            console.print(f"[green]✓[/green] Downloaded and cached to {cache_dir}")

            # Copy from cache to output location
            shutil.copyfile(cached_file, output_file)

        except Exception as e:
            console.print(f"[red]✗ Download failed:[/red] {e}")
            console.print()
            console.print("[yellow]Unable to download the required SACC file.[/yellow]")
            console.print()
            console.print("To proceed, please:")
            console.print(f"  1. Obtain the file manually from: [cyan]{url}[/cyan]")
            console.print(f"  2. Save it to: [cyan]{cached_file}[/cyan]")
            console.print(
                f"  3. Or place it in: [cyan]{cache_dir}[/cyan] with filename: "
                f"[cyan]{cache_filename}[/cyan]"
            )
            console.print()
            console.print(
                "[dim]The cache directory helps avoid repeated downloads in tests "
                "and enables offline usage.[/dim]"
            )
            raise SystemExit(1) from e


def copy_template(template_module: ModuleType, output_file: Path) -> None:
    """Copy template module file to output location.

    :param template_module: Python module to copy
    :param output_file: Destination path
    """
    assert template_module.__file__ is not None
    template = Path(template_module.__file__)
    shutil.copyfile(template, output_file)
