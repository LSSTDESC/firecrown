"""Check for broken anchor links in HTML files.

This script uses Typer for a CLI and Rich for pretty terminal output.
"""

from typing import Annotated
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha1
import tempfile
import os
import shutil
import requests
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from bs4 import BeautifulSoup
import bs4


@dataclass
class PageAnchors:
    """Represent a page's URL, its path, and extracted ids.

    Ids can be either `id` or `name` attributes.
    """

    url_str: str
    path: Path
    ids: set[str]


def _parse_html(path: Path) -> BeautifulSoup:
    """Read and parse an HTML file into a BeautifulSoup object."""
    with open(path, "r", encoding="utf-8") as f:
        return BeautifulSoup(f, "html.parser")


def _extract_ids_from_soup(soup: BeautifulSoup) -> set[str]:
    """Return a set of id and name attributes found in the soup."""
    ids: list[str] = []
    for tag in soup.find_all(attrs={"id": True}):
        if tag is not None and isinstance(tag, bs4.Tag):
            val = tag.get("id")
            if isinstance(val, str):
                ids.append(val)
    for tag in soup.find_all(attrs={"name": True}):
        if tag is not None and isinstance(tag, bs4.Tag):
            val = tag.get("name")
            if isinstance(val, str):
                ids.append(val)
    return set(x for x in ids if x)


def extract_ids(file_path: Path) -> set[str]:
    """Extract IDs from an HTML file."""
    soup = _parse_html(file_path)
    return _extract_ids_from_soup(soup)


class SiteChecker:
    """Represent HTML pages and their extracted IDs.

    This class is a context manager to ensure the temporary download directory
    is removed reliably.
    """

    def __init__(
        self,
        root_dir: str | Path,
        console: Console,
        download_timeout: int,
        verbose: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.html_files: list[Path] = []
        self.targets: dict[str, PageAnchors] = {}
        self.downloaded_files: dict[str, Path] = {}
        self.tmp_root = Path(tempfile.mkdtemp(prefix="html_cache_"))
        self.valid_links: int = 0
        self.invalid_links: int = 0
        self.valid_anchors: int = 0
        self.invalid_anchors: int = 0
        self.console = console
        self.download_timeout = download_timeout
        self.verbose = bool(verbose)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

        self._collect_html_files()

    def _collect_html_files(self) -> None:
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(".html"):
                    full_path = Path(os.path.normpath(Path(dirpath) / fname))
                    self.html_files.append(full_path)
                    url_str = str(full_path)
                    page_anchors = PageAnchors(
                        url_str, full_path, extract_ids(full_path)
                    )
                    self.targets[url_str] = page_anchors

    def add_to_targets(self, url_str: str, full_path: Path) -> None:
        """Add a URL string and its corresponding full path to the targets dictionary."""
        if url_str not in self.targets:
            ids = set()
            if full_path.exists():
                ids = extract_ids(full_path)
            page_anchors = PageAnchors(url_str, full_path, ids)
            self.targets[url_str] = page_anchors

    def _download_url(self, url_str: str) -> Path:
        """Download a URL into a unique subdir under tmp_root and return the Path.

        Raises the underlying exception on failure.
        """
        url_hash = sha1(url_str.encode()).hexdigest()
        subdir = self.tmp_root / url_hash
        subdir.mkdir(parents=True, exist_ok=True)
        filename = Path(url_str.split("/")[-1] or "index.html")
        local_path = subdir / filename

        if local_path.exists():
            return local_path

        try:
            resp = self.session.get(url_str, timeout=self.download_timeout)
            resp.raise_for_status()
        except (requests.RequestException, OSError) as e:
            self.console.print(f"[red][Download failed][/red] {url_str}: {e}")
        else:
            local_path.write_bytes(resp.content)
            if self.verbose:
                self.console.print(
                    f"[green][Downloaded external][/green] "
                    f"{url_str} => {local_path}"
                )
        return local_path

    def _normalize_href(
        self, file_path: Path, href: str
    ) -> tuple[str, Path, str | None]:
        """Normalize an href into (target_path, fragment) or return None if it should
        be skipped.

        - If href contains a fragment (#frag) we return the fragment as the second element.
        - If href points to an http(s) resource we download it to tmp_root and return
          the local path.
        - If the href is purely an anchor ("#frag") the target path is the same file.
        - Returns None when the href should be ignored (empty fragment).
        """
        assert isinstance(href, str)
        url_str, frag = href.split("#", 1) if "#" in href else (href, None)

        # A pure `#frag` link points to the same file
        if not url_str:
            return (str(file_path), file_path, frag)

        # A link to an http(s) resource
        if url_str.startswith("http"):
            if url_str not in self.downloaded_files:
                path = self._download_url(url_str)
                self.downloaded_files[url_str] = path
            else:
                path = self.downloaded_files[url_str]

            return (url_str, path, frag)

        # A link to an anchor in a local file
        path = Path(os.path.normpath(file_path.parent / url_str))
        url_str = str(path)
        return (url_str, path, frag)

    def extract_links(self, file_path: Path) -> dict[str, set[str]]:
        """Extract anchor and non-anchor links from an HTML file.

        Returns a dictionary mapping URL strings to sets of fragments (or None).
        """
        soup = _parse_html(file_path)
        links: dict[str, set[str]] = {}
        for tag in soup.find_all("a", href=True):
            if tag is None or not isinstance(tag, bs4.Tag):
                continue
            href_val = tag.get("href")
            if not href_val:
                continue
            href = str(href_val)

            url_str, target_path, frag = self._normalize_href(file_path, href)
            self.add_to_targets(url_str, target_path)

            if url_str not in links:
                links[url_str] = set()

            if frag:
                links[url_str].add(frag)
        return links

    def check_anchors(self) -> list[tuple[str, str, str]]:
        """Check anchors and links in the collected HTML files.

        Returns a list of tuples: (source_file, target_file, fragment_or_None, reason)
        """
        missing_links: list[tuple[str, str, str]] = []

        for file_path in self.html_files:
            for url_str, anchors in self.extract_links(file_path).items():
                assert url_str in self.targets
                page_anchors = self.targets[url_str]

                if not page_anchors.path.exists():
                    if "http" in url_str:
                        missing_links.append(
                            (str(file_path), url_str, "unreachable link")
                        )
                    else:
                        missing_links.append((str(file_path), url_str, "file missing"))
                    self.invalid_links += 1
                    continue
                self.valid_links += 1

                if not anchors:
                    continue

                if page_anchors.ids is None:
                    missing_links.append((str(file_path), url_str, "no ids extracted"))
                    continue

                # All ids in anchors that are not in page_anchors.ids
                missing_ids = sorted(list(anchors - page_anchors.ids))
                self.invalid_anchors += len(missing_ids)
                self.valid_anchors += len(anchors) - len(missing_ids)
                if missing_ids:
                    missing_links.append(
                        (
                            str(file_path),
                            url_str,
                            f"missing ids: '{', '.join(missing_ids)}'",
                        )
                    )
                    continue

        return missing_links

    def close(self) -> None:
        """Close the temporary directory."""
        shutil.rmtree(self.tmp_root)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def main(
    root_dir: str | Path, download_timeout: int = 20, verbose: bool = False
) -> None:
    """Main function."""

    console = Console()

    with SiteChecker(
        root_dir, download_timeout=download_timeout, verbose=verbose, console=console
    ) as html_pages:
        missing_links = html_pages.check_anchors()

    if not missing_links:
        console.print(Panel("No broken links found", title="OK", style="green"))
        return

    # Now we print how many downloaded files there were, number of valid/invalid links
    # and anchors.
    console.print(
        Panel(
            f"Downloaded {len(html_pages.downloaded_files)} external files\n"
            f"Valid links: {html_pages.valid_links}\n"
            f"Invalid links: {html_pages.invalid_links}\n"
            f"Valid anchors: {html_pages.valid_anchors}\n"
            f"Invalid anchors: {html_pages.invalid_anchors}",
            title="Summary",
            style="blue",
        )
    )

    table = Table(title="Broken links")
    table.add_column("Source", style="cyan")
    table.add_column("Target", style="magenta")
    table.add_column("Reason", style="red")

    for src_file, target_file, reason in missing_links:
        table.add_row(str(src_file), str(target_file), reason)

    console.print(table)


app = typer.Typer(help="Check for broken anchor links in a directory of HTML files.")


@app.command()
def cli(
    root_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Directory with HTML files",
        ),
    ],
    download_timeout: Annotated[
        int,
        typer.Option(
            "-t",
            "--download-timeout",
            help="Timeout in seconds for downloading external links",
        ),
    ] = 20,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Enable verbose output (show downloads)"),
    ] = False,
):
    """Command-line entry point using Typer and Rich for output."""
    main(root_dir, download_timeout=download_timeout, verbose=verbose)


if __name__ == "__main__":
    app()
