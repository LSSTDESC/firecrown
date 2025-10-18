"""Unit tests for link checker module.

Tests for the link checker functionality in the
``firecrown.fctools.link_checker`` module.

These tests avoid real network access by mocking the requests.Session.get
method when necessary.
"""

from pathlib import Path
import requests

import pytest

from rich.console import Console
from typer.testing import CliRunner

from firecrown.fctools import link_checker


@pytest.fixture(name="site_dir")
def fixture_site_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory to be used as the site root for tests.

    Using a fixture makes it easy to change setup globally and reuse the same
    directory name across tests.
    """
    return tmp_path


@pytest.fixture(name="sample_site")
def fixture_sample_site(site_dir: Path) -> Path:
    """Create a small set of HTML files under the provided site_dir.

    Files created:
    - page.html (contains id p1)
    - same.html (contains id localfrag)
    - sub/inside.html (contains id subid)
    """
    (site_dir / "page.html").write_text('<html><body><div id="p1"></div></body></html>')
    (site_dir / "same.html").write_text(
        '<html><body><a id="localfrag"></a></body></html>'
    )
    subdir = site_dir / "sub"
    subdir.mkdir()
    (subdir / "inside.html").write_text(
        '<html><body><div id="subid"></div></body></html>'
    )
    return site_dir


@pytest.fixture(name="mock_requests_ok")
def fixture_mock_requests_ok(monkeypatch):
    """Mock ``requests.Session.get``.

    Mock to return a simple HTML page containing id 'extid'.
    """

    class FakeResp:
        """Fake response object for requests.get()."""

        def __init__(self, content: str):
            """Initialize with given content string."""
            self.content = content.encode("utf-8")

        def raise_for_status(self):
            """Simulate a successful response."""
            return None

    def _fake_get(_self, _url, timeout=None, **_kwargs):
        assert timeout is not None
        return FakeResp('<html><body><div id="extid"></div></body></html>')

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    return _fake_get


def test_extract_ids_basic(site_dir: Path):
    """Test extraction of ids from a simple HTML file."""
    html = site_dir / "page.html"
    html.write_text(
        """
    <html><body>
      <div id="one"></div>
      <a name="two"></a>
    </body></html>
    """,
    )

    ids = link_checker.extract_ids(html)
    assert ids == {"one", "two"}


def test_check_anchors_local_ok(site_dir: Path):
    # target page with an id
    b = site_dir / "b.html"
    b.write_text("<html><body><h1 id=sec>Section</h1></body></html>")

    # source page linking to target#sec
    a = site_dir / "a.html"
    a.write_text('<a href="b.html#sec">to sec</a>')

    console = Console(record=True)
    sc = link_checker.SiteChecker(
        site_dir, console=console, download_timeout=1, verbose=False, skip_external=True
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    assert not missing
    assert sc.valid_links == 1
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 1
    assert sc.invalid_anchors == 0


def test_check_anchors_missing_anchor_returns_failure_and_main_exit(site_dir: Path):
    # target without the requested id
    b = site_dir / "b.html"
    b.write_text("<html><body><h1>Section</h1></body></html>")

    a = site_dir / "a.html"
    a.write_text('<a href="b.html#missing">broken</a>')

    # main() should return non-zero because there is a missing anchor
    code = link_checker.main(
        site_dir, download_timeout=1, verbose=False, skip_external=True
    )
    assert code != 0


def test_skip_external_prevents_download_and_allows_success(
    site_dir: Path, monkeypatch
):
    # source linking to an external URL with fragment
    a = site_dir / "a.html"
    a.write_text('<a href="http://example.invalid/page.html#frag">ext</a>')

    # Simulate requests raising if called (network would fail)
    def _raise_get(self, url, timeout=None):
        raise requests.RequestException("no network in tests")

    monkeypatch.setattr(requests.Session, "get", _raise_get)

    # When skipping external links, the run should succeed because externals are not
    # validated.
    code = link_checker.main(
        site_dir, download_timeout=1, verbose=False, skip_external=True
    )
    assert code == 0

    # When not skipping externals, the SiteChecker will try to download and fail ->
    # return non-zero.
    code2 = link_checker.main(
        site_dir, download_timeout=1, verbose=False, skip_external=False
    )
    assert code2 != 0


@pytest.mark.parametrize(
    "fname, content, expected",
    [
        ("s1.html", '<a href="page.html#p1">ok</a>', 0),
        (
            "s2.html",
            '<div id="localfrag"></div><a href="#localfrag">same file</a>',
            0,
        ),
        ("s3.html", '<a href="missing.html#x">bad</a>', 1),
        ("s4.html", '<a href="sub/inside.html#subid">sub</a>', 0),
    ],
    ids=[
        "ok",
        "same file",
        "bad",
        "sub",
    ],
)
def test_parametrized_link_scenarios(
    sample_site: Path, fname: str, content: str, expected: int
):
    (sample_site / fname).write_text(content)
    code = link_checker.main(
        sample_site, download_timeout=1, verbose=False, skip_external=True
    )
    assert code == expected


def test_external_download_with_mock(sample_site: Path, mock_requests_ok):
    # external link to page with id extid; mocked response supplies that id
    assert mock_requests_ok is not None  # to avoid unused variable warning
    (sample_site / "ext.html").write_text(
        '<a href="http://example.invalid/page.html#extid">ext</a>'
    )
    code = link_checker.main(
        sample_site, download_timeout=1, verbose=False, skip_external=False
    )
    assert code == 0


def test_external_download_fails_when_get_raises(sample_site: Path, monkeypatch):
    (sample_site / "ext2.html").write_text(
        '<a href="http://example.invalid/page.html#extid">ext</a>'
    )

    def _raise(self, url, timeout=None):
        raise requests.RequestException("simulated network error")

    monkeypatch.setattr(requests.Session, "get", _raise)
    code = link_checker.main(
        sample_site, download_timeout=1, verbose=False, skip_external=False
    )
    assert code != 0


def test_download_url_writes_file_when_ok(sample_site: Path, mock_requests_ok):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    console = Console(record=True)
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=True,
        skip_external=False,
    )
    try:
        # pylint: disable-next=protected-access
        path = sc._download_url("http://example.invalid/page.html")
        # file should be present under tmp cache
        assert path.exists()
        content = path.read_bytes().decode("utf-8")
        assert "extid" in content
    finally:
        sc.close()


def test_download_url_handles_failure_and_returns_path(sample_site: Path, monkeypatch):
    console = Console(record=True)

    def _raise(self, url, timeout=None, **_kwargs):
        raise requests.RequestException("network boom")

    monkeypatch.setattr(requests.Session, "get", _raise)

    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=True,
        skip_external=False,
    )
    try:
        # pylint: disable-next=protected-access
        path = sc._download_url("http://example.invalid/page.html")
        # path is returned but file should not exist because download failed
        assert not path.exists()
        # console should have a download-failed message
        out = console.export_text()
        assert "Download failed" in out
    finally:
        sc.close()


def test_normalize_fragment_only(sample_site: Path):
    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        fp = sample_site / "page.html"
        fp.write_text('<a id="x"></a>')
        # pylint: disable-next=protected-access
        url_str, path, frag = sc._normalize_href(fp, "#x")
        assert url_str == str(fp)
        assert path == fp
        assert frag == "x"
    finally:
        sc.close()


def test_missing_file_reports_file_missing(sample_site: Path):
    # source linking to a missing file
    (sample_site / "src.html").write_text('<a href="nope.html">bad</a>')
    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()
    assert missing
    # reason for the first missing link should be 'file missing'
    assert any(r == "file missing" for _, _, r in missing)


def test_unreachable_external_is_reported(sample_site: Path, monkeypatch):
    # external link with download failing
    (sample_site / "ext3.html").write_text(
        '<a href="http://example.invalid/page.html#frag">ext</a>'
    )

    def _raise(self, url, timeout=None, **_kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests.Session, "get", _raise)

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=False,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    assert missing
    # unreachable link should be reported
    assert any(r == "unreachable link" for _, _, r in missing)


def test_ignore_non_html_files(sample_site: Path):
    # create a non-HTML file with a link
    (sample_site / "file.txt").write_text('<a href="page.html#p1">ok</a>')

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    # no links should be checked, so no missing links
    assert not missing
    assert sc.valid_links == 0
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 0
    assert sc.invalid_anchors == 0


def test_downloading_same_url_multiple_times_uses_cache(
    sample_site: Path, mock_requests_ok
):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    console = Console(record=True)
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=True,
        skip_external=False,
    )
    try:
        url = "http://example.invalid/page.html"
        # First download
        # pylint: disable-next=protected-access
        path1 = sc._download_url(url)
        # Second download of the same URL
        # pylint: disable-next=protected-access
        path2 = sc._download_url(url)
        # Both paths should be the same, indicating caching
        assert path1 == path2
        # The mock_requests_ok should have been called only once
        out = console.export_text()
        download_count = out.count("Downloaded")
        assert download_count == 1
    finally:
        sc.close()


def test_normalize_href_with_url_multiple_times_uses_cache(
    sample_site: Path, mock_requests_ok
):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    console = Console(record=True)
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=True,
        skip_external=False,
    )
    try:
        url = "http://example.invalid/page.html#extid"
        # First normalization
        # pylint: disable-next=protected-access
        url_str1, path1, frag1 = sc._normalize_href(sample_site / "dummy.html", url)
        # Second normalization of the same URL
        # pylint: disable-next=protected-access
        url_str2, path2, frag2 = sc._normalize_href(sample_site / "dummy.html", url)
        # Both results should be the same, indicating caching
        assert url_str1 == url_str2
        assert path1 == path2
        assert frag1 == frag2
        # The mock_requests_ok should have been called only once
        out = console.export_text()
        download_count = out.count("Downloaded")
        assert download_count == 1
    finally:
        sc.close()


def test_page_with_id_with_no_value(sample_site: Path, mock_requests_ok):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    # Create a page with an element that has an id attribute with no value
    (sample_site / "idless.html").write_text("<html><body><div id></div></body></html>")

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    # No links should be missing since there are no links in the file
    assert not missing
    assert sc.valid_links == 0
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 0
    assert sc.invalid_anchors == 0


def test_page_with_name_with_no_value(sample_site: Path, mock_requests_ok):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    # Create a page with an element that has a name attribute with no value
    (sample_site / "nameless.html").write_text("<html><body><a name></a></body></html>")

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    # No links should be missing since there are no links in the file
    assert not missing
    assert sc.valid_links == 0
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 0
    assert sc.invalid_anchors == 0


def test_page_with_href_with_no_value(sample_site: Path, mock_requests_ok):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    # Create a page with a link that has an href attribute with no value
    (sample_site / "hrefless.html").write_text("<html><body><a href></a></body></html>")

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    # No links should be missing since the href is empty
    assert not missing
    assert sc.valid_links == 0
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 0
    assert sc.invalid_anchors == 0


def test_skipping_external_links_verbose_output(sample_site: Path, mock_requests_ok):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    # Create a page with an external link
    (sample_site / "external.html").write_text(
        '<a href="http://example.invalid/page.html#frag">ext</a>'
    )

    console = Console(record=True)
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=True,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    # No links should be missing since external links are skipped
    assert not missing
    out = console.export_text()
    # The output should indicate that external links are being skipped
    assert "Skipping external link" in out


def test_page_with_multiple_ids(sample_site: Path, mock_requests_ok):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    # Create a page with multiple elements having ids
    (sample_site / "multiid.html").write_text(
        "<html><body><div id='id1'></div><span id='id2'>"
        "</span><a id='id3'></a></body></html>"
    )

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    for key, value in sc.targets.items():
        if "multiid.html" in str(key):
            assert value.ids == {"id1", "id2", "id3"}

    # No links should be missing since there are no links in the file
    assert not missing
    assert sc.valid_links == 0
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 0
    assert sc.invalid_anchors == 0


def test_page_with_links_with_same_url_and_different_fragments(
    sample_site: Path, mock_requests_ok
):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    # Create a page with multiple links to the same URL but different fragments
    (sample_site / "links.html").write_text(
        '<a href="multiid.html#id1">Link to id1</a>'
        '<a href="multiid.html#id2">Link to id2</a>'
        '<a href="multiid.html#id3">Link to id3</a>'
    )
    # Create the target page with the corresponding ids
    (sample_site / "multiid.html").write_text(
        "<html><body><div id='id1'></div><span id='id2'>"
        "</span><a id='id3'></a></body></html>"
    )

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    # No links should be missing since all fragments exist
    assert not missing
    assert sc.valid_links == 1
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 3
    assert sc.invalid_anchors == 0


def test_link_without_fragment_is_handled_correctly(
    sample_site: Path, mock_requests_ok
):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    # Create a page with a link without a fragment
    (sample_site / "nolinkfrag.html").write_text(
        '<a href="page.html">Link to page without fragment</a>'
    )

    console = Console()
    sc = link_checker.SiteChecker(
        sample_site,
        console=console,
        download_timeout=1,
        verbose=False,
        skip_external=True,
    )
    try:
        missing = sc.check_anchors()
    finally:
        sc.close()

    # No links should be missing since all fragments exist
    assert not missing
    assert sc.valid_links == 1
    assert sc.invalid_links == 0
    assert sc.valid_anchors == 0
    assert sc.invalid_anchors == 0


def test_cli_runner_exit_codes(sample_site: Path, mock_requests_ok):
    assert mock_requests_ok is not None  # to avoid unused variable warning
    runner = CliRunner()

    # success case: create a valid local link
    (sample_site / "ok.html").write_text('<a href="page.html#p1">ok</a>')
    result = runner.invoke(link_checker.app, [str(sample_site), "--skip-external"])
    assert result.exit_code == 0

    # failure case: create a broken link
    (sample_site / "bad.html").write_text('<a href="missing.html#x">bad</a>')
    result2 = runner.invoke(link_checker.app, [str(sample_site), "--skip-external"])
    assert result2.exit_code != 0
