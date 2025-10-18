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
    """Mock ``requests.Session.get`` to return a simple HTML page containing id 'extid'."""

    class FakeResp:
        def __init__(self, content: str):
            self.content = content.encode("utf-8")

        def raise_for_status(self):
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

    # When skipping external links, the run should succeed because externals are not validated
    code = link_checker.main(
        site_dir, download_timeout=1, verbose=False, skip_external=True
    )
    assert code == 0

    # When not skipping externals, the SiteChecker will try to download and fail -> return non-zero
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
