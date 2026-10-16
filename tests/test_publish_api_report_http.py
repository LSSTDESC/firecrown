"""Offline tests for publisher HTTP limits, paths, redirects, and pagination."""

from http.client import HTTPMessage
import io
import json
from unittest.mock import MagicMock, Mock
import urllib.request

import pytest

from tools.publish_api_report import (
    GitHub,
    MAX_DOWNLOAD,
    PublicationError,
    _SafeRedirect,
    _http,
)


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/repos/x/y",
        "/users/example",
        "/repos/x/../y",
        "/repos/x/y#fragment",
    ],
)
def test_request_rejects_unsafe_paths_before_transport(path: str) -> None:
    """Reject paths that could escape the trusted GitHub repository boundary.

    :param path: Absolute URL, non-repository path, traversal, or fragment.
    """
    api = GitHub(
        "test-token",
        lambda _method, _url, _headers, _body: pytest.fail(
            "Unsafe paths must not reach the transport"
        ),
    )
    with pytest.raises(PublicationError, match="^Unsafe GitHub API path$"):
        api.request(path)


@pytest.mark.parametrize("size", [MAX_DOWNLOAD, MAX_DOWNLOAD + 1])
def test_request_enforces_download_limit(size: int) -> None:
    """Accept the download boundary and reject even one excess byte.

    :param size: Transport response size in bytes.
    """
    response = b"x" * size
    api = GitHub("test-token", lambda _method, _url, _headers, _body: response)
    if size > MAX_DOWNLOAD:
        with pytest.raises(PublicationError, match="exceeds download limit"):
            api.request("/repos/x/y")
    else:
        assert api.request("/repos/x/y") == response


@pytest.mark.parametrize("size", [MAX_DOWNLOAD, MAX_DOWNLOAD + 1])
def test_http_bounds_reads_and_configures_safe_opener(
    monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    """Exercise the default transport without opening any network connection.

    :param monkeypatch: Replace urllib's non-injectable opener factory.
    :param size: Response size at or just above the download boundary.
    """
    response = b"x" * size
    reply = MagicMock()
    reply.__enter__.return_value = reply
    reply.read.return_value = response
    opener = Mock()
    opener.open.return_value = reply
    build_opener = Mock(return_value=opener)
    # _http has no opener injection; replace only this network boundary.
    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
    if size > MAX_DOWNLOAD:
        with pytest.raises(PublicationError, match="exceeds download limit"):
            _http("POST", "https://api.github.com/repos/x/y", headers, b"{}")
    else:
        assert (
            _http("POST", "https://api.github.com/repos/x/y", headers, b"{}")
            == response
        )
    build_opener.assert_called_once()
    assert isinstance(build_opener.call_args.args[0], _SafeRedirect)
    opener.open.assert_called_once()
    request = opener.open.call_args.args[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "https://api.github.com/repos/x/y"
    assert request.get_method() == "POST"
    assert request.data == b"{}"
    assert request.get_header("Authorization") == "Bearer secret"
    assert request.get_header("Content-type") == "application/json"
    assert opener.open.call_args.kwargs == {"timeout": 30}
    reply.read.assert_called_once_with(MAX_DOWNLOAD + 1)
    reply.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("method", ["POST", "PATCH", "HEAD"])
def test_redirect_rejects_non_get_methods(method: str) -> None:
    """Never redirect authenticated writes or other non-download requests.

    :param method: HTTP method other than GET.
    """
    request = urllib.request.Request(
        "https://api.github.com/repos/x/y",
        headers={"Authorization": "Bearer secret"},
        method=method,
    )
    with pytest.raises(PublicationError, match="^Unsafe API redirect$"):
        _SafeRedirect().redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            HTTPMessage(),
            "https://storage.example/artifact",
        )


def test_redirect_preserves_parent_declining_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Respect urllib's documented optional result without creating a request.

    :param monkeypatch: Supply the parent's documented no-redirect result.
    """
    parent_redirect = Mock(return_value=None)
    # urllib documents None but its default handler always returns or raises.
    monkeypatch.setattr(
        urllib.request.HTTPRedirectHandler, "redirect_request", parent_redirect
    )
    request = urllib.request.Request("https://api.github.com/repos/x/y")
    stream, headers = io.BytesIO(), HTTPMessage()
    assert (
        _SafeRedirect().redirect_request(
            request, stream, 302, "Found", headers, "https://storage.example/artifact"
        )
        is None
    )
    parent_redirect.assert_called_once_with(
        request, stream, 302, "Found", headers, "https://storage.example/artifact"
    )


@pytest.mark.parametrize("key", [None, "jobs"])
@pytest.mark.parametrize("value", [None, {}, "not a list", 1])
def test_pages_rejects_non_list_collections(key: str | None, value: object) -> None:
    """Reject malformed collections in bare and object-wrapped responses.

    :param key: Optional wrapper key for the paginated collection.
    :param value: Invalid collection value supplied by the transport.
    """
    response = json.dumps(value if key is None else {key: value}).encode("utf-8")
    api = GitHub("test-token", lambda _method, _url, _headers, _body: response)
    with pytest.raises(PublicationError, match="^Invalid GitHub pagination response$"):
        api.pages("/repos/x/y/actions/runs/1/jobs", key)


def test_pages_stops_at_pagination_limit() -> None:
    """Bound a server that never returns a short page to exactly 1,000 requests."""
    urls: list[str] = []
    response = json.dumps([{}] * 100).encode("utf-8")

    def transport(
        _method: str, url: str, _headers: dict[str, str], _body: bytes | None
    ) -> bytes:
        """Record a requested page and return another full collection."""
        urls.append(url)
        return response

    api = GitHub("test-token", transport)
    with pytest.raises(PublicationError, match="^GitHub pagination limit exceeded$"):
        api.pages("/repos/x/y/issues")
    assert len(urls) == 1000
    assert urls[0] == "https://api.github.com/repos/x/y/issues?per_page=100&page=1"
    assert urls[-1] == "https://api.github.com/repos/x/y/issues?per_page=100&page=1000"
