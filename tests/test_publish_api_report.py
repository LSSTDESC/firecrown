"""Network-free trust-boundary tests for the privileged API report publisher."""

from copy import deepcopy
from http.client import HTTPMessage
import io
import json
from pathlib import Path
import stat
from typing import cast
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile

import pytest

from tools.publish_api_report import (
    GitHub,
    Json,
    MARKER,
    MAX_COMMENT,
    MAX_DOWNLOAD,
    MAX_METADATA,
    MAX_REPORT,
    PublicationError,
    StaleRun,
    _SafeRedirect,
    main,
    publish,
)

REPOSITORY = "LSSTDESC/firecrown"
ROOT = f"/repos/{REPOSITORY}"
HEAD = "a" * 40
BASE = "b" * 40
REPO: Json = {"id": 1, "full_name": REPOSITORY}
FORK: Json = {"id": 2, "full_name": "contributor/firecrown"}
VERDICT = "No breaking API changes detected."


def _zip(metadata: Json, report: str = "# Report\n\nFindings") -> bytes:
    """Build an in-memory, never-extracted report artifact fixture."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("api-changes.md", report)
        archive.writestr("metadata.json", json.dumps(metadata))
    return buffer.getvalue()


class FakeGitHub:  # pylint: disable=too-many-instance-attributes
    """Model paginated GitHub resources and capture writes without a network."""

    def __init__(self) -> None:
        """Create a completed fork PR run and matching report artifact."""
        self.pr: Json = {
            "number": 7,
            "state": "open",
            "head": {"sha": HEAD, "ref": "feature", "repo": FORK},
            "base": {"sha": BASE, "ref": "master", "repo": REPO},
        }
        self.run: Json = {
            "id": 100,
            "run_attempt": 1,
            "run_number": 10,
            "workflow_id": 5,
            "name": "firecrown-api-changes",
            "path": ".github/workflows/api-changes.yml",
            "event": "pull_request",
            "status": "completed",
            "repository": REPO,
            "head_sha": HEAD,
            "head_branch": "feature",
            "head_repository": FORK,
            "pull_requests": [{"number": 7, "base": {"sha": BASE, "ref": "master"}}],
        }
        self.event: Json = {
            "action": "completed",
            "repository": REPO,
            "workflow_run": self.run,
        }
        self.metadata: Json = {
            "schema": 1,
            "repository": REPOSITORY,
            "run_id": 100,
            "run_attempt": 1,
            "pr_number": 7,
            "head_sha": HEAD,
            "base_sha": BASE,
            "target": "master",
            "status": "completed",
            "verdict": VERDICT,
            "baseline": "c" * 40,
        }
        self.report = "# API differences\n\nNo findings."
        self.artifacts: list[Json] = [
            {
                "id": 3,
                "name": "public-api-report-100-1",
                "expired": False,
                "size_in_bytes": 500,
            }
        ]
        self.jobs: list[Json] = [
            {
                "id": 8,
                "name": "PR public API changes",
                "html_url": "https://evil.test/job",
            }
        ]
        self.comments: list[Json] = []
        self.runs: list[Json] = []
        self.associated: list[Json] = [{"number": 7}]
        self.latest_attempt = 1
        self.final_pr: Json | None = None
        self.pr_reads = 0
        self.archive: bytes | None = None
        self.fail_path: str | None = None
        self.calls: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, Json]] = []

    def transport(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> bytes:
        """Serve fixtures and assert the request uses only trusted API URLs.

        :param method: REST method.
        :param url: API request URL.
        :param headers: Authenticated trusted headers.
        :param body: Optional encoded comment payload.
        :returns: Fixture response bytes.
        """
        assert url.startswith("https://api.github.com" + ROOT)
        assert headers["Authorization"] == "Bearer test-token"
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path
        self.calls.append((method, url))
        if self.fail_path in (path, method):
            raise urllib.error.HTTPError(
                url, 403, "permission denied", HTTPMessage(), None
            )
        if method != "GET":
            assert body is not None
            self.writes.append((method, path, cast(Json, json.loads(body))))
            return b"{}"
        if path == ROOT + "/actions/runs/100/attempts/1":
            value: object = self.run
        elif path == ROOT + "/actions/runs/100":
            value = {**self.run, "run_attempt": self.latest_attempt}
        elif path == ROOT + "/pulls/7":
            self.pr_reads += 1
            value = (
                self.final_pr
                if self.pr_reads > 1 and self.final_pr is not None
                else self.pr
            )
        elif path == ROOT + "/actions/artifacts/3/zip":
            return (
                self.archive
                if self.archive is not None
                else _zip(self.metadata, self.report)
            )
        else:
            collections: dict[str, tuple[list[Json], str | None]] = {
                ROOT + "/actions/runs/100/attempts/1/jobs": (self.jobs, "jobs"),
                ROOT + "/actions/runs/100/artifacts": (self.artifacts, "artifacts"),
                ROOT + "/issues/7/comments": (self.comments, None),
                ROOT + "/actions/workflows/5/runs": (self.runs, "workflow_runs"),
                ROOT + f"/commits/{HEAD}/pulls": (self.associated, None),
            }
            items, key = collections[path]
            page = int(urllib.parse.parse_qs(parsed.query)["page"][0])
            start, end = (page - 1) * 100, page * 100
            chunk = items[start:end]
            value = chunk if key is None else {key: chunk}
        return json.dumps(value).encode("utf-8")

    def publish(self) -> str:
        """Invoke the publisher with the fixture's injectable transport.

        :returns: Successful publication summary.
        """
        return publish(GitHub("test-token", self.transport), self.event, REPOSITORY)


def _comment(
    body: str = MARKER + "\nOld verdict",
    login: str = "github-actions[bot]",
    kind: str = "Bot",
) -> Json:
    """Construct an existing issue comment with a selectable author."""
    return {"id": 42, "body": body, "user": {"login": login, "type": kind}}


@pytest.mark.parametrize(
    "target,verdict",
    [
        ("master", VERDICT),
        (
            "master",
            "Breaking API changes detected (2 findings). "
            "Informational because the target is `master`. "
            "Reviewer assessment required.",
        ),
        (
            "v1_16_support",
            "Breaking API changes detected (1 findings). "
            "This fails support-line policy.",
        ),
    ],
)
@pytest.mark.parametrize("update", [False, True])
def test_all_verdicts(target: str, verdict: str, update: bool) -> None:
    """Publish each API policy verdict on creation and update.

    :param target: PR target.
    :param verdict: Full metadata verdict.
    :param update: Whether an owned comment already exists.
    """
    fake = FakeGitHub()
    cast(Json, fake.pr["base"])["ref"] = target
    cast(Json, cast(list[Json], fake.run["pull_requests"])[0]["base"])["ref"] = target
    fake.metadata.update(target=target, verdict=verdict)
    if update:
        fake.comments = [_comment()]
    assert "PR #7" in fake.publish()
    method, path, payload = fake.writes[0]
    assert method == ("PATCH" if update else "POST")
    assert path == (
        ROOT + "/issues/comments/42" if update else ROOT + "/issues/7/comments"
    )
    assert verdict in str(payload["body"])
    assert "/actions/runs/100/job/8" in str(payload["body"])
    assert "evil.test" not in str(payload["body"])


@pytest.mark.parametrize(
    "login,kind,body",
    [
        ("human", "User", MARKER + "\nOld"),
        ("github-actions[bot]", "User", MARKER + "\nOld"),
        ("other[bot]", "Bot", MARKER + "\nOld"),
        ("github-actions[bot]", "Bot", "User text " + MARKER),
    ],
)
def test_only_owned_comments(login: str, kind: str, body: str) -> None:
    """Never edit comments merely containing a marker or resembling the bot.

    :param login: Comment author login.
    :param kind: Comment author type.
    :param body: Existing body.
    """
    fake = FakeGitHub()
    fake.comments = [_comment(body, login, kind)]
    fake.publish()
    assert fake.writes[0][0] == "POST"


@pytest.mark.parametrize(
    "failure",
    ["missing", "expired", "ambiguous", "failed", "corrupt", "download", "jobs"],
)
def test_reporting_failure_replaces_old_verdict(failure: str) -> None:
    """Replace old verdicts with unassessed and still fail the publisher check.

    :param failure: Analysis, artifact, or retrieval failure.
    """
    fake = FakeGitHub()
    fake.comments = [_comment()]
    if failure == "missing":
        fake.artifacts = []
    elif failure == "expired":
        fake.artifacts[0]["expired"] = True
    elif failure == "ambiguous":
        fake.artifacts *= 2
    elif failure == "failed":
        fake.metadata["status"] = "failed"
    elif failure == "corrupt":
        fake.archive = b"not a zip"
    elif failure == "download":
        fake.fail_path = ROOT + "/actions/artifacts/3/zip"
    else:
        fake.fail_path = ROOT + "/actions/runs/100/attempts/1/jobs"
    with pytest.raises(PublicationError, match="Published compatibility not assessed"):
        fake.publish()
    assert fake.writes[0][0] == "PATCH"
    assert "Compatibility was not assessed" in str(fake.writes[0][2]["body"])
    assert VERDICT not in str(fake.writes[0][2]["body"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("repository", "attacker/repo"),
        ("run_id", 101),
        ("run_attempt", 2),
        ("pr_number", 999),
        ("base_sha", "not-a-sha"),
        ("target", ""),
        ("head_sha", "d" * 40),
        ("verdict", "Compatible!"),
        ("baseline", "main"),
        ("status", "other"),
    ],
)
def test_metadata_cannot_redirect_or_forge_verdict(field: str, value: object) -> None:
    """Reject mismatched metadata without posting to artifact-selected PRs.

    :param field: Metadata field to corrupt.
    :param value: Untrusted replacement.
    """
    fake = FakeGitHub()
    fake.metadata[field] = value
    with pytest.raises(PublicationError):
        fake.publish()
    assert fake.writes[0][1] == ROOT + "/issues/7/comments"
    assert "not assessed" in str(fake.writes[0][2]["body"])


@pytest.mark.parametrize(
    "attack",
    [
        "traversal",
        "absolute",
        "duplicate",
        "symlink",
        "directory",
        "extra",
        "report_size",
        "metadata_size",
        "compressed_size",
        "download_size",
    ],
)
def test_zip_attacks(attack: str) -> None:
    """Reject unsafe ZIP structure and size limits without extraction.

    :param attack: Malicious archive fixture.
    """
    fake = FakeGitHub()
    buffer = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(buffer, "w") as archive:
        warnings.simplefilter("ignore", UserWarning)
        name = {"traversal": "../api-changes.md", "absolute": "/api-changes.md"}.get(
            attack, "api-changes.md"
        )
        member = zipfile.ZipInfo(name)
        if attack == "symlink":
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
        if attack == "directory":
            member.external_attr = (stat.S_IFDIR | 0o777) << 16
        archive.writestr(
            member,
            (
                "x" * (MAX_REPORT + 1)
                if attack in ("report_size", "compressed_size")
                else "report"
            ),
        )
        archive.writestr(
            "metadata.json",
            (
                "x" * (MAX_METADATA + 1)
                if attack == "metadata_size"
                else json.dumps(fake.metadata)
            ),
        )
        if attack in ("extra", "duplicate"):
            archive.writestr("evil.py" if attack == "extra" else name, "pass")
    fake.archive = (
        b"x" * (MAX_DOWNLOAD + 1) if attack == "download_size" else buffer.getvalue()
    )
    with pytest.raises(PublicationError):
        fake.publish()
    assert "not assessed" in str(fake.writes[0][2]["body"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "untrusted"),
        ("path", ".github/workflows/evil.yml"),
        ("event", "push"),
        ("repository", FORK),
        ("status", "in_progress"),
    ],
)
def test_reject_untrusted_run(field: str, value: object) -> None:
    """Reject runs outside the trusted upstream workflow boundary.

    :param field: Trusted run field being corrupted.
    :param value: Unexpected value.
    """
    fake = FakeGitHub()
    fake.run[field] = value
    with pytest.raises(PublicationError):
        fake.publish()
    assert not fake.writes


@pytest.mark.parametrize(
    "change",
    [
        "attempt",
        "head",
        "head_repo",
        "head_ref",
        "base",
        "target",
        "base_repo",
        "closed",
        "new_run",
        "new_in_progress",
        "comment",
    ],
)
def test_stale_report_never_writes(change: str) -> None:
    """Reject changed PR snapshots, retry attempts, and superseding runs.

    :param change: Staleness discovered at the final trust check.
    """
    fake = FakeGitHub()
    final = deepcopy(fake.pr)
    if change == "attempt":
        fake.latest_attempt = 2
    elif change in ("new_run", "new_in_progress"):
        fake.runs = [
            {
                **fake.run,
                "id": 101,
                "run_number": 11,
                "status": "in_progress" if change == "new_in_progress" else "completed",
            }
        ]
    elif change == "comment":
        fake.comments = [
            _comment(MARKER + "\n<!-- firecrown-api-run:100:2 -->\nNew verdict")
        ]
    elif change == "closed":
        final["state"] = "closed"
    else:
        component = cast(Json, final["head" if change.startswith("head") else "base"])
        field = (
            "repo"
            if change.endswith("repo")
            else "ref" if change in ("target", "head_ref") else "sha"
        )
        component[field] = (
            FORK if field == "repo" else "other" if field == "ref" else "d" * 40
        )
        if change == "head_repo":
            component[field] = REPO
    fake.final_pr = final
    with pytest.raises(StaleRun):
        fake.publish()
    assert not fake.writes


def test_fork_empty_association_fallback() -> None:
    """Resolve empty fork associations via commit PRs, never metadata."""
    fake = FakeGitHub()
    fake.run["pull_requests"] = []
    fake.associated = [deepcopy(fake.pr)]
    fake.publish()
    assert any(f"/commits/{HEAD}/pulls" in url for _, url in fake.calls)


@pytest.mark.parametrize("updated_head", [False, True])
def test_compact_mutable_run_association(updated_head: bool) -> None:
    """Use the real compact association schema without treating it as historical.

    :param updated_head: Whether the association now points to a newer PR head.
    """
    fake = FakeGitHub()
    current_head = "d" * 40 if updated_head else HEAD
    fake.run["pull_requests"] = [
        {
            "number": 7,
            "head": {
                "ref": "feature",
                "sha": current_head,
                "repo": {
                    "id": 2,
                    "name": "firecrown",
                    "url": "https://api.github.com/repos/contributor/firecrown",
                },
            },
            "base": {
                "ref": "master",
                "sha": BASE,
                "repo": {
                    "id": 1,
                    "name": "firecrown",
                    "url": "https://api.github.com/repos/LSSTDESC/firecrown",
                },
            },
        }
    ]
    cast(Json, fake.pr["head"])["sha"] = current_head
    if updated_head:
        with pytest.raises(StaleRun):
            fake.publish()
        assert not fake.writes
    else:
        fake.publish()
        assert len(fake.writes) == 1


@pytest.mark.parametrize("change", ["none", "head", "attempt", "closed"])
def test_read_only_resolution(tmp_path: Path, change: str) -> None:
    """Resolve per-PR serialization identity without artifacts or write permission.

    :param tmp_path: Isolated workflow environment files.
    :param change: Optional stale PR head, run attempt, or PR state.
    """
    fake = FakeGitHub()
    fake.run["pull_requests"] = []
    if change == "head":
        cast(Json, fake.pr["head"])["sha"] = "d" * 40
    elif change == "attempt":
        fake.latest_attempt = 2
    elif change == "closed":
        fake.pr["state"] = "closed"
    event = tmp_path / "event.json"
    event.write_text(json.dumps(fake.event))
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    status = main(
        {
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_TOKEN": "test-token",
            "GITHUB_STEP_SUMMARY": str(summary),
            "GITHUB_OUTPUT": str(output),
            "API_REPORT_RESOLVE_ONLY": "true",
        },
        fake.transport,
    )
    assert status == 0
    assert not fake.writes
    assert not any("artifacts" in url or "comments" in url for _, url in fake.calls)
    if change == "none":
        assert output.read_text() == "pr_number=7\n"
    else:
        assert not output.exists()


def test_publish_rechecks_resolved_concurrency_identity() -> None:
    """Never write outside the PR whose trusted identity holds the concurrency key."""
    fake = FakeGitHub()
    with pytest.raises(PublicationError, match="concurrency resolution"):
        publish(GitHub("test-token", fake.transport), fake.event, REPOSITORY, "8")
    assert not fake.writes


@pytest.mark.parametrize(
    "field,value", [("target", "v1_16_support"), ("base_sha", "d" * 40)]
)
def test_retargeted_artifact_never_replaces_comment(field: str, value: str) -> None:
    """Reject old target/base reports even when fork associations are current.

    :param field: Stale artifact target or base field.
    :param value: Previous comparison identity that no longer matches the PR.
    """
    fake = FakeGitHub()
    fake.run["pull_requests"] = []
    fake.associated = [deepcopy(fake.pr)]
    fake.metadata[field] = value
    fake.comments = [_comment()]
    with pytest.raises(StaleRun):
        fake.publish()
    assert not fake.writes


@pytest.mark.parametrize(
    "field,value", [("sha", "d" * 40), ("ref", "another"), ("repo", REPO)]
)
def test_fork_fallback_requires_exact_head(field: str, value: object) -> None:
    """Require matching SHA, fork repository, and branch in fallback PRs.

    :param field: PR head field.
    :param value: Mismatched value.
    """
    fake = FakeGitHub()
    fake.run["pull_requests"] = []
    cast(Json, fake.pr["head"])[field] = value
    with pytest.raises(StaleRun):
        fake.publish()
    assert not fake.writes


def test_all_collections_paginate() -> None:
    """Find jobs, artifacts, comments, associated PRs, and newer runs on page two."""
    fake = FakeGitHub()
    fake.run["pull_requests"] = []
    padding: list[Json] = [{"name": "other"}] * 100
    fake.jobs = padding + fake.jobs
    fake.artifacts = padding + fake.artifacts
    fake.comments = [_comment(login="human")] * 100 + [_comment()]
    fake.associated = [{"number": 7}] * 100
    # Repeated PRs are intentionally ambiguous, proving fallback consumed page two.
    with pytest.raises(StaleRun):
        fake.publish()
    assert any("pulls?per_page=100&page=2" in url for _, url in fake.calls)
    fake = FakeGitHub()
    fake.jobs = padding + fake.jobs
    fake.artifacts = padding + fake.artifacts
    fake.comments = [_comment(login="human")] * 100 + [_comment()]
    fake.runs = [{**fake.run, "run_number": 9}] * 100
    fake.publish()
    for resource in ("jobs", "artifacts", "comments", "runs"):
        assert any(
            resource + "?" in url
            and "page=2" in url
            or resource + "?event=" in url
            and "page=2" in url
            for _, url in fake.calls
        )
    assert fake.writes[0][0] == "PATCH"


def test_oversized_report_preserves_validated_verdict_and_escapes_markers() -> None:
    """Retain metadata verdict and trusted provenance while truncating raw text."""
    fake = FakeGitHub()
    fake.report = MARKER + "\n<!-- firecrown-api-run:999:999 -->\n" + "\u2603" * 50000
    fake.publish()
    body = str(fake.writes[0][2]["body"])
    assert len(body.encode("utf-8")) <= MAX_COMMENT
    assert body.count(MARKER) == 1
    assert "<!-- firecrown-api-run:999:999 -->" not in body
    assert VERDICT in body
    assert "Report truncated" in body
    assert "https://github.com/LSSTDESC/firecrown/actions/runs/100/attempts/1" in body


def test_markdown_signatures_are_preserved() -> None:
    """Reuse the report Markdown without entity-escaping its API code snippets."""
    fake = FakeGitHub()
    fake.report = "```python\ndef f() -> list[int]:\n```\n\n`A & B`"
    fake.publish()
    assert fake.report in str(fake.writes[0][2]["body"])


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_write_error_fails_visible_summary(tmp_path: Path, method: str) -> None:
    """Expose permission errors in the publisher check and step summary.

    :param tmp_path: Isolated trusted event and summary paths.
    :param method: Failing comment write method.
    """
    fake = FakeGitHub()
    if method == "PATCH":
        fake.comments = [_comment()]
    fake.fail_path = method
    event = tmp_path / "event.json"
    event.write_text(json.dumps(fake.event), encoding="utf-8")
    summary = tmp_path / "summary.md"
    assert (
        main(
            {
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_TOKEN": "test-token",
                "GITHUB_STEP_SUMMARY": str(summary),
            },
            fake.transport,
        )
        == 1
    )
    assert "permission denied" in summary.read_text(encoding="utf-8")


def test_redirect_strips_authorization() -> None:
    """Do not leak the API token to GitHub's signed artifact storage redirect."""
    request = urllib.request.Request(
        "https://api.github.com/repos/x/y", headers={"Authorization": "Bearer secret"}
    )
    redirected = _SafeRedirect().redirect_request(
        request,
        io.BytesIO(),
        302,
        "Found",
        HTTPMessage(),
        "https://storage.example/artifact",
    )
    assert redirected is not None
    assert not redirected.has_header("Authorization")
    with pytest.raises(PublicationError):
        _SafeRedirect().redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            HTTPMessage(),
            "http://storage.example/artifact",
        )


@pytest.mark.parametrize("stale", [False, True])
def test_cli_success_and_stale_summary(tmp_path: Path, stale: bool) -> None:
    """Write visible summaries for publication success and stale skips.

    :param tmp_path: Isolated event and summary files.
    :param stale: Whether a newer attempt exists.
    """
    fake = FakeGitHub()
    fake.latest_attempt = 2 if stale else 1
    event, summary = tmp_path / "event.json", tmp_path / "summary.md"
    event.write_text(json.dumps(fake.event), encoding="utf-8")
    assert (
        main(
            {
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_TOKEN": "test-token",
                "GITHUB_STEP_SUMMARY": str(summary),
            },
            fake.transport,
        )
        == 0
    )
    assert ("Skipped stale" if stale else "Published public API") in summary.read_text(
        encoding="utf-8"
    )
    assert bool(fake.writes) is not stale


def test_api_error_before_pr_resolution_never_posts() -> None:
    """Fail without posting when the trusted PR cannot be established."""
    fake = FakeGitHub()
    fake.fail_path = ROOT + "/pulls/7"
    with pytest.raises(urllib.error.HTTPError):
        fake.publish()
    assert not fake.writes


def test_newer_run_on_second_page_blocks_write() -> None:
    """Detect in-progress superseding runs even beyond the first page."""
    fake = FakeGitHub()
    fake.runs = [{**fake.run, "run_number": 9}] * 100 + [
        {**fake.run, "run_number": 11, "status": "in_progress", "pull_requests": []}
    ]
    with pytest.raises(StaleRun):
        fake.publish()
    assert not fake.writes


def test_ambiguous_owned_comments_never_writes() -> None:
    """Avoid choosing arbitrarily between duplicated publisher-owned comments."""
    fake = FakeGitHub()
    fake.comments = [_comment(), _comment()]
    with pytest.raises(PublicationError, match="Multiple publisher-owned"):
        fake.publish()
    assert not fake.writes
