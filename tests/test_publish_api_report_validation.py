"""Offline publisher validation tests for malformed provenance and resource bounds."""

import json
from typing import cast
import urllib.parse

import pytest

from tests.test_publish_api_report import FakeGitHub, REPOSITORY, ROOT
from tools.publish_api_report import (
    GitHub,
    Json,
    MAX_DOWNLOAD,
    PublicationError,
    StaleRun,
    publish,
    resolve,
    resolve_in_run,
)


@pytest.mark.parametrize(
    "repository",
    ["", "LSSTDESC", "LSSTDESC/firecrown/extra", "LSSTDESC/firecrown#fragment"],
)
@pytest.mark.parametrize("in_run", [False, True])
def test_resolution_rejects_invalid_repository(repository: str, in_run: bool) -> None:
    """Reject malformed trusted repository names before making any API request.

    :param repository: Repository identity that is not an owner/name pair.
    :param in_run: Whether resolution occurs inside the analysis workflow.
    """
    fake = FakeGitHub()
    api = GitHub("test-token", fake.transport)
    with pytest.raises(PublicationError, match="^Invalid GITHUB_REPOSITORY$"):
        if in_run:
            resolve_in_run(api, fake.event, repository, "100", "1")
        else:
            resolve(api, fake.event, repository)
    assert not fake.calls
    assert not fake.writes


@pytest.mark.parametrize("field", ["id", "run_attempt", "workflow_id", "run_number"])
@pytest.mark.parametrize("value", [None, "1", True, False, 0, -1, 1.5])
def test_resolution_requires_positive_integer_identifiers(
    field: str, value: object
) -> None:
    """Reject missing, non-integer, boolean, and nonpositive run identifiers.

    :param field: Trusted run identifier being validated.
    :param value: Invalid identifier value.
    """
    fake = FakeGitHub()
    fake.run[field] = value
    with pytest.raises(
        PublicationError, match="^Expected a positive integer identifier$"
    ):
        fake.publish()
    assert not fake.calls
    assert not fake.writes


@pytest.mark.parametrize("value", [None, {}, "not a list"])
def test_resolution_rejects_missing_pr_association_array(value: object) -> None:
    """Do not treat malformed run associations as an empty fallback collection.

    :param value: Invalid pull_requests collection in the trusted run response.
    """
    fake = FakeGitHub()
    fake.run["pull_requests"] = value
    with pytest.raises(PublicationError, match="^Missing run pull_requests array$"):
        fake.publish()
    assert not any("/commits/" in url or "/pulls/" in url for _, url in fake.calls)
    assert not fake.writes


@pytest.mark.parametrize("field,value", [("sha", "d" * 40), ("ref", "another")])
def test_resolution_rejects_stale_association_base(field: str, value: str) -> None:
    """Refuse publication when the association no longer matches the PR target.

    :param field: Association base identity component to change.
    :param value: Stale base SHA or target branch.
    """
    fake = FakeGitHub()
    association = cast(list[Json], fake.run["pull_requests"])[0]
    cast(Json, association["base"])[field] = value
    with pytest.raises(StaleRun, match="^PR base or target no longer matches the run$"):
        fake.publish()
    assert not fake.writes


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 101),
        ("run_attempt", 2),
        ("workflow_id", 6),
        ("run_number", 11),
        ("head_sha", "d" * 40),
        ("head_branch", "another"),
        ("status", "in_progress"),
        ("head_repository", {"id": 3, "full_name": "other/firecrown"}),
    ],
)
def test_resolution_rejects_attempt_identity_mismatch(
    field: str, value: object
) -> None:
    """Reject every differing identity field in the attempt-specific API response.

    :param field: Attempt response field that differs from the workflow event.
    :param value: Valid but mismatched attempt identity.
    """
    fake = FakeGitHub()

    def transport(
        method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> bytes:
        """Replace one attempt identity field in the existing offline response."""
        response = fake.transport(method, url, headers, body)
        if urllib.parse.urlsplit(url).path == ROOT + "/actions/runs/100/attempts/1":
            attempt = cast(Json, json.loads(response))
            attempt[field] = value
            return json.dumps(attempt).encode("utf-8")
        return response

    message = (
        "Run attempt head repository mismatch"
        if field == "head_repository"
        else f"Run attempt identity mismatch: {field}"
    )
    with pytest.raises(PublicationError, match=f"^{message}$"):
        resolve(GitHub("test-token", transport), fake.event, REPOSITORY)
    assert len(fake.calls) == 1
    assert not fake.writes


@pytest.mark.parametrize("value", [None, {}, "not a list"])
def test_freshness_rejects_malformed_newer_run_associations(value: object) -> None:
    """Fail closed when a newer run has no usable PR association collection.

    :param value: Invalid association collection in a newer matching run.
    """
    fake = FakeGitHub()
    fake.runs = [{**fake.run, "run_number": 11, "pull_requests": value}]
    with pytest.raises(PublicationError, match="^Invalid newer-run PR associations$"):
        fake.publish()
    assert not fake.writes


def test_newer_run_for_different_pr_does_not_block_publication() -> None:
    """Keep an unrelated PR's newer run from superseding the resolved PR report."""
    fake = FakeGitHub()
    fake.runs = [{**fake.run, "run_number": 11, "pull_requests": [{"number": 8}]}]
    assert "PR #7" in fake.publish()
    assert [(method, path) for method, path, _ in fake.writes] == [
        ("POST", ROOT + "/issues/7/comments")
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_attempt", 2),
        ("head_sha", "d" * 40),
        ("head_branch", "another"),
        ("workflow_id", 6),
        ("status", "in_progress"),
    ],
)
def test_final_run_identity_change_never_writes(field: str, value: object) -> None:
    """Detect run identity changes after initial resolution but before publication.

    :param field: Current run field that changes after the first identity check.
    :param value: New identity or state at the final check.
    """
    fake = FakeGitHub()
    run_reads = 0

    def transport(
        method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> bytes:
        """Change the live run response only when the final trust check reads it."""
        nonlocal run_reads
        response = fake.transport(method, url, headers, body)
        if urllib.parse.urlsplit(url).path == ROOT + "/actions/runs/100":
            run_reads += 1
            if run_reads == 2:
                latest = cast(Json, json.loads(response))
                latest[field] = value
                return json.dumps(latest).encode("utf-8")
        return response

    with pytest.raises(
        StaleRun, match="^Run identity or attempt changed before publication$"
    ):
        publish(GitHub("test-token", transport), fake.event, REPOSITORY)
    assert run_reads == 2
    assert not fake.writes


def test_declared_oversized_artifact_is_not_downloaded() -> None:
    """Publish an unassessed result without downloading an oversized artifact."""
    fake = FakeGitHub()
    fake.artifacts[0]["size_in_bytes"] = MAX_DOWNLOAD + 1
    with pytest.raises(PublicationError, match="Artifact exceeds download limit"):
        fake.publish()
    assert not any("/zip" in url for _, url in fake.calls)
    assert len(fake.writes) == 1
    assert fake.writes[0][:2] == ("POST", ROOT + "/issues/7/comments")
    assert "Compatibility was not assessed" in str(fake.writes[0][2]["body"])
