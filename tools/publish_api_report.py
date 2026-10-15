"""Publish untrusted API report text using only trusted GitHub run provenance."""

# Nested f-string lookups retain the project's pre-3.12 quoting convention.
# pylint: disable=inconsistent-quotes

from __future__ import annotations

from collections.abc import Callable, Mapping
import html
from http.client import HTTPException, HTTPMessage
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import IO, cast
import urllib.parse
import urllib.request
import zipfile
import zlib

MARKER = "<!-- firecrown-public-api-report -->"
STATE = re.compile(r"<!-- firecrown-api-run:(\d+):(\d+) -->")
SHA = re.compile(r"[0-9a-f]{40}")
MAX_DOWNLOAD = 2 * 1024 * 1024
MAX_REPORT = 1024 * 1024
MAX_METADATA = 16 * 1024
MAX_COMMENT = 60000
Json = dict[str, object]
Transport = Callable[[str, str, dict[str, str], bytes | None], bytes]


class PublicationError(ValueError):
    """An API report could not be safely published."""


class StaleRun(PublicationError):
    """A superseded report must not replace the current PR comment."""


def _object(value: object) -> Json:
    if not isinstance(value, dict):
        raise PublicationError("Expected a JSON object")
    return cast(Json, value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PublicationError("Expected a positive integer identifier")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationError("Expected a nonempty string")
    return value


def _sha(value: object) -> str:
    text = _text(value)
    if not SHA.fullmatch(text):
        raise PublicationError("Invalid commit SHA")
    return text


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Follow HTTPS artifact redirects without forwarding the GitHub token.

        :param req: Original authenticated request.
        :param fp: Original response stream.
        :param code: Redirect status code.
        :param msg: Redirect reason.
        :param headers: Response headers.
        :param newurl: HTTPS redirect destination.
        :returns: Unauthenticated redirected request.
        """
        if req.get_method() != "GET" or urllib.parse.urlsplit(newurl).scheme != "https":
            raise PublicationError("Unsafe API redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def _http(method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.build_opener(_SafeRedirect()).open(
        request, timeout=30
    ) as reply:
        result = reply.read(MAX_DOWNLOAD + 1)
    if len(result) > MAX_DOWNLOAD:
        raise PublicationError("GitHub response exceeds download limit")
    return result


class GitHub:
    """A bounded stdlib REST client with an injectable, network-free transport."""

    def __init__(self, token: str, transport: Transport = _http) -> None:
        """Configure the authenticated transport.

        :param token: Trusted workflow token, never supplied by an artifact.
        :param transport: HTTP boundary, injectable for local tests.
        """
        self.transport = transport
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }

    def request(
        self, path: str, body: Json | None = None, method: str = "GET"
    ) -> bytes:
        """Request a trusted API-relative path, without artifact-provided URLs.

        :param path: API-relative path constructed from validated identifiers.
        :param body: Optional JSON request body.
        :param method: REST method.
        :returns: Bounded response bytes.
        """
        if not path.startswith("/repos/") or ".." in path or "#" in path:
            raise PublicationError("Unsafe GitHub API path")
        result = self.transport(
            method,
            "https://api.github.com" + path,
            self.headers,
            None if body is None else json.dumps(body).encode("utf-8"),
        )
        if len(result) > MAX_DOWNLOAD:
            raise PublicationError("GitHub response exceeds download limit")
        return result

    def get(self, path: str) -> Json:
        """Read one JSON object.

        :param path: Trusted API-relative resource path.
        :returns: Decoded JSON object.
        """
        return _object(json.loads(self.request(path)))

    def pages(self, path: str, key: str | None = None) -> list[Json]:
        """Read every page without following server-supplied URLs.

        :param path: Trusted list resource path, optionally with query parameters.
        :param key: Collection key for object-wrapped responses.
        :returns: All resource objects.
        """
        result: list[Json] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 1001):
            value = json.loads(
                self.request(f"{path}{separator}per_page=100&page={page}")
            )
            items = value if key is None else _object(value).get(key)
            if not isinstance(items, list):
                raise PublicationError("Invalid GitHub pagination response")
            result.extend(_object(item) for item in items)
            if len(items) < 100:
                return result
        raise PublicationError("GitHub pagination limit exceeded")


def _repository(value: object) -> tuple[int, str]:
    repo = _object(value)
    return _integer(repo.get("id")), _text(repo.get("full_name"))


def _snapshot(pr: Json) -> tuple[str, tuple[int, str], str, str, tuple[int, str], str]:
    head, base = _object(pr.get("head")), _object(pr.get("base"))
    return (
        _sha(head.get("sha")),
        _repository(head.get("repo")),
        _text(head.get("ref")),
        _sha(base.get("sha")),
        _repository(base.get("repo")),
        _text(base.get("ref")),
    )


def _validate_run(run: Json, repo: tuple[int, str]) -> None:
    if (
        run.get("name") != "firecrown-api-changes"
        or run.get("path") != ".github/workflows/api-changes.yml"
        or run.get("event") != "pull_request"
        or _repository(run.get("repository")) != repo
    ):
        raise PublicationError("Not a trusted upstream firecrown-api-changes PR run")
    _integer(run.get("id"))
    _integer(run.get("run_attempt"))
    _integer(run.get("workflow_id"))
    _integer(run.get("run_number"))
    _sha(run.get("head_sha"))
    _text(run.get("head_branch"))
    _repository(run.get("head_repository"))


def _resolve_pr(api: GitHub, root: str, run: Json, repo: tuple[int, str]) -> Json:
    candidates = run.get("pull_requests")
    if not isinstance(candidates, list):
        raise PublicationError("Missing run pull_requests array")
    if not candidates:
        candidates = api.pages(f"{root}/commits/{_sha(run['head_sha'])}/pulls")
    matches: list[Json] = []
    for candidate in candidates:
        association = _object(candidate)
        number = _integer(association.get("number"))
        pr = api.get(f"{root}/pulls/{number}")
        snapshot = _snapshot(pr)
        if (
            snapshot[:3]
            == (
                run["head_sha"],
                _repository(run["head_repository"]),
                run["head_branch"],
            )
            and snapshot[4] == repo
            and pr.get("state") == "open"
            and _integer(pr.get("number")) == number
        ):
            # Associations may reflect current PR state; recheck the resolved target.
            if "base" in association:
                base = _object(association["base"])
                if (base.get("sha"), base.get("ref")) != (snapshot[3], snapshot[5]):
                    raise StaleRun("PR base or target no longer matches the run")
            matches.append(pr)
    if len(matches) != 1:
        raise StaleRun("Run has no unique current, matching open PR")
    return matches[0]


def _artifact(data: bytes, expected: Json) -> tuple[str, str]:
    """Validate the ZIP and metadata before returning any report text."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != 2 or {item.filename for item in members} != {
            "api-changes.md",
            "metadata.json",
        }:
            raise PublicationError("Artifact must contain exactly the two report files")
        for item in members:
            limit = MAX_METADATA if item.filename == "metadata.json" else MAX_REPORT
            mode = item.external_attr >> 16
            if (
                item.file_size > limit
                or item.compress_size > limit
                or item.flag_bits & 1
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            ):
                raise PublicationError("Unsafe or oversized artifact member")
        metadata = _object(json.loads(archive.read("metadata.json")))
        _sha(metadata.get("base_sha"))
        _text(metadata.get("target"))
        for key, value in expected.items():
            if metadata.get(key) != value or type(metadata.get(key)) is not type(value):
                if key in ("target", "base_sha") and isinstance(metadata.get(key), str):
                    raise StaleRun("Artifact target or base no longer matches the PR")
                raise PublicationError(f"Artifact metadata mismatch: {key}")
        if metadata.get("status") != "completed":
            raise PublicationError(
                "API analysis failed; compatibility was not assessed"
            )
        verdict = _text(metadata.get("verdict"))
        _sha(metadata.get("baseline"))
        target = _text(expected["target"])
        suffix = (
            "This fails support-line policy."
            if re.fullmatch(r"v\d+_\d+_support", target)
            else (
                f"Informational because the target is `{target}`. "
                "Reviewer assessment required."
            )
        )
        if verdict != "No breaking API changes detected." and not re.fullmatch(
            r"Breaking API changes detected \([1-9]\d* findings\)\. "
            + re.escape(suffix),
            verdict,
        ):
            raise PublicationError("Invalid API policy verdict")
        return archive.read("api-changes.md").decode("utf-8"), verdict


def _fresh(api: GitHub, root: str, run: Json, pr: Json) -> None:
    """Revalidate attempts, superseding runs, and the PR immediately before a write."""
    runs = api.pages(
        f"{root}/actions/workflows/{run['workflow_id']}/runs"
        f"?event=pull_request&head_sha={run['head_sha']}",
        "workflow_runs",
    )
    for other in runs:
        if (
            other.get("workflow_id") == run["workflow_id"]
            and other.get("head_sha") == run["head_sha"]
            and other.get("head_branch") == run["head_branch"]
            and _repository(other.get("head_repository"))
            == _repository(run["head_repository"])
            and _integer(other.get("run_number")) > _integer(run["run_number"])
        ):
            associated = other.get("pull_requests")
            if not isinstance(associated, list):
                raise PublicationError("Invalid newer-run PR associations")
            if not associated or any(
                _object(item).get("number") == pr["number"] for item in associated
            ):
                raise StaleRun("A newer PR workflow run supersedes this report")
    latest = api.get(f"{root}/actions/runs/{run['id']}")
    _validate_run(latest, _repository(run["repository"]))
    if any(
        latest.get(key) != run[key]
        for key in ("run_attempt", "head_sha", "head_branch", "workflow_id", "status")
    ):
        raise StaleRun("Run identity or attempt changed before publication")
    current = api.get(f"{root}/pulls/{pr['number']}")
    if current.get("state") != "open" or _snapshot(current) != _snapshot(pr):
        raise StaleRun("PR head, base, or target changed before publication")


def resolve(api: GitHub, event: Json, repository: str) -> tuple[Json, Json]:
    """Establish upstream run and PR identity without reading untrusted artifacts.

    :param api: Trusted REST client, with an injected transport in tests.
    :param event: Trusted workflow_run webhook payload.
    :param repository: Trusted GITHUB_REPOSITORY identity.
    :returns: The validated upstream run and its unique current open PR.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository):
        raise PublicationError("Invalid GITHUB_REPOSITORY")
    repo = _repository(event.get("repository"))
    run = _object(event.get("workflow_run"))
    if (
        repo[1] != repository
        or event.get("action") != "completed"
        or run.get("status") != "completed"
    ):
        raise PublicationError("Expected an upstream completed workflow_run event")
    _validate_run(run, repo)
    root = f"/repos/{repository}"
    run_path = f"{root}/actions/runs/{run['id']}"
    attempt = api.get(f"{run_path}/attempts/{run['run_attempt']}")
    _validate_run(attempt, repo)
    for field in (
        "id",
        "run_attempt",
        "workflow_id",
        "run_number",
        "head_sha",
        "head_branch",
        "status",
    ):
        if attempt.get(field) != run.get(field):
            raise PublicationError(f"Run attempt identity mismatch: {field}")
    if _repository(attempt.get("head_repository")) != _repository(
        run.get("head_repository")
    ):
        raise PublicationError("Run attempt head repository mismatch")
    if api.get(run_path).get("run_attempt") != run["run_attempt"]:
        raise StaleRun("A newer run attempt supersedes this report")
    pr = _resolve_pr(api, root, attempt, repo)
    return run, pr


def resolve_in_run(
    api: GitHub, event: Json, repository: str, run_id: str, run_attempt: str
) -> tuple[Json, Json]:
    """Establish run and PR identity inside the still-running analysis workflow.

    Only a same-repository PR is accepted, because this mode executes publisher
    code from the PR branch with a token that can write comments.

    :param api: Trusted REST client, with an injected transport in tests.
    :param event: Trusted pull_request webhook payload.
    :param repository: Trusted GITHUB_REPOSITORY identity.
    :param run_id: Trusted GITHUB_RUN_ID of this running workflow.
    :param run_attempt: Trusted GITHUB_RUN_ATTEMPT of this running workflow.
    :returns: The validated current run and its unique current open PR.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository):
        raise PublicationError("Invalid GITHUB_REPOSITORY")
    repo = _repository(event.get("repository"))
    if repo[1] != repository:
        raise PublicationError("Event repository does not match GITHUB_REPOSITORY")
    root = f"/repos/{repository}"
    run = api.get(f"{root}/actions/runs/{int(run_id)}")
    _validate_run(run, repo)
    if run["run_attempt"] != int(run_attempt) or run.get("status") == "completed":
        raise StaleRun("The running attempt is not the current in-progress attempt")
    number = _integer(_object(event.get("pull_request")).get("number"))
    pr = api.get(f"{root}/pulls/{number}")
    snapshot = _snapshot(pr)
    if snapshot[1] != repo or snapshot[4] != repo:
        raise PublicationError("In-run publication requires a same-repository PR")
    if (
        snapshot[0] != run["head_sha"]
        or snapshot[1] != _repository(run["head_repository"])
        or snapshot[2] != run["head_branch"]
        or pr.get("state") != "open"
        or _integer(pr.get("number")) != number
    ):
        raise StaleRun("PR no longer matches the running analysis")
    return run, pr


def publish(
    api: GitHub, event: Json, repository: str, pr_number: str | None = None
) -> str:
    """Create or refresh the bot comment, rejecting stale and unsafe provenance.

    :param api: Trusted REST client, with an injected transport in tests.
    :param event: Trusted workflow_run webhook payload.
    :param repository: Trusted GITHUB_REPOSITORY identity.
    :param pr_number: Resolved concurrency identity from the read-only job.
    :returns: A brief successful publication summary.
    """
    run, pr = resolve(api, event, repository)
    if pr_number is not None and pr_number != str(pr["number"]):
        raise PublicationError("PR identity changed after concurrency resolution")
    return publish_resolved(api, repository, run, pr)


# The publication transaction stays cohesive across validation and the final write.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def publish_resolved(  # noqa: C901
    api: GitHub, repository: str, run: Json, pr: Json
) -> str:
    """Write the bot comment for an already validated run and PR identity.

    :param api: Trusted REST client, with an injected transport in tests.
    :param repository: Trusted GITHUB_REPOSITORY identity.
    :param run: Validated analysis run.
    :param pr: Validated current open PR.
    :returns: A brief successful publication summary.
    """
    root = f"/repos/{repository}"
    run_path = f"{root}/actions/runs/{run['id']}"
    snapshot = _snapshot(pr)
    expected: Json = {
        "schema": 1,
        "repository": repository,
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "pr_number": pr["number"],
        "head_sha": snapshot[0],
        "base_sha": snapshot[3],
        "target": snapshot[5],
    }
    run_url = (
        f"https://github.com/{repository}/actions/runs/{run['id']}"
        f"/attempts/{run['run_attempt']}"
    )
    failure: str | None = None
    try:
        jobs = api.pages(f"{run_path}/attempts/{run['run_attempt']}/jobs", "jobs")
        job_links = [
            f"[API job](https://github.com/{repository}/actions/runs/"
            f"{run['id']}/job/{_integer(job.get('id'))})"
            for job in jobs
            if job.get("name") == "PR public API changes"
        ][:1]
        artifacts = api.pages(f"{run_path}/artifacts", "artifacts")
        selected = [
            item
            for item in artifacts
            if item.get("name") == f"public-api-report-{run['id']}-{run['run_attempt']}"
        ]
        if len(selected) != 1 or selected[0].get("expired") is not False:
            raise PublicationError(
                "API report artifact is missing, expired, or ambiguous"
            )
        artifact = selected[0]
        if _integer(artifact.get("size_in_bytes")) > MAX_DOWNLOAD:
            raise PublicationError("Artifact exceeds download limit")
        report, verdict = _artifact(
            api.request(f"{root}/actions/artifacts/{_integer(artifact.get('id'))}/zip"),
            expected,
        )
        # Preserve code spans/signatures while preventing artifact-owned state markers.
        report = report.replace("<!--", "&lt;!--")
        text = f"**{html.escape(verdict, quote=False)}**\n\n" + report
    except StaleRun:
        raise
    except (
        PublicationError,
        OSError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
        zlib.error,
        HTTPException,
        EOFError,
    ) as exc:
        failure = str(exc)
        text = "**API analysis/reporting failed. Compatibility was not assessed.**"
        job_links = []
    prefix = (
        f"{MARKER}\n<!-- firecrown-api-run:{run['id']}:{run['run_attempt']} -->\n\n"
        f"PR revision: `{snapshot[0]}`  \n"
        f"Target: `{html.escape(snapshot[5], quote=False)}`\n\n"
    )
    label = "Workflow run" if failure else "Complete report artifact and workflow run"
    footer = f"\n\n[{label}]({run_url})"
    if job_links:
        footer += " | " + " | ".join(job_links)
    if len((prefix + text + footer).encode("utf-8")) > MAX_COMMENT:
        note = (
            "\n\n_Report truncated; the complete report is in the "
            "linked workflow artifact._"
        )
        budget = MAX_COMMENT - len((prefix + footer + note).encode("utf-8"))
        text = text.encode("utf-8")[:budget].decode("utf-8", errors="ignore") + note
    body = prefix + text + footer
    comments = api.pages(f"{root}/issues/{pr['number']}/comments")
    own = [
        item
        for item in comments
        if _object(item.get("user")).get("login") == "github-actions[bot]"
        and _object(item["user"]).get("type") == "Bot"
        and str(item.get("body", "")).startswith(MARKER + "\n")
    ]
    if len(own) > 1:
        raise PublicationError("Multiple publisher-owned comments found")
    if own:
        state = STATE.search(_text(own[0].get("body")))
        if state and tuple(map(int, state.groups())) > (run["id"], run["run_attempt"]):
            raise StaleRun("A newer publisher-owned comment already exists")
    _fresh(api, root, run, pr)
    if own:
        api.request(
            f"{root}/issues/comments/{_integer(own[0].get('id'))}",
            {"body": body},
            "PATCH",
        )
    else:
        api.request(f"{root}/issues/{pr['number']}/comments", {"body": body}, "POST")
    if failure is not None:
        raise PublicationError(f"Published compatibility not assessed: {failure}")
    return (
        f"Published public API report for PR #{pr['number']} "
        f"(run {run['id']}, attempt {run['run_attempt']})."
    )


def main(
    environment: Mapping[str, str] | None = None, transport: Transport = _http
) -> int:
    """Run the environment-driven trusted publisher and record its own outcome.

    :param environment: Trusted workflow environment, injectable for tests.
    :param transport: HTTP boundary, injectable without real network access.
    :returns: Zero on publication or stale skip; one on publication failure.
    """
    env = os.environ if environment is None else environment
    result = 0
    try:
        event = _object(
            json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        )
        api = GitHub(env["GITHUB_TOKEN"], transport)
        if env.get("API_REPORT_RESOLVE_ONLY") == "true":
            run, pr = resolve(api, event, env["GITHUB_REPOSITORY"])
            _fresh(api, f"/repos/{env['GITHUB_REPOSITORY']}", run, pr)
            with Path(env["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
                output.write(f"pr_number={_integer(pr['number'])}\n")
            summary = f"Resolved trusted PR #{pr['number']} for public API publication."
        elif env.get("API_REPORT_IN_RUN") == "true":
            run, pr = resolve_in_run(
                api,
                event,
                env["GITHUB_REPOSITORY"],
                env["GITHUB_RUN_ID"],
                env["GITHUB_RUN_ATTEMPT"],
            )
            summary = publish_resolved(api, env["GITHUB_REPOSITORY"], run, pr)
        else:
            summary = publish(
                api, event, env["GITHUB_REPOSITORY"], env.get("API_REPORT_PR_NUMBER")
            )
    except StaleRun as exc:
        summary = f"Skipped stale public API report: {exc}"
    except (
        PublicationError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        HTTPException,
    ) as exc:
        summary = f"Public API report publication failed: {exc}"
        result = 1
    print(summary, file=sys.stderr)
    try:
        with Path(env["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
            output.write(
                "## Public API Report Publisher\n\n" + html.escape(summary) + "\n"
            )
    except (OSError, KeyError) as exc:
        print(f"Cannot write publisher summary: {exc}", file=sys.stderr)
        result = 1
    return result


if __name__ == "__main__":
    sys.exit(main())
