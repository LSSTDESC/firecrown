"""Workflow configuration tests for API analysis and publication trust boundaries."""

from pathlib import Path

import yaml


def test_api_publisher_workflow_trust_boundary() -> None:
    """Keep privileged publication automatic and separate from PR execution."""
    root = Path(__file__).resolve().parents[1]
    analysis = yaml.safe_load((root / ".github/workflows/api-changes.yml").read_text())
    full_ci = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    publisher = yaml.safe_load(
        (root / ".github/workflows/publish-api-report.yml").read_text()
    )
    # PyYAML's YAML 1.1 parser reads the Actions 'on' key as True.
    assert publisher[True] == {
        "workflow_run": {
            "workflows": ["firecrown-api-changes"],
            "types": ["completed"],
        }
    }
    assert analysis["name"] == "firecrown-api-changes"
    assert analysis[True] == {"pull_request": None}
    assert "api-changes" not in full_ci["jobs"]
    assert "pull_request_target" not in analysis[True]
    job = analysis["jobs"]["api-changes"]
    assert job["permissions"] == {"contents": "read"}
    artifact = next(
        step for step in job["steps"] if "upload-artifact" in step.get("uses", "")
    )
    assert artifact["if"] == "${{ always() }}"
    assert "github.run_attempt" in artifact["with"]["name"]
    assert publisher["permissions"] == {}
    job = publisher["jobs"]["publish"]
    assert job["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "write",
    }
    resolve = publisher["jobs"]["resolve"]
    assert "conclusion" not in resolve["if"]
    assert resolve["permissions"]["pull-requests"] == "read"
    assert resolve["steps"][-1]["env"]["API_REPORT_RESOLVE_ONLY"] == "true"
    assert job["concurrency"]["cancel-in-progress"] is False
    assert "needs.resolve.outputs.pr_number" in job["concurrency"]["group"]
    assert job["needs"] == "resolve"
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["sparse-checkout"] == "tools/publish_api_report.py"
    assert resolve["steps"][0] == checkout
    assert job["steps"][-1]["run"] == "python tools/publish_api_report.py"


def test_same_repository_pr_publishes_from_its_own_workflow() -> None:
    """Publish for same-repository PRs in-run while keeping analysis read-only."""
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/api-changes.yml").read_text())
    publisher = yaml.safe_load(
        (root / ".github/workflows/publish-api-report.yml").read_text()
    )
    analysis = workflow["jobs"]["api-changes"]
    job = workflow["jobs"]["publish-same-repository"]
    assert analysis["permissions"] == {"contents": "read"}
    assert job["needs"] == "api-changes"
    assert "always()" in job["if"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in job["if"]
    )
    # Dependabot's pull_request token is read-only; the default-branch publisher
    # covers those PRs instead of this job failing with a permission error.
    assert "github.actor != 'dependabot[bot]'" in job["if"]
    assert job["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "write",
    }
    # One writer per PR across the in-run and default-branch publishers.
    assert job["concurrency"]["cancel-in-progress"] is False
    assert (
        job["concurrency"]["group"] == "public-api-report-${{ github.repository }}-"
        "${{ github.event.pull_request.number }}"
    )
    assert (
        publisher["jobs"]["publish"]["concurrency"]["group"]
        == "public-api-report-${{ github.repository }}-"
        "${{ needs.resolve.outputs.pr_number }}"
    )
    last = job["steps"][-1]
    assert last["run"] == "python tools/publish_api_report.py"
    assert last["env"]["API_REPORT_IN_RUN"] == "true"
    assert last["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert job["steps"][0]["with"]["persist-credentials"] is False
