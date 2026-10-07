# Separate PR dependency consistency, rebuildability, and API policy

Accepted in the CI design interview on 2026-09-29. This records agreed intent,
not completed implementation or authorization to implement it. Terminology is
defined in [GLOASSARY.md](../../GLOSSARY.md).

Local dependency consistency must block merging, while live environment
rebuildability remains a reviewed advisory signal. Package repositories can
change or become unavailable independently of a PR, so a fresh-install failure
must not be confused with an internally inconsistent repository. Public API
analysis is a third signal, governed by the target branch's compatibility policy;
none of these signals alone establishes runtime compatibility.

## Dependency consistency

- Run an independent consistency gate on every PR. Generated dependency lists
  and Python requirements must agree with the authoritative dependency manifest.
- Check applicable direct runtime, workaround, and development requirements
  against locked selections for every supported Python/platform combination.
  Respect conda/pip package mappings and their respective version semantics.
  Missing required packages, incompatible selections, or missing supported
  combinations fail the gate.
- Validated constraints must be reproducible from the committed locks. This
  does not claim that every version within a derived range was runtime-tested.
- Compatible older locks remain valid. Changed historical input digests alone
  do not invalidate them; neither a fresh solve nor newest-available selections
  are required. This deliberately differs from the current digest-based check.
- Do not implement an independent transitive dependency solver. Transitive
  solving belongs to conda-lock; existing CI supplies installation/runtime checks.

## PR rebuild canary

- Retain a Linux/Python 3.12 sample that freshly resolves and installs the
  dependency environment. Success says nothing about other supported combinations,
  Firecrown runtime behavior, or differences from locked versions.
- Conservatively route dependency declarations, locks, generators, and relevant
  CI configuration changes to the canary. Prefer unnecessary runs over omitted
  relevant edits. Routing errors fail visibly rather than masquerading as
  irrelevant changes. Inconsistent inputs skip the canary with an explanation.
- Keep the canary technically non-blocking. The PR author investigates failures;
  the reviewer must accept unresolved risk before merge. An advisory failure is
  not permission to ignore the result.
- Summaries distinguish success, setup failure, rebuild failure, timeout, and
  skip. Identify the tested revision, platform/Python, and failure or skip reason.
  Preserve available setup and solve/install logs for successful and unsuccessful
  attempts, and link only artifacts that actually exist. Diagnostic publication
  failure must be visible rather than reported as unqualified success.
- Budget at most 5 minutes for setup and 35 minutes for rebuilding within a
  45-minute job ceiling, leaving a nominal 5-minute termination/reporting reserve.
  Cancellation or runner loss can prevent publication; the reserve is best effort,
  not a guarantee of diagnostics or successful rebuilding.

## Public API analysis

- Retain the current public-path definition and static detection scope. Do not
  expand into semantic compatibility or external-dependency signature analysis.
- Detected breaks reject support-targeting PRs. On master they are informational:
  reviewers assess findings, and intentional breaks are identified for release
  notes. Analysis errors remain failures.
- A confirmed support-line false positive may receive a maintainer exception
  specific to the finding and revision, with technical rationale and reviewer
  confirmation. The job remains failed. This is neither a global suppression nor
  permission for genuine support-line breaks; it does not assume or configure a
  GitHub bypass mechanism.
- Prominently distinguish no detected breaks, informational breaks, support-policy
  rejection, and analysis/reporting failure. Identify the compared revision,
  target, and baseline, and state the static-analysis limits. Successful analysis
  with an unpublished report is a reporting failure, not an invisible success.

### Example: interpreting the API result

Firecrown exports `TwoPoint` through the public
[`firecrown.likelihood` module](../../firecrown/likelihood/__init__.py), even
though its implementation lives in the private `_two_point` module. The public
import path protected by the API policy is:

```python
from firecrown.likelihood import TwoPoint
```

Suppose a PR removes that public export. This is a hypothetical change, not a
finding about the PR that introduced API checking. The summary should make the
following distinctions clear:

| Situation | What the summary tells the reviewer |
| --- | --- |
| The checker finds no breaking changes | "No breaking API changes detected." This does not promise that all behavior is unchanged. |
| The PR removes `firecrown.likelihood.TwoPoint` and targets `master` | "Breaking API change detected. Informational because the target is `master`. Reviewer assessment required." |
| The same removal targets `v1_16_support` | "Breaking API change detected. This fails support-line policy." |
| CI cannot fetch the target branch or analyze the source | "API comparison failed. Compatibility was not assessed." |

A green API job can therefore mean either "no breaks found" or "breaks found,
but allowed on this target branch." The summary must tell reviewers which one
occurred and identify the PR commit, target branch, and baseline commit. If the
comparison succeeds but its report cannot be published, reviewers should see a
reporting failure rather than a green check with no findings to inspect.

## Boundaries

This effort is PR-only. Fresh-environment runtime tests, resolved-version
comparison reports, broader PR rebuild coverage, nightly diagnostic parity, and
feedstock publication are deferred. Nightly's existing fresh rebuilds do not
close the fresh-runtime-testing gap: its runtime tests use locked environments.

These limits keep routine PR checks bounded and their meanings explicit. Future
implementation must honor the separate contracts rather than interpreting a
green canary or informational API job as a general statement of PR safety.
