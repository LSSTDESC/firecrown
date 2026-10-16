# PR CI Signal Boundaries Specification

Status: local draft, prepared 2026-09-29. Tracker publication and the
`ready-for-agent` label are pending tracker configuration. This document does
not authorize implementation, commits, pushes, or CI reruns.

Decision authority: [ADR 0001](adr/0001-pr-ci-signal-boundaries.md) and the
confirmed CI design interview. Use the definitions in
[GLOSSARY.md](../GLOSSARY.md). Unconfirmed suggestions in the original handoff
are not requirements.

## Problem Statement

PR authors and reviewers need to distinguish three questions: whether the
repository's dependency declarations and committed artifacts agree, whether
the dependency environment can be freshly rebuilt today, and whether detected
public API changes comply with the target branch's policy. Current checks do
not consistently expose these distinctions. A green advisory job can conceal
important findings, while a live package-service failure can be confused with
a repository inconsistency.

### Current Behavior

These observations describe inspected source, not successful test execution:

- Local dependency tooling checks generated environment requirements and the
  project dependency array, but PR CI does not run an independent dependency
  consistency gate. The generator does not synchronize the project's Python
  requirement.
- Validated dependency constraints have a separate checking path. Currency
  checks rely on historical input digests rather than establishing lock
  compatibility. Lock aggregation does not validate every required package in
  every supported environment, and it ignores pip lock records.
- Supported locked environments currently span Python 3.12, 3.13, and 3.14 on
  `linux-64` and `osx-arm64`.
- A path-filtered, advisory Linux/Python 3.12 rebuild already has a 45-minute job
  ceiling. It lacks the agreed consistency prerequisite, complete relevant-path
  coverage, separate phase budgets, and explicit outcome reporting. Logs are
  uploaded only after failure, and summaries can name an artifact that was not
  successfully published.
- Static API analysis already makes detected breaks informational on `master`
  and failing on support targets. Its PR baseline is the merge base. Existing
  CLI tests cover these policies, public exports, and analysis failures.
- API summaries do not clearly identify the policy verdict and compared
  revision. A summary-publication error can be masked by the checker's successful
  exit status; setup and fetch failures can occur before report generation.
- Existing runtime CI uses locked environments. Neither the PR nor nightly
  fresh rebuild establishes Firecrown runtime compatibility.

## Solution

Expose three separate contracts to reviewers:

1. A mandatory, independent dependency consistency gate on every PR establishes
   dependency-list consistency, Python-requirement agreement, lock compatibility,
   and reproducibility of validated constraints from committed locks.
2. A bounded, advisory rebuild canary samples current environment rebuildability
   on Linux/Python 3.12, with explicit outcomes and available diagnostic evidence.
3. Public API reporting states the target-dependent policy verdict separately
   from whether analysis and reporting succeeded.

No one signal establishes overall PR safety or runtime compatibility. Review
obligations remain explicit even when a job is technically non-blocking.

## User Stories

1. As a PR author, I want dependency consistency checked on every PR, so that
   unrelated-looking changes cannot silently bypass repository consistency.
2. As a reviewer, I want generated lists and Python requirements checked against
   the dependency manifest, so that conflicting declarations block acceptance.
3. As a maintainer, I want applicable direct runtime, workaround, and development
   requirements checked, so that development environments are not overlooked.
4. As a maintainer, I want every supported Python/platform combination validated,
   so that a partial collection of compatible locks is not mistaken for coverage.
5. As a contributor, I want conda/pip mappings and version semantics respected,
   so that valid selections pass and incompatible selections fail correctly.
6. As a maintainer, I want compatible older locks accepted, so that consistency
   does not depend on unnecessary fresh solves or newest package selections.
7. As a reviewer, I want validated constraints reproduced from committed locks,
   so that they are grounded in the selections the repository actually records.
8. As a PR author, I want consistency failures to identify the affected input or
   environment, so that I can correct the inconsistency without guessing.
9. As a reviewer, I want conservative routing of dependency-related edits, so
   that relevant changes receive the advisory rebuild signal.
10. As a reviewer, I want routing errors distinguished from irrelevant changes,
    so that a failed decision is not presented as an intentional skip.
11. As a PR author, I want inconsistent inputs to skip rebuilding with an
    explanation, so that live-resolution results do not distract from a failed
    consistency gate.
12. As a contributor, I want transient package-service failures kept advisory,
    so that they are not mislabeled as internal repository inconsistencies.
13. As a reviewer, I want success, setup failure, rebuild failure, timeout, and
    skip distinguished, so that I know what the canary established.
14. As a maintainer, I want bounded setup and rebuild phases, so that diagnostics
    have a best-effort opportunity to publish before the job ceiling.
15. As a reviewer, I want revision and environment identity with available logs,
    so that the evidence is attributable to the attempt being reviewed.
16. As a reviewer, I want only real artifact links and visible publication
    failures, so that missing evidence is not disguised as success.
17. As a PR author, I want my obligation to investigate canary failures stated,
    so that technically advisory does not mean safe to ignore.
18. As a reviewer, I want unresolved canary risk explicitly accepted before
    merge, so that the merge decision remains informed.
19. As a reviewer of a `master` PR, I want detected API breaks prominently shown
    as informational, so that a green job does not hide release-note work.
20. As a support-line maintainer, I want genuine detected public API breaks
    rejected, so that the support-line compatibility policy remains enforced.
21. As a maintainer, I want confirmed false positives handled per finding and
    revision with rationale and reviewer confirmation, so that an exception does
    not become a global suppression or permission for real breaks.
22. As a reviewer, I want analysis/reporting failures separated from no detected
    breaks, so that missing evidence cannot be interpreted as compatibility.
23. As a reviewer, I want the API target, baseline, revision, and static-analysis
    limits visible, so that I understand the scope of the comparison.

## Implementation Decisions

- Extend the existing dependency synchronization/checking interface rather than
  introducing a second manifest authority. Consistency checking must not require
  a fresh package solve or refresh committed artifacts to make them pass.
- Validate direct requirements applicable to each supported environment using
  the relevant conda/pip identity mappings and version semantics. Do not build
  an independent transitive solver; conda-lock owns transitive solving.
- Derive validated constraints from committed locks. A derived range does not
  assert that every version within it was runtime-tested. Historical input
  digest differences alone do not invalidate compatible locks.
- Modify PR orchestration to expose an independent mandatory consistency gate
  and an advisory canary with a consistency prerequisite. Routing must include
  dependency declarations, locks, generators, and relevant CI configuration.
- Preserve the single Linux/Python 3.12 canary sample. Apply a 5-minute setup
  maximum and a 35-minute rebuild maximum within a 45-minute job ceiling.
  The remaining nominal 5 minutes are a best-effort termination/reporting reserve.
- Preserve available setup and solve/install logs for successful and unsuccessful
  attempts. Publish links only after confirming the corresponding artifacts
  exist. Treat diagnostic-publication failure as visible incomplete reporting,
  without silently turning the advisory canary into a mandatory rebuild gate.
- Preserve existing static API detection, public-path scope, and PR baseline
  selection. Improve API report generation and workflow failure handling rather
  than expanding compatibility analysis.
- Keep support-line false-positive exceptions a documented review policy: the
  job remains failed, and no GitHub bypass mechanism is assumed or configured.
- Update directly affected contributor guidance to match these contracts.
  Do not use this work to expand release or nightly policy.

## Acceptance Criteria

### Dependency Consistency

- **D1:** Every PR invokes the independent consistency gate, regardless of
  canary path relevance. A consistency failure is a mandatory failure, not an
  advisory canary result.
- **D2:** Fixture drift in each generated dependency list, the project dependency
  requirements, or the declared Python requirement produces a failing check;
  matching inputs pass.
- **D3:** For each supported Python/platform combination, applicable direct
  runtime, workaround, and development requirements are checked against locked
  selections. Missing required packages, incompatible versions, or missing
  supported combinations fail with an attributable diagnostic.
- **D4:** Tests include conda/pip name mappings and version constraints whose
  interpretation depends on the respective ecosystem's semantics. Applicable
  pip selections cannot be silently ignored.
- **D5:** Compatible older locks pass without a fresh solve, including when
  historical input digests differ. Incompatible selections still fail even if
  a digest appears current.
- **D6:** Validated constraints matching regeneration from committed locks pass;
  differing constraints fail. Checking does not rewrite committed inputs or
  outputs and does not independently solve transitive dependencies.

### PR Routing And Gating

- **R1:** Representative edits to dependency declarations, Python/project
  requirements, locks, generators, build/check entry points, and relevant PR CI
  configuration route to the canary. Conservative extra runs are acceptable.
- **R2:** An irrelevant change can skip rebuilding but not consistency checking.
  A routing error fails visibly and cannot appear as an irrelevant-change skip.
- **R3:** Failed consistency prevents the fresh rebuild and yields an explained
  skip. The failed mandatory gate remains visible independently of that skip.
- **R4:** With consistent inputs and a relevant change, the canary runs the
  Linux/Python 3.12 fresh resolve/install sample. Its rebuild failure remains
  technically non-blocking; author investigation and reviewer acceptance of
  unresolved risk remain required policy.

### Canary Evidence And Time Budgets

- **C1:** Controlled success, setup failure, rebuild failure, phase timeout, and
  intentional skip each produce a distinguishable outcome. Failure/skip reports
  include a reason, and reports identify the tested revision and platform/Python
  (or the intended identity for attempts that never reached execution).
- **C2:** Setup cannot consume more than its 5-minute budget and rebuilding
  cannot consume more than its 35-minute budget. The job ceiling remains
  45 minutes; timeout handling attempts termination and diagnostic publication
  within the remaining reserve.
- **C3:** Available setup and solve/install logs are preserved for both successful
  and failed attempts. Early setup failure does not assume a rebuild log exists.
  An artifact link appears only if that artifact was actually published.
- **C4:** A log/artifact/summary publication failure is visible through an
  available CI failure channel and does not yield an unqualified success report.
  It does not falsely claim that an otherwise successful solve/install failed.
- **C5:** Summaries state the sample's limits: no broader platform/Python claim,
  no Firecrown runtime compatibility claim, and no comparison with locked versions.
  Documentation explicitly acknowledges that cancellation or runner loss can
  prevent diagnostics from being published at all.

### Public API Outcome Reporting

- **A1:** A completed comparison with no detected breaks prominently reports
  that result without claiming unchanged behavior or complete compatibility.
- **A2:** A completed comparison detecting a break on `master` remains successful
  but prominently reports an informational break requiring reviewer assessment.
  Intentional breaks must be identified for release notes.
- **A3:** The same detected break on a support target produces a failing policy
  verdict and publishes the findings when reporting is available. Tests include
  a public reexport removal equivalent to removing
  `firecrown.likelihood.TwoPoint`; its private implementation location does not
  exempt the public import path.
- **A4:** Successful comparisons identify the compared revision, target branch,
  and resolved baseline, and state static-analysis limits. Failures identify
  available identities without fabricating an unresolved baseline.
- **A5:** Target-fetch, source-analysis, and report-publication failures are
  distinguishable from no detected breaks and informational findings. A
  successful checker cannot mask failed report publication. If comparison itself
  failed, reporting says compatibility was not assessed.
- **A6:** Documented support-line exceptions require a confirmed false positive,
  finding/revision-specific rationale, and reviewer confirmation. The job stays
  failed. Genuine support breaks remain prohibited; exceptions neither suppress
  future findings nor imply an available GitHub bypass.
- **A7:** Existing tests for public exports, signatures, private implementation
  moves, and PR baseline selection retain their meaning. No new semantic or
  external-dependency signature analysis is introduced.

## Testing Decisions

The user confirmed these testing boundaries before this draft was written:

- Prefer externally observable behavior: exit status, generated/check results,
  report contents, routing decisions, and published evidence. Avoid assertions
  about internal helper layout or exact prose where a semantic outcome suffices.
- Extend existing API CLI tests using real temporary Git repositories and small
  Python packages. Existing support-target, master-target, no-break, and invalid
  baseline tests provide prior art; keep lower-level detection tests where they
  already protect the static-analysis contract.
- Exercise the dependency checker through its CLI using fixture manifests,
  generated declarations, validated constraints, and locks. Cover all supported
  environment identities and both passing and failing cases for D2-D6. Assert
  that checking leaves fixture inputs unchanged.
- Test workflow routing, orchestration, reporting, and budgets with controlled
  command outcomes rather than live package solves. This is a proposed test
  boundary, not an assertion that a workflow test harness already exists.
- Keep any needed workflow test seam at the orchestration boundary, exercising
  the logic CI actually uses rather than a separately implemented simulation.
  Test phase timeout handling without waiting 35 minutes, and verify configured
  production budgets separately.
- Include successful and failed artifact publication, failure before log
  creation, API report-publication failure after checker success, and routing
  failure. Check classification and available evidence rather than asserting
  impossible publication guarantees after runner loss.
- Human risk acceptance, release-note intent, and false-positive exception
  rationale are review obligations, not facts an automated test can establish.
  Verify their documentation and retain explicit reviewer responsibility.

### Local macOS Development and Verification

Discussion recorded 2026-09-30 for colleague review, following approval of the
13-ticket breakdown in
[the local ticket directory](../.scratch/pr-ci-signal-boundaries/issues/).
This records the intended development and verification boundary, not evidence
that the implementation or its local test infrastructure already exists.

Development and automated acceptance testing for this batch should be possible
on a macOS laptop without GitHub CI runs. GitHub-hosted execution is a separate
integration-validation step, not a prerequisite for implementation or local
acceptance testing. Local verification does not establish hosted CI correctness.

| Ticket area | Local verification approach |
| --- | --- |
| Dependency declarations and locks (01-04) | Exercise the real checker with fixture manifests, declarations, validated constraints, and lockfiles. No fresh solve or installation of all supported environments is required. |
| PR gating and canary (05-09) | Exercise production orchestration with controlled success, failure, timeout, missing-log, and publication outcomes; check workflow configuration and wiring separately. |
| API CLI (10-11) | Use temporary local Git repositories and small Python packages, without a GitHub connection. |
| API workflow failure handling (12-13) | Exercise production failure handling with controlled outcomes at GitHub-facing boundaries, including failures before comparison and during publication. |

Linux lockfiles can be inspected and validated as data on macOS; that does not
require running Linux or prove those environments install successfully. Timeout
tests use short deadlines or controlled timing rather than waiting 35 minutes,
while separately verifying the production 5/35/45-minute budgets.

#### Implementation Requirements for Local Tests

- Introduce small testable entry points where needed. CI must invoke the same
  orchestration logic exercised locally, not a separately implemented simulation.
- Make GitHub-facing operations, including artifact publication, replaceable at
  test boundaries so local tests can supply controlled outcomes.
- Keep Linux-specific execution details from requiring Linux for the ordinary
  local acceptance suite. Any platform-specific behavior not exercised locally
  must be identified as an integration-validation gap.
- Isolate tooling tests from scientific-dependency imports in the existing test
  configuration where necessary. This work belongs within the relevant tickets;
  it is not a claim that a lightweight harness is already available.

#### Limits and Completion Reporting

Local tests alone cannot fully verify GitHub Actions scheduling, permissions,
job-status propagation, cancellation behavior, third-party actions on hosted
runners, or actual artifact upload and summary publication. They also do not
establish that the real Linux/Python 3.12 fresh resolve/install currently succeeds
against live package services. Required-check repository settings remain a
separate, explicitly authorized concern.

Completion reports must distinguish locally verified behavior from behavior
verified on GitHub and list remaining hosted or platform-specific validation
gaps. A passing local suite is not evidence of a successful hosted CI run.

The current prohibition on CI runs remains in force. Recording this discussion
does not authorize implementation, test execution, commits, pushes, remote
publication, or repository-setting changes. This record and the tickets remain
uncommitted and local to this checkout until separately authorized otherwise.

## Out of Scope

- Firecrown runtime tests in freshly resolved environments.
- Resolved-version difference reports or comparisons with locked environments.
- Broader Python/platform coverage for PR rebuilds.
- Nightly diagnostic parity, nightly policy changes, and feedstock publication.
- An independent transitive dependency solver or mandatory fresh lock solves
  solely because an input digest changed.
- Semantic API compatibility analysis, external-dependency signature analysis,
  or expansion of the existing public-path definition.
- Global suppression of support-line findings, permission for genuine support
  breaks, and configuration or assumptions about GitHub bypass mechanisms.
- Guarantees of diagnostic publication after cancellation or runner loss.
- Implementation, ticket creation, commits, pushes, repository-setting changes,
  and CI reruns as part of drafting this specification.

## Further Notes

The candidate areas for later ticketing are dependency consistency, PR routing
and gating, canary evidence and time budgets, and API outcome reporting. These
are not a finalized ticket split or dependency graph. A subsequent `/to-tickets`
step should define tracer-bullet tickets and explicit blocking relationships.

The mandatory gate is the intended merge policy. Workflow implementation alone
must not be represented as proof that repository required-check settings have
been configured; any necessary settings work needs separate authorization.

Configure the project tracker before publishing this local draft and applying
`ready-for-agent`. No tracker issue or implementation ticket has been created
by this drafting step.
