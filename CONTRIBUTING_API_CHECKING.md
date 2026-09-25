# Checking Firecrown's public API

Use the API change report to review changes to Firecrown's public Python import paths before merging or releasing. The definition of a public API path and the meaning of a support line are in [CONTEXT.md](CONTEXT.md). The comparison uses Griffe to inspect the current checkout against a Git baseline. Run these commands from the repository root with the developer environment active (see [CONTRIBUTING.md](CONTRIBUTING.md)); the environment provides Griffe.

## Check unreleased support-line work

On the checked-out `vx_y_support` branch, run:

```sh
make api-support-check
```

The baseline is the highest numeric stable `vX.Y.Z` tag on that branch's support line, not necessarily the immediately preceding commit. The tag must be an ancestor of `HEAD`. The report compares it with the working checkout, including uncommitted source changes. If `HEAD` is the tag's commit, the tool reports no changes without inspecting the checkout; run it on a later commit to assess work after the tag. It errors if the current branch is not named `vx_y_support`, if no matching release tag exists, or if the latest tag is not an ancestor.

Breaking findings make this target exit with status 1. Review each finding and fix unintended breaks before merging maintenance work. An addition alone does not fail the check.

## Draft API sections for a release

After the intended release changes are in the checkout, generate a Markdown draft using the proposed version **without** the `v` prefix:

```sh
make api-release-notes VERSION=1.17.0 OUTPUT=api-notes.md
make api-release-notes VERSION=1.16.1 OUTPUT=api-notes.md
```

Use the first form for a new `x.y.0` release and the second for a maintenance `x.y.z` release. For `x.y.0`, the baseline is the highest stable release tag with a numerically lower version, potentially on a different support line. For `x.y.z` where `z > 0`, it is the highest lower stable tag on the same `x.y` support line. Prerelease tags are excluded. The new side is the working checkout in both cases. Missing eligible tags or an invalid version produce an error.

`OUTPUT` is optional for all three Make targets: omit it to print the report to the terminal, or set it to write a Markdown file (creating parent directories if needed). Review and incorporate the draft into release notes as appropriate; it is not published automatically. Keep the output outside `dist/`, which release packaging clears. For the full release procedure, see [Making_a_new_release.md](Making_a_new_release.md).

Release-note generation reports breaks but does **not** fail merely because it finds them. A maintenance release should also pass the support-line check on its support branch before release.

## Check a pull request

To preview changes locally against a target branch available in your repository, run:

```sh
make api-pr-report BASE=master
make api-pr-report BASE=v1_16_support OUTPUT=api-pr-notes.md
```

This compares the PR checkout with the merge base of `HEAD` and `BASE`, so it reports changes introduced by the PR rather than all changes since a release tag. `BASE` must resolve to the intended target branch or a fetched target ref; ensure it is up to date. The Make target passes `BASE` as both the comparison ref and the policy target. For a remote-tracking ref, use the underlying tool to name the support target separately, for example:

```sh
python tools/api_changes.py pr --base refs/remotes/origin/v1_16_support --target v1_16_support
```

On a GitHub PR, the **PR public API changes** job fetches the target branch, compares the PR head with their merge base, and puts the Markdown report in the job summary. Open that job's summary to inspect the findings; if comparison itself fails, consult its logs. PRs targeting `vx_y_support` fail this job when breaking changes are detected. PRs targeting `master` (or another non-support branch) still show breaks, but these findings are informational and do not by themselves fail the job. The hosted PR job and its summary publication have not yet been validated end to end; see [API_CHECKING_FACILITY_TESTING.md](API_CHECKING_FACILITY_TESTING.md).

## Read and act on the report

The report heading names the old Git ref and `HEAD`. **Executive summary** counts detected breaking-change categories, or says no breaks were detected. **Breaking changes** lists public paths and Griffe's explanations, including removals, changes to explicit `__all__` exports, and incompatible signature changes. **New public API** groups newly visible public paths by their parent path (a module or class). An empty section says `None.`

Exit status 0 means the comparison succeeded and the policy did not reject the findings; it does not mean the API is unchanged. Status 1 means detected breaks violate support-line policy (support checks and support-targeting PR checks). Status 2 means the comparison could not run, for example because a required ref or tag was missing. Inspect the report even when a command succeeds, and distinguish a comparison error from a policy failure.

This is a static change detector, **not** a guarantee of semantic compatibility or an exhaustive account of runtime behavior. In particular, Griffe can flag a changed source expression such as a `typing.cast(...)` wrapper even when the runtime value is unchanged. Investigate findings against the actual public behavior before treating them as release-note claims or changing code to silence them. Conversely, no detected break does not prove existing callers will keep working. Validation details and known limits, including the untested tagged-`HEAD` shortcut, are recorded in [API_CHECKING_FACILITY_TESTING.md](API_CHECKING_FACILITY_TESTING.md).
