# Making a new release

Firecrown release versions are derived from git tags through `setuptools-scm`.
This guide covers feature-line releases, maintenance releases, GitHub releases, and the conda-forge handoff.

## Feature-line Release `x.y.0`

1. Create a local branch named for the release, such as `prep-vx.y.0`, from `master`.
2. Update any code, documentation, dependency constraints, and tutorial content that belong in the release.
3. If `dependencies.yaml` contains a change from the previous release, then
4. run `make deps-sync` # This updates `environment`.yml and `pyproject.toml`.
5. run `make conda-lock`   # This updates the lockfiles from the updated `environment.yml`
6. Commit any changes.
7. `git push` the `prep-vx.y.0` branch
8. Open a PR targeting `master` and let CI run.
   You can use:

   ```
   gh pr create --base master --title "Prep vx.y.0" 
   ```

9. Verify that the CI passes, then merge the PR.
10. Check out a clean local copy of `master` at the merged release commit.
11. Run `make release-tag VERSION=x.y.0`.  # creates the tag and the long-term support branch
 Note that this runs many checks but not the full `pre-commit` suite.
12. Run `make install` to refresh the installed Firecrown metadata so later documentation builds display the new release version and for use in building the conda installation artifacts
13. Run `make release-sdist VERSION=x.y.0`.
14. Run `make release-verify-sdist VERSION=x.y.0`.
15. Run `make release-push VERSION=x.y.0`.  # pushes refs and tags to `origin`

## Maintenance Release `x.y.z` where `z > 0`

1. Create a local branch named for the maintenance release, such as `prep-v1.15.1`, from `vx_y_support`.
2. Update the code, documentation, and dependency constraints needed for the maintenance release.
3. If `dependencies.yaml` changes, run `make deps-sync` then `make conda-lock` (in that order: the lockfiles are solved from the environment the first step generates, and the validated pins are read back out of the lockfiles), and commit the regenerated files.
4. Commit and push the branch.
5. Open a PR targeting `vx_y_support` and let CI run.
6. Merge the PR after CI passes.
7. Check out a clean local copy of `vx_y_support` at the merged release commit.
8. Run `make release-tag VERSION=x.y.z`. # creates the new tag
   This reruns the fast release-specific checks and reuses the successful full check for the same `HEAD` and `VERSION` instead of rerunning `make pre-commit`.
9. Run `make install` to refresh the installed Firecrown metadata so later documentation builds display the new release version and for use in building the conda installation artifacts
10. Run `make release-sdist VERSION=x.y.z`.
11. Run `make release-verify-sdist VERSION=x.y.z`.
12. Run `make release-push VERSION=x.y.z`.  # pushes refs and tags to `origin`

For maintenance releases, `release-check` requires the checked-out branch to be `vx_y_support` and confirms that the support branch exists on `origin` before validation continues.

## Shared Validation and Tagging Behavior

Before running any release target, activate the `firecrown_developer` conda environment so the release tooling, documentation tools, and test dependencies come from the project developer environment.
Run `conda activate firecrown_developer` before invoking the release targets.
Ensure that GitHub CLI is installed and authenticated for `github.com`.
Run `gh auth status --hostname github.com` to confirm the current login.
If needed, run `gh auth login --hostname github.com --web` before continuing.

The `release-check` target first confirms that the active conda environment is `firecrown_developer`.
It then runs the release-specific validation for `VERSION=x.y.z`, including checkout cleanliness, version format, tag absence, support-branch checks, GitHub CLI authentication, and Python `build` availability.
It also runs `make pre-commit` once for the current `HEAD` and `VERSION` and records the successful result in `.git/`.
For `x.y.0` releases, it also confirms that the current branch is `master` and that the support branch name is available.
For maintenance releases, it confirms that the current branch is `vx_y_support` and that `HEAD` matches `origin/vx_y_support`.

The `release-tag` target reruns the fast release-specific checks, reuses the successful full check for the same `HEAD` and `VERSION`, and creates the annotated tag `vX.Y.Z` locally.
If the wrong conda environment is active, it fails immediately with instructions to activate `firecrown_developer`.
For `x.y.0` releases, it also creates `vx_y_support` locally.

The `release-sdist` target builds `dist/firecrown-X.Y.Z.tar.gz` from the tagged checkout.
It requires the local tag to exist and requires `HEAD` to match that tag.

The `release-verify-sdist` target installs the sdist into a temporary target directory and verifies both `importlib.metadata.version("firecrown")` and `firecrown.__version__` against `x.y.z`.

The `release-clean` target removes local release state.
Without `VERSION`, it removes `dist/` and `.git/firecrown-release/`.
With `VERSION=x.y.z`, it also deletes the local `vX.Y.Z` tag if present.
For `x.y.0` releases, it also deletes the local `vx_y_support` branch if it exists and is not the current branch.
It warns before force-deleting that local support branch.
This target never changes remote refs.

The `release-push` target reruns the sdist verification and then pushes the tag to `origin`.
For `x.y.0` releases, it also pushes `vx_y_support`.

The `release-verify-archive` target confirms that the GitHub auto-archive
(`/archive/vX.Y.Z.tar.gz`) is **not** a valid source for the conda-forge recipe.
It produces a `.git`-less tree from the tag using `git archive`, installs it
with `--no-deps --no-build-isolation`, and verifies that the installed version is
**not** `x.y.z` (because the auto-archive has no `PKG-INFO` and `setuptools-scm`
cannot determine the version).
Run `make release-verify-archive VERSION=x.y.z` after tagging to document this
constraint explicitly; a correct outcome prints a confirmation that the
auto-archive source is unsupported.

## Publish the GitHub Release

1. Run `make release-github VERSION=x.y.z`.
2. The target requires an authenticated `gh` session and fails with login instructions when authentication is missing.
3. The target rebuilds and verifies `dist/firecrown-X.Y.Z.tar.gz` before publication.
4. The target requires `vX.Y.Z` to be present on `origin` and fails with instructions to run `make release-push VERSION=x.y.z` when it is missing.
5. The target creates the GitHub release with generated notes, uploads the sdist as a release asset, and sets the latest flag from version ordering.

## Start the Conda-forge Handoff

### Required: Use the Release Sdist, not the GitHub Auto-archive

The conda-forge recipe `source.url` **must** point at the **release sdist asset**
uploaded to the GitHub release, not the GitHub auto-generated archive.

| URL pattern | Acceptable? |
| --- | --- |
| `…/releases/download/vX.Y.Z/firecrown-X.Y.Z.tar.gz` | **Yes** — contains `PKG-INFO` with the correct version |
| `…/archive/vX.Y.Z.tar.gz` | **No** — no `PKG-INFO`, no `.git`, `setuptools-scm` cannot resolve the version → `firecrown.__version__ == '0.0.0'` |

The `make release-conda-forge` target (below) computes the sdist sha256 and
emits the exact ready-to-paste `source` block so the correct URL is used
every time.

### Sync the Feedstock Fork before Each Handoff (Manual Git — no `make` target)

Contributions to the conda-forge feedstock go through your own fork of the feedstock repository, such as
`<github-username>/firecrown-feedstock` via a PR to
`conda-forge/firecrown-feedstock`.
Never push branches directly to the conda-forge feedstock (conda-forge policy).
There is no `make` target for fork lifecycle management; the steps below are
intentionally manual.

**One-time setup** (first time only):

```sh
# In your local firecrown-feedstock clone:
git remote add upstream https://github.com/conda-forge/firecrown-feedstock.git
```

**Before each handoff** — bring the fork up to date with conda-forge `main`:

```sh
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main

# Create the fix/update branch off the synced main
git checkout -b update-firecrown-x.y.z
```

### Handoff Steps

1. Run `make release-conda-forge VERSION=x.y.z`.
   The target requires:
   - An authenticated `gh` session; it fails with login instructions when missing.
   - A rebuilt and verified sdist `dist/firecrown-X.Y.Z.tar.gz`.
   - The GitHub release `vX.Y.Z` to exist; it fails with instructions to run
      `make release-github VERSION=x.y.z` when it is missing.

   The target computes the sha256 of the local sdist and files an issue in
   [conda-forge/firecrown-feedstock](https://github.com/conda-forge/firecrown-feedstock)
   with the exact `source.url` (release sdist) and `sha256` ready to paste into
   the recipe, along with a note that the auto-archive URL is not acceptable.

2. Sync the feedstock fork and create a branch (see above).
3. In `recipe/meta.yaml` on the update branch:
   - Set `source.url` to the release sdist URL from the issue body
     (`…/releases/download/vX.Y.Z/firecrown-X.Y.Z.tar.gz`).
   - Set `source.sha256` to the value from the issue body.
   - Ensure `setuptools-scm` is listed under `requirements.host`.
   - Ensure `test.commands` contains a version assertion:

     ```yaml
      commands:
        - python -c "import firecrown, importlib.metadata as md; assert firecrown.__version__ == '{{ version }}'; assert md.version('firecrown') == '{{ version }}'"
     ```

   - Update any dependency versions required for the release.
   - Bump `build.number` when re-publishing the same version (corrected build);
     reset to `0` for a new version.

4. Push the branch to your fork and open a PR to `conda-forge/firecrown-feedstock`.
5. Add the comment `@conda-forge-admin, please rerender` to the PR.
6. Wait for GitHub Actions to finish on all variants.
   The `test.commands` assertion will fail immediately on any build that does not
   produce the correct version.

7. Approve and merge the PR after CI passes.

The same conda-forge handoff applies to feature-line releases and maintenance releases.
