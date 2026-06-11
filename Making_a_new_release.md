# Making a new release

Firecrown release versions are derived from git tags through `setuptools-scm`.
This guide covers feature-line releases, maintenance releases, GitHub releases, and the conda-forge handoff.

## Feature-line release `x.y.0`

1. Create a local branch named for the release, such as `prep-v1.15.0`, from `master`.
2. Update any code, documentation, dependency constraints, and tutorial content that belong in the release.
3. If `environment.yml` changes, run `make conda-lock` and commit the regenerated lockfiles.
4. Commit and push the branch.
5. Open a PR targeting `master` and let CI run.
6. Merge the PR after CI passes.
7. Check out a clean local copy of `master` at the merged release commit.
8. Run `make release-tag VERSION=x.y.0`.
   This reruns the fast release-specific checks and reuses the successful full check for the same `HEAD` and `VERSION` instead of rerunning `make pre-commit`.
9. Run `make release-sdist VERSION=x.y.0`.
10. Run `make release-verify-sdist VERSION=x.y.0`.
11. Run `make release-push VERSION=x.y.0`.

For `x.y.0` releases, `release-tag` creates the annotated tag `vX.Y.0` locally and creates the support branch `vx_y_support` locally from the same release commit.
After the sdist is verified, `release-push` pushes both refs to `origin`.

## Maintenance release `x.y.z` where `z > 0`

1. Create a local branch named for the maintenance release, such as `prep-v1.15.1`, from `vx_y_support`.
2. Update the code, documentation, and dependency constraints needed for the maintenance release.
3. If `environment.yml` changes, run `make conda-lock` and commit the regenerated lockfiles.
4. Commit and push the branch.
5. Open a PR targeting `vx_y_support` and let CI run.
6. Merge the PR after CI passes.
7. Check out a clean local copy of `vx_y_support` at the merged release commit.
8. Run `make release-tag VERSION=x.y.z`.
   This reruns the fast release-specific checks and reuses the successful full check for the same `HEAD` and `VERSION` instead of rerunning `make pre-commit`.
9. Run `make release-sdist VERSION=x.y.z`.
10. Run `make release-verify-sdist VERSION=x.y.z`.
11. Run `make release-push VERSION=x.y.z`.

For maintenance releases, `release-check` requires the checked-out branch to be `vx_y_support` and confirms that the support branch exists on `origin` before validation continues.

## Shared validation and tagging behavior

Before running any release target, activate the `firecrown_developer` conda environment so the release tooling, documentation tools, and test dependencies come from the project developer environment.
Run `conda activate firecrown_developer` before invoking the release targets.
Ensure that GitHub CLI is installed and authenticated for `github.com`.
Run `gh auth status --hostname github.com` to confirm the current login.
If needed, run `gh auth login --hostname github.com --web` before continuing.

The `release-check` target first confirms that the active conda environment is `firecrown_developer`.
It then runs the release-specific validation for `VERSION=x.y.z`, including checkout cleanliness, version format, tag absence, support-branch checks, GitHub CLI authentication, and Python `build` availability.
It also runs `make pre-commit` once for the current `HEAD` and `VERSION` and records the successful result in `.git/`.
For `x.y.0` releases, it also confirms that the support branch name is available.
For maintenance releases, it confirms that the current branch is `vx_y_support`.

The `release-tag` target reruns the fast release-specific checks, reuses the successful full check for the same `HEAD` and `VERSION`, and creates the annotated tag `vX.Y.Z` locally.
If the wrong conda environment is active, it fails immediately with instructions to activate `firecrown_developer`.
For `x.y.0` releases, it also creates `vx_y_support` locally.

The `release-sdist` target builds `dist/firecrown-X.Y.Z.tar.gz` from the tagged checkout.
It requires the local tag to exist and requires `HEAD` to match that tag.

The `release-verify-sdist` target installs the sdist into a temporary target directory and verifies both `importlib.metadata.version("firecrown")` and `firecrown.__version__` against `x.y.z`.

The `release-push` target reruns the sdist verification and then pushes the tag to `origin`.
For `x.y.0` releases, it also pushes `vx_y_support`.

## Publish the GitHub release

1. Run `make release-github VERSION=x.y.z`.
2. The target requires an authenticated `gh` session and fails with login instructions when authentication is missing.
3. The target requires the verified sdist `dist/firecrown-X.Y.Z.tar.gz` to exist.
4. The target requires `vX.Y.Z` to be present on `origin` and fails with instructions to run `make release-push VERSION=x.y.z` when it is missing.
5. The target creates the GitHub release with generated notes, uploads the sdist as a release asset, and sets the latest flag from version ordering.

## Start the conda-forge handoff

1. Run `make release-conda-forge VERSION=x.y.z`.
2. The target requires an authenticated `gh` session and fails with login instructions when authentication is missing.
3. The target creates the issue in [conda-forge/firecrown-feedstock](https://github.com/conda-forge/firecrown-feedstock).
4. Review the PR created by the bot.
5. In `recipe/meta.yaml`, confirm that the `version`, release-asset `source.url`, and `sha256` match the new sdist.
6. Update any dependency versions required for the release.
7. Add the comment `@conda-forge-admin, please rerender` to the PR.
8. Wait for GitHub Actions to finish, approve the PR, and merge it.

The same conda-forge handoff applies to feature-line releases and maintenance releases.
