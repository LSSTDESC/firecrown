# Making a new release

Firecrown derives release versions from git tags through `setuptools-scm`.
This runbook covers feature-line and maintenance releases, publication on
GitHub, and the conda-forge handoff.

## Naming conventions

- `x.y.0` is a feature-line release.
- `x.y.z`, where `z > 0`, is a maintenance release.
- `vx.y.z` is the corresponding git tag and GitHub release tag.
- `firecrown-x.y.z.tar.gz` is the release source distribution (sdist).
- `vx_y_support` is the support branch for release line `x.y`.

For example, version `1.16.0` uses tag `v1.16.0`, support branch
`v1_16_support`, and sdist `firecrown-1.16.0.tar.gz`.

Run commands from the Firecrown repository unless a step explicitly says to
run them from the feedstock checkout.

## Prerequisites

Activate the required developer environment:

```sh
conda activate firecrown_developer
```

Confirm that the Python build frontend and GitHub CLI authentication are
available:

```sh
python -m build --version
gh auth status --hostname github.com
```

If GitHub CLI is not authenticated, log in and check again:

```sh
gh auth login --hostname github.com --web
gh auth status --hostname github.com
```

Use the `firecrown_developer` environment throughout the release. The
validation, build, and publication targets enforce this environment and
require a clean tracked working tree. Before running one of those targets,
run:

```sh
git status --short
```

This command must produce no output.

## 1. Prepare and merge the release

### Feature-line release (`x.y.0`)

Synchronize `master`, then create the preparation branch:

```sh
git fetch origin
git switch master
git merge --ff-only origin/master
git switch -c prep-vx.y.0
```

Make the code, documentation, dependency, and tutorial changes for the
release. If `dependencies.yaml` changed, regenerate the dependent files in
this order:

```sh
make deps-sync
make conda-lock
```

Review and commit all regenerated files. Commit and push the complete release
preparation, then open a pull request targeting `master`. For example:

```sh
git push -u origin prep-vx.y.0
gh pr create --base master --title "Prep vx.y.0"
```

Wait for CI to pass, review the results, and merge the pull request.

After the merge, synchronize local `master` again:

```sh
git fetch origin
git switch master
git merge --ff-only origin/master
git status --short
```

Confirm that `HEAD` is the merged release commit and that
`git status --short` produces no output. Feature-line release validation
requires the current branch to be `master`. It does not independently confirm
that `master` matches `origin/master`.

### Maintenance release (`x.y.z`, where `z > 0`)

Synchronize the support branch, then create the preparation branch:

```sh
git fetch origin
git switch vx_y_support
git merge --ff-only origin/vx_y_support
git switch -c prep-vx.y.z
```

Make the code, documentation, and dependency changes for the maintenance
release. If `dependencies.yaml` changed, regenerate the dependent files in
this order:

```sh
make deps-sync
make conda-lock
```

Review and commit all regenerated files. Commit and push the complete release
preparation, then open a pull request targeting `vx_y_support`. For example:

```sh
git push -u origin prep-vx.y.z
gh pr create --base vx_y_support --title "Prep vx.y.z"
```

Wait for CI to pass, review the results, and merge the pull request.

After the merge, synchronize the local support branch again:

```sh
git fetch origin
git switch vx_y_support
git merge --ff-only origin/vx_y_support
git status --short
```

Confirm that `HEAD` is the merged release commit and that
`git status --short` produces no output. Maintenance release validation
requires the current branch to be `vx_y_support`, requires that branch to
exist on `origin`, and requires `HEAD` to match `origin/vx_y_support`. A
preparation branch or detached `HEAD` is not sufficient.

## 2. Tag and build the release

From the synchronized release branch and merged release commit, create the
local release refs:

```sh
make release-tag VERSION=x.y.z
```

For a feature-line release, use `VERSION=x.y.0`. The target creates the
annotated tag `vx.y.0` and local support branch `vx_y_support`. For a
maintenance release, it creates only the annotated tag `vx.y.z`.

`release-tag` performs release-specific validation and runs the full
`make pre-commit` suite unless a successful check is already cached for the
same `HEAD` and `VERSION`. Before creating refs, it reruns the fast validation,
including environment, clean-tree, version, branch, remote-tag, and support-
branch checks.

Refresh the installed Firecrown metadata, build the sdist, and verify both
reported package versions:

```sh
make install
make release-sdist VERSION=x.y.z
make release-verify-sdist VERSION=x.y.z
```

Use `VERSION=x.y.0` throughout for a feature-line release.
`release-sdist` requires the local release tag to exist and `HEAD` to match the
tagged commit. `release-verify-sdist` installs the sdist into a temporary
directory and verifies both `firecrown.__version__` and
`importlib.metadata.version("firecrown")` against the requested version.

If this release changes `setuptools-scm`, package metadata, the build backend
or configuration, or release artifact generation, also run:

```sh
make release-verify-archive VERSION=x.y.z
```

This diagnostic confirms that a GitHub auto-generated archive still cannot
provide the release version and therefore remains unsuitable for conda-forge.
It is not required for releases without versioning or build-behavior changes.

## 3. Publish the release

Push the verified release refs:

```sh
make release-push VERSION=x.y.z
```

This target rebuilds and verifies the sdist before pushing the tag to
`origin`. For a feature-line release, it also pushes `vx_y_support`.

Publish the GitHub release:

```sh
make release-github VERSION=x.y.z
```

This target rebuilds and verifies `dist/firecrown-x.y.z.tar.gz`, requires the
tag to be present on `origin`, creates the GitHub release with generated notes,
uploads the sdist as a release asset, and sets the latest-release flag from
version ordering.

## 4. Hand off to conda-forge

The recipe `source.url` must use the sdist uploaded to the GitHub release:

```text
https://github.com/LSSTDESC/firecrown/releases/download/vx.y.z/firecrown-x.y.z.tar.gz
```

Never use the GitHub auto-generated archive at `/archive/vx.y.z.tar.gz`. It
contains neither `PKG-INFO` nor `.git`, so `setuptools-scm` cannot determine
the release version.

### Create the handoff issue

From the Firecrown repository at the release commit, run:

```sh
make release-conda-forge VERSION=x.y.z
```

The target rebuilds and verifies the sdist, requires the GitHub release to
exist, computes the local sdist's SHA256, and creates an issue in
[`conda-forge/firecrown-feedstock`](https://github.com/conda-forge/firecrown-feedstock)
containing the exact source URL and checksum.

Run this target once per release and retain the issue URL printed by `gh`.
Every invocation creates a new issue. If an invocation is interrupted or its
result is uncertain, search the feedstock issues for the version before
running it again.

### Prepare the feedstock branch

Contribute through a fork of `conda-forge/firecrown-feedstock`; never push an
update branch directly to the conda-forge repository.

The first time you use a local feedstock clone, add the upstream remote. Run
this command from the feedstock checkout:

```sh
git remote add upstream https://github.com/conda-forge/firecrown-feedstock.git
```

Before every handoff, synchronize the fork and create an update branch. Run
these commands from the feedstock checkout:

```sh
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch -c update-firecrown-x.y.z
```

### Update the recipe

In the feedstock's `recipe/meta.yaml`:

- Set the recipe version to `x.y.z`.
- Set `source.url` and `source.sha256` to the values from the handoff issue.
- Ensure `setuptools-scm` is listed under `requirements.host`.
- Update any required non-generated dependencies or recipe fields.
- Set `build.number` to `0` for a new version. Increment it when correcting and
  republishing the same version.
- Ensure `test.commands` asserts both package version values:

```yaml
test:
  commands:
    - >-
      python -c "import firecrown, importlib.metadata as md;
      assert firecrown.__version__ == '{{ version }}';
      assert md.version('firecrown') == '{{ version }}'"
```

For every release, return to the Firecrown repository while it remains at the
release commit and regenerate the feedstock's dependency blocks:

```sh
make feedstock-sync FEEDSTOCK=<path-to-firecrown-feedstock>
```

Do not set `ALLOW_VERSION_MISMATCH=1` for a routine release handoff. The target
checks that the installed Firecrown version matches the version now specified
by the recipe. Review and commit all recipe changes on the feedstock update
branch.

Push the feedstock update branch to your fork and open a pull request targeting
`conda-forge/firecrown-feedstock`. Add this comment to request rerendering:

```text
@conda-forge-admin, please rerender
```

Wait for GitHub Actions to pass for all variants. Approve and merge the pull
request after CI succeeds.

## 5. Confirm completion

Before considering the release complete, confirm that:

- The intended commit has tag `vx.y.z`.
- The tag is present on `origin`.
- The GitHub release exists and contains `firecrown-x.y.z.tar.gz`.
- Sdist verification reported the expected value from both
  `firecrown.__version__` and `importlib.metadata.version("firecrown")`.
- The conda-forge pull request passed all variants and was merged.

## Recovery and local cleanup

For routine local cleanup, remove generated artifacts and cached release
checks without deleting refs:

```sh
make release-clean
```

To discard a failed release attempt that has not been published remotely, run:

```sh
make release-clean VERSION=x.y.z
```

The versioned form also deletes the local release tag. For a feature-line
release, it warns before force-deleting the local support branch and refuses
to delete that branch while it is checked out.

Neither command changes remote refs. The versioned command does not roll back
a published release and should not be used as routine post-publication
cleanup.
