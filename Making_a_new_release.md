# Making a new release

This is the procedure we use for create a new release of Firecrown.

1. Create a local branch in your clone of the repo to be used for the PR that will be started later. Name it for the new release, e.g.:  v1.10
2. Update your local conda environment using `mamba update --all` and `pip install -U --no-deps autoclasstoc cobaya pygobject-stubs`
3. Make sure you have updated the `environment.yml` file to release, or force, any versioning required.
4. Update your local conda environment to match the possibly-modified `environment.yml`.
5. Make sure any new directories are being processed by black, flake8, mypy, and pylint.
6. Make sure all tests pass, all code checking is passed, all examples are working.
6a. Make sure the documentation builds:
     make -C clean
	 quarto render tutorial --output-dir=../docs/_static
	 make -C docs html
7. Make sure that any new features introduced are described, and any features removed from the code have been removed, in the tutorial.
8. Update the version in `firecrown/version.py`
9. Update the test `tests/test_version.py`
10. Re-run the version test:

    
    pytest -v tests/test_version.py

11. Update the tutorial files with the new release number.
    1. _quarto.yml
    2. introduction_to_firecrown.qmd
    3. `docs/conf.py`
    4. You can find all the files to change with: `rg -g '!*.ipynb' -g '!*.ini' -g '!*.yaml' -g '!*.yml' -l -F x.y` where "x.y" should be replaced with the old release number.
12. Commit and push all the changes to the new branch for the PR.
    1. can be done either through the web interface, or through the `gh` command line utility.
13. Create a PR for the generation of the new release.
   The CI system will automatically run on the PR.
14. If the CI system complains, fix things and iterate as needed.
15. When the CI system passes, merge the PR.
16. Use the GitHub web interface to tag the commit.
   Allow the automated system to genreate the release notes.
17. Create the new `conda` release.
   You do this in your local clone of the feedstock repository.

    1. `git pull` to make sure you have the latest version of everything.
    2. Edit `recipe/meta.yaml` to update the version, and to reset the build number to 0.
      Make sure any version pinning in `meta.yml` (which will be used by the conda solver)  is consistent with Firecrown's `environment.yml`.
      Add any new dependencies, if appropriate.
      Download the new `.tar.gz` file from the Firecrown repo, find the `sha256sum` of the file.
      Update that in the `meta.yml` file.
    3. Commit and push changes.
    4. If you make an error, make sure you update the build number.

18. Immediately change the version (in all the places listed above) the next number. If the release just made is x.y.z, the new one should be x.y.(z+1)a0

