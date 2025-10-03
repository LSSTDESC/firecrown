# Making a new release

This is the procedure we use for create a new release of Firecrown.

1. Create a local branch in your clone of the repo to be used for the PR that will be started later. Name it for the new release, e.g.:  v1.10
2. Make sure you have updated the `environment.yml` file to release, or force, any versioning required.
3. Update your local conda environment using `conda env update --file environment.yml --prune`
4. Make sure any new directories are being processed by black, flake8, mypy, and pylint.
5. Make sure all tests pass, all code checking is passed, all examples are working.
   Run these commands to verify:

   ```bash
   bash pre-commit-check
   ```

6. Make sure the documentation builds:

   ```bash
   make -C docs clean
   quarto render tutorial --output-dir=../docs/_static
   make -C docs html
   ```

7. Make sure that any new features introduced are described, and any features removed from the code have been removed, in the tutorial.
8. Update the version in `firecrown/version.py` (this is the ONLY file you need to edit manually).
9. Update the tutorial version metadata by running:

    ```bash
    python tutorial/update_quarto_version.py
    ```

10. Re-run the version tests to verify everything is correct:

    ```bash
    pytest -v tests/test_version.py
    ```

    This should pass. Note: The version in `docs/conf.py`, `tests/test_version.py`, and `tutorial/introduction_to_firecrown.qmd` are now automatically derived from `firecrown/version.py`, so you don't need to update them manually.

11. Commit and push all the changes to the new branch for the PR.
    This can be done either through the web interface, or through the `gh` command line utility.
12. Create a PR for the generation of the new release.
    The CI system will automatically run on the PR.
13. If the CI system complains, fix things and iterate as needed.
14. When the CI system passes, merge the PR.
15. Use the GitHub web interface to tag the commit.
    Allow the automated system to generate the release notes.
16. Create the new `conda` release.
    You do this in your local clone of the conda-forge feedstock repository (<https://github.com/conda-forge/firecrown-feedstock>).

    1. `git pull` to make sure you have the latest version of everything.
    2. Edit `recipe/meta.yaml` to update the version, and to reset the build number to 0.
       Make sure any version pinning in `meta.yaml` (which will be used by the conda solver) is consistent with Firecrown's `environment.yml`.
       Add any new dependencies, if appropriate.
       Download the new `.tar.gz` file from the Firecrown repo, find the `sha256sum` of the file.
       Update that in the `meta.yaml` file.
    3. Commit and push changes.
    4. If you make an error, make sure you update the build number.

17. Immediately change the version in `firecrown/version.py` to the next development version, then run `python tutorial/update_quarto_version.py` to update the tutorial.
    If the release just made is x.y.z, the new development version should be x.y.(z+1)a0 (where 'a0' indicates alpha/development status).
    For example: if you just released 1.8.3, change to 1.8.4a0.
