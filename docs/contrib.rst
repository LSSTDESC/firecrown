Contributing
============

.. role:: bash(code)
   :language: bash

Contributions to Firecrown are welcome.

For any contribution, we suggest you start by `opening a discussion <https://github.com/LSSTDESC/firecrown/discussions>`_.
We are intending to use GitHub discussions to come to a consensus on the ideas for new additions.
Once a consensus is reached, we will convert the discussion into a GitHub issue that can be used to track the progress of the work.

New development for issues should be done on a branch.
To create a branch you will need write access; if you don't have write access, please send a request to the @LSSTDESC/firecrown-devs team.
You can also fork the repository and send a pull request from your fork.

You should be using the various code quality tools that are being used by the continuous integration (CI) system.
These tools are described below.
Starting this early in the development process is recommended.

You should regularly make sure your branch is up-to-date with the master branch.
This will help minimize the number of conflicts that can arise when merging.
It is OK for a PR in draft status to not be up-to-date with master, but any PRs submitted for review *must* be up-to-date.
Draft PRs can be used for final discussion of the design.

To prepare for converting a draft PR to one ready for review, please make sure that all the code quality tools are passing.
All the necessary tools are included in the `firecrown_developer` conda environment.
They will be run by the CI system on every pull request, and a review for merging will not begin until they pass.


As a shortcut, the script `pre-commit-check` can be run from the command line to check all the code quality tools are passing.
The tools used in this script are described below.

Code formatting
---------------

We are using the command-line tool :bash:`black` to auto-format Python code in Firecrown.
Please make sure to run black on your code before creating any commits.

.. code:: bash

    black firecrown/ examples/ tests/

We are also using :bash:`flake8` to check for coding style issues.

.. code:: bash

    flake8 firecrown/ examples/ tests/

Linting
-------

We are using :bash:`pylint` to check for a variety of possible problems.
`pylint` is run in the CI system with the following flags:

.. code:: bash

    pylint firecrown
    pylint --rcfile firecrown/models/pylintrc firecrown/models
    pylint --rcfile tests/pylintrc  tests

Type checking
-------------

We are using type-hinting in (most of) the code, to help ensure correct use of the framework.
We are using :bash:`mypy` to verify the code is conforming to these type hints.
Please run:

.. code:: bash

    mypy -p firecrown -p examples -p tests

and fix any errors reported before pushing commits to the GitHub repository.

Testing
-------

We are using :bash:`pytest` to run tests on Firecrown.
To run the same set of tests that will be run by the CI system, use:

.. code:: bash

    python -m pytest --runslow -vv --integration tests

Please note that when the CI system runs the tests, it will also ensure that all modified or newly-added code is actually tested.
For a PR to be reviewed, it must pass this requirement.

