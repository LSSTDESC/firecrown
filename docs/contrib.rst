Contributing
============

.. role:: bash(code)
   :language: bash

Contributions to Firecrown are welcome.

For any contribution, please start by `opening a discussion <https://github.com/LSSTDESC/firecrown/discussions>`_.
We are intending to use GitHub discussions to come to a consensus on the ideas for new additions.
Once a consensus is reached, we will convert the discussion into a GitHub issue that can be used to track the progress of the work.

New development for issues should be done on a branch.
To create a branch you will need write access; if you don't have write access, please send a request to the @LSSTDESC/firecrown-devs team.
You can also fork the repository and send a pull request from your fork.

When you have completed the task, push your commits to the branch you created for the issue and create a pull request.

We are using several tools to help keep the code tidy and correct; these are described below.
These tools are all applied by the continuous integration (CI) system that is run on every pull request.

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

.. warning::

    We are working on improving the coverage of testing for Firecrown; it is currently inadequate.
    As the coverage improves, we will provide instructions for writing tests for new code.

We are using :bash:`pytest` to run tests on Firecrown.

Use of :bash:`pylint`
---------------------

We are using :bash:`pylint` to check for a variety of possible problems.
`pylint` is run in the CI system with the following flags:

.. code:: bash

    pylint firecrown
    pylint --rcfile tests/pylintrc  tests
    pylint --rcfile firecrown/models/pylintrc firecrown/models

Code formatting
---------------

We are using the command-line tool :bash:`black` to auto-format Python code in Firecrown.
Please make sure to run black on your code before creating any commits.

.. code:: bash

    black firecrown examples tests
