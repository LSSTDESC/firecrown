
Contributing
============

.. role:: bash(code)
   :language: bash

Contributions to Firecrown are welcome.

For any contribution, please start by `opening an issue <https://github.com/LSSTDESC/firecrown/issues>`_,
and using the GitHub interface to create a branch for that issue.

To create a branch you will need write access; if you don't have write access, please send a request to the @LSSTDESC/firecrown-devs team.
You can also fork the repository and send a pull request from your fork.

When you have completed the task, push your commits to the branch you created for the issue and create a pull request.

We are using several tools to help keep the code tidy and correct; these are described below.

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

    We are working on improving the coverage of testing for Firecrown; it is currently very inadequate.
    As the coverage improves, we will provide instructions for writing tests for new code.

We are using :bash:`pytest` to run tests on Firecrown.
Before running tests, one must first build the code.
In addition, the environment variable :bash:`PYTHONPATH` must be correctly set to run the tests.
Please see the instructions, above, for this setup.

Use of :bash:`pylint`
---------------------

We are using :bash:`pylint` to check for a variety of possible problems.
Firecrown is not currently "clean" of all :bash:`pylint` issues, so we are not yet using :bash:`pylint` in the CI testing.

We are actively working on getting a "clean" report from :bash:`pylint`.
When this is achieved, we plan to activate :bash:`pylint` in the CI checking.
This will require that all new code pass :bash:`pylint`'s checks.

Code formatting
---------------

We are using the command-line tool :bash:`black` to auto-format Python code in Firecrown.
Please make sure to run black on your code before creating any commits.
