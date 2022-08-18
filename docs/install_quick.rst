=======================
Installation Quickstart
=======================

There are several ways to obtain Firecrown for your own use.
The method you should use to obtain Firecrown depends on how you want to use it.

Currently, the installation instructions for users installing on Mac machines with M1 (also called Apple Silicon) chips are different from other installation instructions.
If you are installing on such a machine please use the :doc:`Apple M1 installation instructions<apple_m1_instructions>`.

* *Developer use*: If you want to modify existing Firecrown code, or if you may produce new code and may produce a `pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_ to submit your code to Firecrown, use the :doc:`developer installation <developer_installation>`.

* *Non-developer usage*: If you want to write your own likelihood script or even create subclasses of classes already in Firecrown, but do not intend to submit code back to Firecrown, you can use the :doc:`non-development installation <non_developer_installation>`. If you choose this installation and decide later that you do want to submit your code back to Firecrown you will need to copy the new code files you write into a developer-usage environment at a later date.

.. toctree::
   :maxdepth: 1
   :name: installations

   Developer installation<developer_installation.rst>
   Non-development installation<non_developer_installation.rst>
   Apple M1 installation instructions<apple_m1_instructions.rst>

