# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "firecrown"
copyright = "2022--2024, LSST DESC Firecrown Contributors"
author = "LSST DESC Firecrown Contributors"

# The full version, including alpha/beta/rc tags
release = "1.8.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoclasstoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# To see how to further customize the theme, see:
# https://sphinx-book-theme.readthedocs.io/en/stable/index.html

html_theme = "sphinx_book_theme"
html_theme_options = {
    "collapse_navigation": False,
    "repository_url": "https://github.com/LSSTDESC/firecrown",
    "use_repository_button": True,
}
html_title = "firecrown"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

# mathjax
mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

# autosummary
autosummary_generate = True

# Some style options
highlight_language = "python3"
pygments_style = "sphinx"
todo_include_todos = True
add_function_parentheses = True
add_module_names = True

set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
autodoc_mock_imports = [
    "ccl",
    "pyccl",
    "numpy.typing._ufunc",
    "pandas._typing",
    "pandas",
    "numpy._typing._ufunc",
]

# Napoleon compiles the docstrings into .rst

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

sphinx_apidoc_options = [
    "members",
    "show-inheritance",
    "private-members",
    "special-members",
]
os.environ["SPHINX_APIDOC_OPTIONS"] = ",".join(sphinx_apidoc_options)

autoclasstoc_sections = [
    "public-methods",
    "private-methods",
]

# Copied from github.com/sanderslab/magellanmapper:
# automate building API .rst files, necessary for ReadTheDocs, as inspired by:
# https://github.com/readthedocs/readthedocs.org/issues/1139#issuecomment-398083449


def run_apidoc(_):
    ignore_paths = []

    argv = [
        "--separate",
        "-f",
        "-M",
        "-e",
        "-E",
        "-T",
        "-d",
        "1",
        "-o",
        "_api",
        "../firecrown",
    ] + ignore_paths

    try:
        # Sphinx >= 1.7
        from sphinx.ext import apidoc

        apidoc.main(argv)
    except ImportError:
        # Sphinx  < 1.7
        from sphinx import apidoc

        argv.insert(0, apidoc.__file__)
        apidoc.main(argv)
