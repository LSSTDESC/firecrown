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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'firecrown'
copyright = '2022, LSST DESC Firecrown Contributors'
author = 'LSST DESC Firecrown Contributors'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
#templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['diagrams']


# -- Extension configuration -------------------------------------------------

# Some style options
highlight_language = 'python3'
pygments_style = 'sphinx'
todo_include_todos = True
add_function_parentheses = True
add_module_names = True

set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
autodoc_mock_imports = ["ccl", "pyccl", "numpy.typing._ufunc", "pandas._typing", "pandas"]

# Napoleon compiles the docstrings into .rst

# If True, include class __init__ docstrings separately from class
napoleon_include_init_with_doc = False
# If True, include docstrings of private functions
napoleon_include_private_with_doc = False
# Detail for converting docstrings to rst
napoleon_use_ivar = True


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
