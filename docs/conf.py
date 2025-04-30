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
import re
from docutils import nodes
from sphinxawesome_theme.postprocess import Icons

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, basedir)


html_permalinks_icon = Icons.permalinks_icon  # SVG as a string

# -- Project information -----------------------------------------------------

project = "franken"
copyright = "2024, franken team"
author = "franken team"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxawesome_theme",
    "myst_parser",
    "sphinxarg.ext",
    "nbsphinx",
]

myst_enable_extensions = ["amsmath", "dollarmath", "html_image"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "python": ("https://docs.python.org/3.9/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]

autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
# to handle functions as default input arguments
autodoc_preserve_defaults = True
# Warn about broken links
nitpicky = True
autodoc_inherit_docstrings = False
# autodoc_class_signature = "separated"
# autoclass_content = "class"
# autosummary_generate = False

# autodoc_member_order = "groupwise"
# napoleon_preprocess_types = True
# napoleon_use_rtype = False

# master_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["css/mystyle.css"]
templates_path = ["_templates"]
# Favicon configuration
# html_favicon = '_static/favicon.ico'
# Configure syntax highlighting for Awesome Sphinx Theme
pygments_style = "default"
pygments_style_dark = "material"
html_title = "franken"
# Additional theme configuration
html_theme_options = {
    "show_prev_next": True,
    "show_scrolltop": True,
    "main_nav_links": {
        "Docs": "index",
        "API Reference": "reference/index",
    },
    "extra_header_link_icons": {
        "GitHub": {
            "link": "https://github.com/CSML-IIT-UCL/mlpot_transfer",
            "icon": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" height="18" fill="currentColor"><path d="M12 0C5.373 0 0 5.373 0 12c0 5.302 3.438 9.8 8.205 11.387.6.111.82-.261.82-.577 0-.285-.01-1.04-.015-2.04-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.09-.745.083-.729.083-.729 1.205.084 1.84 1.238 1.84 1.238 1.07 1.835 2.807 1.305 3.492.998.108-.775.418-1.305.76-1.605-2.665-.305-5.466-1.332-5.466-5.93 0-1.31.467-2.38 1.235-3.22-.125-.303-.535-1.523.115-3.176 0 0 1.005-.322 3.3 1.23.955-.265 1.98-.398 3-.403 1.02.005 2.045.138 3 .403 2.28-1.552 3.285-1.23 3.285-1.23.655 1.653.245 2.873.12 3.176.77.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.62-5.475 5.92.43.37.81 1.1.81 2.22 0 1.605-.015 2.895-.015 3.285 0 .32.215.694.825.575C20.565 21.795 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg>""",
        },
    },
    # "logo_light": "_static/[logo_light].png",
    # "logo_dark": "_static/[logo_dark].png",
}

## Teletype role
tt_re = re.compile('^:tt:`(.*)`$')
def tt_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Can be used as :tt:`SOME_TEXT_HERE`,
    """
    result = []
    m = tt_re.search(rawtext)
    if m:
        arg = m.group(1)
        result = [nodes.literal('', arg)]
        result[0]['classes'].append('literal-no-code')
    return result,[]


def setup(app):
    app.add_role('tt', tt_role)
