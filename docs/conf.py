# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib.metadata
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sciSOM'
copyright = '2024, Luis A. Sanchez'
author = 'Luis A. Sanchez'
try:
    # Attempt to get the version from the installed package
    version = release = importlib.metadata.version("sciSOM")
except importlib.metadata.PackageNotFoundError:
    # Fallback to import the local sciSOM module directly
    try:
        import sciSOM
        version = release = getattr(sciSOM, '__version__', '0.0.0')
    except ImportError:
        # If that also fails, use a default version
        version = release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser',
              'sphinx.ext.autodoc', 
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx_autodoc_typehints',
              'sphinx_copybutton',]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
templates_path = ['_templates']
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

master_doc = 'index'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True
autosummary_generate = True
