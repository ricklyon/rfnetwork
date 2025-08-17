import os
import sys
import rfnetwork

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rfnetwork'
copyright = '2025, ricklyon'
author = 'ricklyon'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery"
]

templates_path = ['_templates']

exclude_patterns = [
    "**/build",
    ".venv",
    "debug",
]

html_show_sourcelink = False

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'filename_pattern': '/',
    'ignore_pattern': r'interactive_.*\.py',
    'example_extensions': {'.py'},
    'download_all_examples': False,
    'line_numbers': True
}

# Make autosummary generate stub pages
autosummary_generate = True

autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": False,
}

autosummary_context = {
    # Methods that should be skipped when generating the docs
    "skipmethods": ["__init__"],
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
