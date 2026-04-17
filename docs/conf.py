# Configuration file for the Sphinx documentation builder.
import importlib.metadata

project = "bagpuss"
author = "Institute for Gravitational Research, University of Glasgow"
release = importlib.metadata.version("bagpuss")
version = ".".join(release.split(".")[:2])

extensions = [
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# sphinx-autoapi: generate API docs from source without importing the package
autoapi_dirs = ["../bagpuss"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# Napoleon — support NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

html_theme = "kentigern"
html_title = f"bagpuss {version}"
