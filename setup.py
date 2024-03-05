#!/usr/bin/env python

from os import path, walk

import sys
from setuptools import setup, find_packages

NAME = "Orange3-Explain"

VERSION = "0.6.9"

AUTHOR = "Bioinformatics Laboratory, FRI UL"
AUTHOR_EMAIL = "contact@orange.biolab.si"

URL = "https://orangedatamining.com/download"
DESCRIPTION = "Orange3 add-on for explanatory AI"
LONG_DESCRIPTION = open(
    path.join(path.dirname(__file__), "README.pypi"), "r", encoding="utf-8"
).read()

LICENSE = "GPL3+"

KEYWORDS = (
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    "orange3 add-on",
    "orange3-explain",
)

PACKAGES = find_packages()

PACKAGE_DATA = {
    "orangecontrib.explain.widgets": ["icons/*"],
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

INSTALL_REQUIRES = [
    "AnyQt",
    # shap's requirement, force users for numba to get updated because compatibility
    # issues with numpy - completely remove this pin after october 2024
    "numba >=0.58",
    "numpy",
    "Orange3 >=3.34.0",
    "orange-canvas-core >=0.1.28",
    "orange-widget-base >=4.19.0",
    "pyqtgraph",
    "scipy",
    "shap >=0.42.1",
    "scikit-learn>=1.0.1",
]

EXTRAS_REQUIRE = {
    'test': ['pytest', 'coverage'],
    'doc': ['sphinx', 'recommonmark', 'sphinx_rtd_theme'],
}

ENTRY_POINTS = {
    "orange3.addon": ("Orange3-Explain = orangecontrib.explain",),
    # Entry point used to specify packages containing widgets.
    "orange.widgets": ("Explain = orangecontrib.explain.widgets",),
    # Register widget help
    "orange.canvas.help": (
        "html-index = orangecontrib.explain.widgets:WIDGET_HELP_PATH",
    ),
}

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3 :: Only",
]


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    # if "bdist_wheel" in sys.argv and not path.exists(local_dir):
    #     print(
    #         "Directory '{}' does not exist. "
    #         "Please build documentation before running bdist_wheel.".format(
    #             path.abspath(local_dir)
    #         )
    #     )
    #     sys.exit(0)

    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append(
            (
                dirpath.replace(local_dir, install_dir),
                [path.join(dirpath, f) for f in files],
            )
        )
    DATA_FILES.extend(doc_files)


if __name__ == "__main__":
    include_documentation("doc/_build/html", "help/orange3-explain")
    setup(
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        classifiers=CLASSIFIERS,
        data_files=DATA_FILES,
        description=DESCRIPTION,
        entry_points=ENTRY_POINTS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        keywords=KEYWORDS,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license=LICENSE,
        name=NAME,
        namespace_packages=["orangecontrib"],
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        url=URL,
        version=VERSION,
        zip_safe=False,
    )
