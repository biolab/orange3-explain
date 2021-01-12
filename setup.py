#!/usr/bin/env python

from os import path, walk

import sys
from setuptools import setup, find_packages

NAME = "Orange3-Explain"

VERSION = "0.2.0"

AUTHOR = "Bioinformatics Laboratory, FRI UL"
AUTHOR_EMAIL = "contact@orange.biolab.si"

URL = "http://orange.biolab.si/download"
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
    "orangecontrib.example.widgets": ["icons/*"],
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

INSTALL_REQUIRES = [
    "Orange3 >= 3.27.1",
    "AnyQt",
    "numpy",
    "pyqtgraph",
    "scipy",
    "shap ==0.37.*",  # shap makes significant changes between versions
]

EXTRAS_REQUIRE = {
    'test': ['pytest', 'coverage']
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
    include_documentation("doc/_build/htmlhelp", "help/orange3-example")
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
