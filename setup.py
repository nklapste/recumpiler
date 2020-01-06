#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""setup.py for recumpiler"""

import codecs
import re
import sys
import os

from setuptools import setup, find_packages
from setuptools.command.test import test
from setuptools.command.install import install as _install


def find_version(*file_paths):
    with codecs.open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), *file_paths), "r"
    ) as fp:
        version_file = fp.read()
    m = re.search(r"^__version__ = \((\d+), ?(\d+), ?(\d+)\)", version_file, re.M)
    if m:
        return "{}.{}.{}".format(*m.groups())
    raise RuntimeError("Unable to find a valid version")


VERSION = find_version("recumpiler", "__init__.py")


class Pylint(test):
    def run_tests(self):
        from pylint.lint import Run

        Run(
            [
                "recumpiler",
                "--persistent",
                "y",
                "--rcfile",
                ".pylintrc",
                "--output-format",
                "colorized",
            ]
        )


class PyTest(test):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = "-v --cov={}".format("recumpiler")

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk

        nltk.download("wordnet")
        nltk.download("cmudict")
        nltk.download("punkt")


setup(
    name="recumpiler",
    version=VERSION,
    ## TODO: populate
    # description="",
    # long_description=open("README.rst").read(),
    # long_description_content_type="text/x-rst",
    keywords="joke funny vulgar stupid text",
    author="Nathan Klapstein",
    author_email="nklapste@ualberta.ca",
    url="https://github.com/nklapste/recumpiler",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "textblob>=0.15.3,<1.0.0",
        "word2number>=1.1,<2.0",
        "nltk>=3.4.5,<4.0.0",
        "lorem>=0.1.1,<1.0.0",
        "pronouncing>=0.2.0,<1.0.0",
        "pandas>=0.25.3,<1.0.0",
        "numpy>=1.18.0,<2.0.0",
        "better-profanity>=0.5.0,<1.0.0",
        "homoglyphs>=1.3.5,<2.0.0",
        "inflect>=4.0.0,<5.0.0",
        "compound-word-splitter>=0.4,<1.0",
    ],
    setup_requires=["nltk>=3.4.5,<4.0.0",],
    tests_require=[
        "pytest>=5.3.2,<6.0.0",
        "pytest-cov>=2.8.1,<3.0.0",
        "pylint>=2.4.4,<3.0.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=2.1.2,<3.0.0",
            "sphinx_rtd_theme>=0.4.3,<1.0.0",
            "sphinx-autodoc-typehints>=1.6.0,<2.0.0",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    cmdclass={"test": PyTest, "lint": Pylint},
    entry_points={"console_scripts": ["recumpiler = recumpiler.__main__:main"]},
)
