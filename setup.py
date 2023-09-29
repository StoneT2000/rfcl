import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="rfcl",
    version="0.0.1",
    author="Anonymous",
    description="Reverse Forward Curriculum Learning",
    license="MIT",
    keywords=["reinforcement-learning", "machine-learning", "ai"],
    url="http://packages.python.org/rfcl",
    packages=["rfcl"],
    long_description=read("README.md"),
)
