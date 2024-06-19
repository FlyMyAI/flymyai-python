import pathlib

from setuptools import setup

_BASE_PATH = pathlib.Path(__file__).parent

setup(
    name="flymyai",
    version="0.1.10",
    packages=[
        "core",
        "utils",
        "multipart",
    ],
    package_dir={"": str((_BASE_PATH / "flymyai"))},
    license="",
    author="oleg",
    author_email="lyerhd@gmail.com",
    description="",
)
