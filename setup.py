import pathlib
from setuptools import setup, find_packages

_BASE_PATH = pathlib.Path(__file__).parent

setup(
    name="flymyai",
    version="0.1.0",
    packages=find_packages(where=str(_BASE_PATH)),
    package_dir={"": str(_BASE_PATH)},
    include_package_data=True,
    license="",
    author="oleg",
    author_email="lyerhd@gmail.com",
    description="",
)
