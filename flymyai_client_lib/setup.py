import pathlib

from setuptools import setup

_BASE_PATH = pathlib.Path(__file__).parent

setup(
    name="flymyai",
    version="0.1.0",
    packages=[
        "cli",
        "conf",
        "core",
        "utils",
        "generators",
        "model_maker",
        "schema_obtainers",
    ],
    package_dir={"": str((_BASE_PATH / "flymyai").absolute())},
    license="",
    author="oleg",
    author_email="lyerhd@gmail.com",
    description="",
)
