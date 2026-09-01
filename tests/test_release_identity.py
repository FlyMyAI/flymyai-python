"""Release-candidate identity stays consistent across Python build metadata."""

import re
from pathlib import Path

import tomli

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.2.0rc1"


def test_python_sdk_build_metadata_uses_one_unreleased_rc_identity():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomli.load(handle)
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    setup_version = re.search(r'version="([^"]+)"', setup_source)

    assert pyproject["tool"]["poetry"]["version"] == EXPECTED_VERSION
    assert setup_version is not None
    assert setup_version.group(1) == EXPECTED_VERSION
