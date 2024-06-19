import pathlib

import pytest

from flymyai.multipart import BinaryField, SimpleField
from .FixtureFactory import FixtureFactory

factory = FixtureFactory(__file__)


@pytest.fixture
def binary_field_path() -> pathlib.Path:
    return pathlib.Path(factory("binary_field_path"))


@pytest.fixture
def binary_field_desc(binary_field_path):
    return open(binary_field_path, "rb")


@pytest.fixture
def binary_field_bytes(binary_field_desc):
    image_bytes = binary_field_desc.read()
    return image_bytes


@pytest.fixture
def simple_field_inputs():
    return factory("simple_field_inputs")


def test_binary_field(binary_field_path, binary_field_desc, binary_field_bytes):
    bin_f = BinaryField(binary_field_path)
    assert isinstance(bin_f.serialize(), tuple)
    bin_f = BinaryField(binary_field_desc)
    assert isinstance(bin_f.serialize(), tuple)
    bin_f = BinaryField(binary_field_bytes)
    assert isinstance(bin_f.serialize(), tuple)


def test_simple_field(simple_field_inputs):
    for inp in simple_field_inputs:
        field = SimpleField(inp)
        field.validate()
        assert field.serialize()
