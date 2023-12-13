import pathlib

import pytest
from flymyai.multipart import BinaryField, SimpleField


@pytest.fixture
def binary_field_path():
    return "/home/oleg/PycharmProjects/vli_client_python/640.jpeg"


@pytest.fixture
def binary_field_desc(binary_field_path):
    image_path = pathlib.Path("/home/oleg/PycharmProjects/vli_client_python/640.jpeg")
    return open(image_path, "rb")


@pytest.fixture
def binary_field_bytes(binary_field_desc):
    image_bytes = binary_field_desc.read()
    return image_bytes


def test_binary_field(binary_field_path, binary_field_desc, binary_field_bytes):
    bin_f = BinaryField(binary_field_path)
    assert isinstance(bin_f.serialize(), tuple)
    bin_f = BinaryField(binary_field_desc)
    assert isinstance(bin_f.serialize(), tuple)
    bin_f = BinaryField(binary_field_bytes)
    assert isinstance(bin_f.serialize(), tuple)


@pytest.fixture
def simple_field_inputs():
    return [4, "string", 4.0, [4.0, 5.0], 100, {"v": "1"}]


def test_simple_field(simple_field_inputs):
    for inp in simple_field_inputs:
        field = SimpleField(inp)
        field.validate()
        assert field.serialize()
