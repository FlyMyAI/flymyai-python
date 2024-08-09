import io
import pathlib

from .binary_field import BinaryField
from .simple_field import SimpleField


class MultipartPayload:
    """
    This class provides a way to create a multipart-prepared
    payload (multipart/form-data) from a python dict
    """

    def __init__(self, input_data: dict):
        self.data = input_data

    @classmethod
    def _serialize_bin(cls, value):
        data = BinaryField(value)
        data.validate()
        return data.serialize()

    @classmethod
    def _serialize_simple(cls, value):
        data = SimpleField(value)
        data.validate()
        return data.serialize()

    def serialize(self) -> dict:
        files = {}
        data = {}

        for key, value in self.data.items():
            if isinstance(value, (pathlib.Path, io.BufferedIOBase, bytes)):
                files[key] = self._serialize_bin(value)
            else:
                data[key] = self._serialize_simple(value)
        return {"files": files, "data": data}
