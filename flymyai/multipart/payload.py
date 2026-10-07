import io
import json
import pathlib

from .binary_field import BinaryField
from .simple_field import SimpleField

_BINARY_TYPES = (pathlib.Path, io.BufferedIOBase, bytes)


class MultipartPayload:
    """
    This class provides a way to create a multipart-prepared
    payload (multipart/form-data) from a python dict

    A form field carries text, so nested values are encoded on the way out:
    a dict (``response_schema``) is sent as its JSON text, a list (``images``)
    as one field per item - the form the gateway reads lists from - and a list
    of files as one file part per item. httpx sent a dict as its Python repr
    in a urlencoded body and refused it in a multipart one.
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

    @classmethod
    def _form_value(cls, value):
        """A JSON-able value as form text: containers as JSON, primitives as they are."""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return cls._serialize_simple(value)

    def serialize(self) -> dict:
        files = []
        data = {}

        for key, value in self.data.items():
            if isinstance(value, _BINARY_TYPES):
                files.append((key, self._serialize_bin(value)))
            elif isinstance(value, dict):
                data[key] = self._form_value(value)
            elif isinstance(value, (list, tuple)) and value:
                if all(isinstance(item, _BINARY_TYPES) for item in value):
                    files.extend((key, self._serialize_bin(item)) for item in value)
                elif any(isinstance(item, _BINARY_TYPES) for item in value):
                    raise TypeError(
                        f"Field {key!r} mixes files with other values;"
                        " send either files or JSON-able values"
                    )
                else:
                    data[key] = [self._form_value(item) for item in value]
            else:
                data[key] = self._serialize_simple(value)
        return {"files": files or {}, "data": data}
