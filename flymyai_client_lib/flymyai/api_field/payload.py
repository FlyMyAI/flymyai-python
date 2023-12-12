import io
import pathlib

from api_field.binary_field import BinaryField
from api_field.simple_field import SimpleField


class MultipartPayload:
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
            if isinstance(value, str):
                if pathlib.Path(value).is_file():
                    files[key] = self._serialize_bin(value)
                else:
                    data[key] = self._serialize_simple(value)
            elif isinstance(value, (pathlib.Path, io.BufferedIOBase)):
                files[key] = self._serialize_simple(value)
            else:
                raise TypeError("Unsupported type for field {}".format(key))
        return {"files": files, "data": data}
