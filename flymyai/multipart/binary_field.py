import io
import mimetypes
import pathlib
import uuid
from io import BytesIO
from typing import Union, BinaryIO, Any

from .base_field import BaseField

_BinaryInput = Union[bytes, pathlib.Path, BinaryIO, str]
_IOOutput = Union[BinaryIO, io.BytesIO]
_FieldOutput = tuple[str, _IOOutput, str]  # filename, io[binary], mime


def is_binary_input(value: _BinaryInput) -> bool:
    if isinstance(value, (bytes, BinaryIO)):
        return True
    if isinstance(value, str):
        try:
            pathlib.Path(value).resolve()
            return True
        except FileNotFoundError:
            return False
    if isinstance(value, pathlib.Path):
        return True
    return False


class BinaryField(BaseField):

    """
    Primitive that handles a binary input
    """

    def __init__(self, value: _BinaryInput):
        super().__init__(value)

    def validate(self, value: _BinaryInput | None = None) -> None:
        value = value or self.value
        if not is_binary_input(value):
            raise TypeError()

    @staticmethod
    def to_io(value: _BinaryInput) -> _IOOutput:
        if isinstance(value, bytes):
            io_obj = io.BytesIO(value)
            io_obj.name = uuid.uuid4().hex
        elif isinstance(value, (pathlib.Path, str)):
            io_obj = open(value, "rb")
        elif isinstance(value, io.BufferedIOBase):
            io_obj = value
        else:
            raise TypeError(
                f"Required one of: bytes, str, pathlib.Path, got {type(value)}"
            )
        return io_obj

    def serialize(
        self, value=None
    ) -> tuple[BinaryIO | BytesIO, str | Any, tuple[str | None, str | None] | str]:
        value = value or self.value
        io_obj = self.to_io(value)
        filename = io_obj.name
        mime = mimetypes.guess_type(filename)[0] or "applications/octet-stream"
        io_obj.seek(0)
        return filename, io_obj, mime
