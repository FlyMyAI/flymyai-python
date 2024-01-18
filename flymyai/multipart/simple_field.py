import json
from typing import Any

from .base_field import BaseField


class SimpleField(BaseField):

    """
    Primitive field that stores a single jsonable-value
    """

    def __init__(self, value):
        super().__init__(value)

    def validate(self, value: Any = None):
        json.dumps(value or self.value)

    def serialize(self, value: Any = None):
        return value or self.value
