import json
from typing import Any

from .base_field import BaseField


class SimpleField(BaseField):
    def __init__(self, value):
        super().__init__(value)

    def validate(self, value: Any = None):
        json.dumps(value or self.value)

    def serialize(self, value: Any = None):
        return value or self.value
