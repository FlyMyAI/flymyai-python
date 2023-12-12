import json
from typing import Any

from flymyai.api_field.base_field import BaseField


class SimpleField(BaseField):
    def __init__(self, value):
        super().__init__(value)

    def validate(self, value: Any = None):
        pass

    def serialize(self, value: Any = None):
        return json.dumps(value or self.value)
