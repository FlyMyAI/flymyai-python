from typing import Any


class BaseField(object):
    def __init__(self, value, name=None):
        self.name = name
        self.value = value

    def validate(self, value: Any = None):
        raise NotImplementedError

    def serialize(self, value: Any = None):
        self.validate(value)
        return value
