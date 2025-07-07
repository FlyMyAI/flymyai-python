from dataclasses import dataclass
from typing import List, Dict

from flymyai.core.types.m1 import M1Record


@dataclass
class M1History:
    _records: List[M1Record]

    def __init__(self):
        self._records = []

    def add(self, M1Record):
        self._records.append(M1Record)

    def serialize(self) -> List[Dict]:
        return [
            {
                "role": record.role.value,
                "content": record.content,
            }
            for record in self._records
        ]

    def pop(self) -> M1Record:
        return self._records.pop()
