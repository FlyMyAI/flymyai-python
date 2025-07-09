from dataclasses import dataclass
from enum import Enum
from typing import Optional


class M1Role(Enum):
    user = "user"
    assistant = "assistant"


@dataclass
class M1Record:
    role: M1Role
    content: str


@dataclass
class M1GenerationTask:
    request_id: str
