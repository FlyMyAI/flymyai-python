import dataclasses
import json
from typing import Union

import httpx

from flymyai.core.models.base import ResponseLike


@dataclasses.dataclass
class Base4xxResponse(ResponseLike):
    """
    Base class for all 4xx
    """

    status_code: int
    url: httpx.URL
    content: bytes

    requires_retry: bool = False

    @classmethod
    def from_response(cls, response: Union[httpx.Response, ResponseLike]):
        return cls(response.status_code, response.url, response.content)


@dataclasses.dataclass
class FlyMyAI400Response(Base4xxResponse):
    """
    400 response
    requires_retry = False means we do not resend this request
    """

    requires_retry: bool = False

    def to_msg(self):
        return f"""
            Bad request happened: {self.content.decode()}
        """


@dataclasses.dataclass
class FlyMyAI401Response(Base4xxResponse):
    """
    401 response
    requires_retry = False means we do not resend this request
    """

    requires_retry: bool = False

    def to_msg(self):
        return f"""
            Authentication error: verify your credentials!
        """


@dataclasses.dataclass
class FlyMyAI421Response(Base4xxResponse):
    requires_retry = False

    def to_msg(self):
        jsoned = json.loads(self.content)
        msg = super().to_msg()
        if detail := jsoned.get("detail"):
            msg += f"\nDetail: {detail}"
        return msg


@dataclasses.dataclass
class FlyMyAI422Response(Base4xxResponse):
    """
    422 response
    requires_retry = False means we do not resend this request
    """

    requires_retry: bool = False

    def to_msg(self):
        jsoned = json.loads(self.content)
        msg = super().to_msg()
        if detail := jsoned.get("detail"):
            msg += f"Details: {detail}"
        return msg


@dataclasses.dataclass
class FlyMyAI425Response(Base4xxResponse):
    requires_retry: bool = True
