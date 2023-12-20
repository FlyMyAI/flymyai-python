import dataclasses
import json
from typing import TypeVar, Union, Any

import httpx
import pydantic

from flymyai.core._streaming import ServerSentEvent


@dataclasses.dataclass
class Base4xxResponse:
    status_code: int
    url: httpx.URL
    content: bytes

    requires_retry: bool = False

    def to_msg(self):
        return f"""
            BAD REQUEST DETECTED ({self.status_code}):
            REQUEST URL: {self.url};
        """

    @classmethod
    def from_response(cls, response: httpx.Response):
        return cls(response.status_code, response.url, response.content)


@dataclasses.dataclass
class FlyMyAI400Response(Base4xxResponse):
    requires_retry: bool = False

    def to_msg(self):
        return f"""
            Bad request happened: {self.content.decode()}
        """


@dataclasses.dataclass
class FlyMyAI401Response(Base4xxResponse):
    requires_retry: bool = False

    def to_msg(self):
        return f"""
            Authentication error: verify your credentials!
        """


@dataclasses.dataclass
class FlyMyAI422Response(Base4xxResponse):
    requires_retry: bool = False

    def to_msg(self):
        jsoned = json.loads(self.content)
        msg = super().to_msg()
        if detail := jsoned.get("detail"):
            msg += f"Details: {detail}"
        return msg


class UnTypedResponse(pydantic.BaseModel):
    exc_history: list | None
    data: Any
    status_code: int


class PredictionResponse(pydantic.BaseModel):
    exc_history: list | None
    output_data: dict

    @classmethod
    def from_untyped(cls, untyped_response: UnTypedResponse):
        return cls(
            exc_history=untyped_response.exc_history,
            output_data=untyped_response.output_data,
        )


class OpenAPISchemaResponse(pydantic.BaseModel):
    exc_history: list | None
    schema: dict

    @classmethod
    def from_untyped(cls, untyped_response: UnTypedResponse):
        return cls(
            exc_history=untyped_response.exc_history, schema=untyped_response.data
        )
