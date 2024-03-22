import dataclasses
import json

import httpx
import pydantic
from pydantic import PrivateAttr

from flymyai.core._response import FlyMyAIResponse


@dataclasses.dataclass
class Base4xxResponse:
    """
    Base class for all 4xx
    """

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


class PredictionResponse(pydantic.BaseModel):
    """
    Prediction response from FlyMyAI
    """

    exc_history: list | None
    output_data: dict
    _response: FlyMyAIResponse = PrivateAttr()

    inference_time: float | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._response = data.get("response")

    @property
    def response(self):
        return self._response


class OpenAPISchemaResponse(pydantic.BaseModel):
    """
    OpenAPI schema for current project. Use it to construct your own schema
    """

    exc_history: list | None
    openapi_schema: dict
    _response: FlyMyAIResponse = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._response = data.get("response")

    @property
    def response(self):
        return self._response
