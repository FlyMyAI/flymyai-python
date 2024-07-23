import dataclasses
import json
from typing import Optional

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


class BaseFromServer(pydantic.BaseModel):
    _response: FlyMyAIResponse = PrivateAttr()

    @property
    def response(self):
        return self._response

    @classmethod
    def from_response(cls, response: FlyMyAIResponse, **kwargs):
        status_code = kwargs.pop("status", response.status_code)
        response_json = response.json()
        response_json["status"] = response_json.get("status", status_code)
        self = cls(**response_json, **kwargs)
        self._response = response
        return self


class PredictionResponse(BaseFromServer):
    """
    Prediction response from FlyMyAI
    """

    exc_history: Optional[list]
    output_data: dict
    status: int

    inference_time: Optional[float] = None

    @property
    def response(self):
        return self._response


class OpenAPISchemaResponse(BaseFromServer):
    """
    OpenAPI schema for the current project. Use it to construct your own schema
    """

    exc_history: Optional[list]
    openapi_schema: dict
    status: int


class PredictionPartial(BaseFromServer):
    status: int
    output_data: Optional[dict] = None

    _response: FlyMyAIResponse = PrivateAttr()


class StreamDetails(pydantic.BaseModel):
    input_tokens: int
    output_tokens: int
    size_in_billions: float = pydantic.Field(alias="model_size_in_billions")
