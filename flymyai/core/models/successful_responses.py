from typing import Optional

import pydantic
from pydantic import PrivateAttr

from flymyai.core._response import FlyMyAIResponse
from flymyai.core.types.event_types import EventType


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


class PredictionEvent(BaseFromServer):
    status: int
    event_type: EventType

    prediction_id: Optional[str] = None  # EventType.STREAM_ID


class StreamDetails(pydantic.BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    size_in_billions: Optional[float] = pydantic.Field(
        default=None, alias="model_size_in_billions"
    )
