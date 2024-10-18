from typing import Optional, Generic, TypeVar, List, TypedDict, Union, Awaitable

import pydantic
from pydantic import PrivateAttr, model_validator, Field
from pydantic_core._pydantic_core import PydanticCustomError
from typing_extensions import Self

from flymyai.core._response import FlyMyAIResponse
from flymyai.core.authorizations import APIKeyClientInfo
from flymyai.core.exceptions import BaseFlyMyAIException, FlyMyAIExceptionGroup
from flymyai.core.models.base import ResponseLike
from flymyai.core.types.event_types import EventType

_ClientT = TypeVar("_ClientT", bound="BaseClient")


class BaseFromServer(pydantic.BaseModel):
    _response: Optional[FlyMyAIResponse] = PrivateAttr(default=None)

    @property
    def response(self):
        return self._response

    @classmethod
    def from_response(cls, response: FlyMyAIResponse, **kwargs):
        status_code = kwargs.pop("status", response.status_code)
        response_json = response.json()
        response_json["status"] = response_json.get("status", status_code)
        ctx = kwargs.pop("context", None)
        self = cls.model_validate(dict(**response_json, **kwargs), context=ctx)
        self._response = response
        return self


class BasePredictionResponse(BaseFromServer):
    exc_history: Optional[list] = Field(default_factory=list)
    output_data: dict


class PredictionResponse(BasePredictionResponse):
    """
    Prediction response from FlyMyAI
    """

    status: int
    inference_time: Optional[float] = None

    @property
    def response(self):
        return self._response


class AsyncPredictionResponse(BasePredictionResponse):
    infer_details: dict
    output_data: dict = Field(validation_alias="response")

    @property
    def status(self) -> int:
        return self.infer_details.get("status", 200)

    @model_validator(mode="after")
    def validate_inference_details(self, ctx):
        self._response = ctx.context.get("_response")
        if (status := self.infer_details.get("status", 200)) != 200:
            raise PydanticCustomError(
                "inference_response_error",
                "Inference details contains incorrect status: {failure_status}",
                dict(failure_status=status, instance=self),
            )
        return self


class AsyncPredictionResponseList(BaseFromServer):
    inference_responses: List[AsyncPredictionResponse]

    @classmethod
    def from_response(cls, response: FlyMyAIResponse, **kwargs):
        result = super().from_response(
            response, context={"_response": response}, **kwargs
        )
        return result

    class _ErrorCTX(TypedDict):
        failure_status: int
        instance: Self

    @classmethod
    def convert_error(cls, e: pydantic.ValidationError) -> FlyMyAIExceptionGroup:
        errors = []
        error_data = e.errors(include_input=False)
        for error in error_data:
            err_t = error.get("type")
            ctx: Optional[cls._ErrorCTX] = error.get("ctx", {})
            if err_t == "inference_response_error":
                if not ctx:
                    raise KeyError("Inference response pydantic error should have ctx!")
                errors.append(
                    BaseFlyMyAIException.from_response(
                        ResponseLike(
                            status_code=ctx["failure_status"],
                            url=getattr(ctx["instance"], "_response").url,
                            content=ctx["instance"].model_dump_json().encode(),
                        )
                    )
                )
        if len(errors) != error_data:
            errors.append(e)
        exc_group = FlyMyAIExceptionGroup(errors)
        return exc_group


class AsyncPredictionTask(BaseFromServer, Generic[_ClientT]):
    _affiliated_client: Optional[_ClientT] = PrivateAttr(default=None)
    _client_info: APIKeyClientInfo = PrivateAttr(default=None)

    prediction_id: str

    def result(
        self, timeout=None
    ) -> Union[AsyncPredictionResponseList, Awaitable[AsyncPredictionResponseList]]:
        return self._affiliated_client.prediction_task_result(self, timeout=timeout)

    @property
    def client_info(self) -> APIKeyClientInfo:
        return self._client_info

    @client_info.setter
    def client_info(self, v: APIKeyClientInfo):
        self._client_info = v

    def set_client(self, client: _ClientT):
        self._affiliated_client = client


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
