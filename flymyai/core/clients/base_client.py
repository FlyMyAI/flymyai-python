import os
from typing import Generic, Optional, overload, AsyncIterator, Iterator, Callable
from typing import (
    TypeVar,
    Union,
)

import httpx
import pydantic

from flymyai.core.response_factory.async_task_result_factory import (
    AsyncTaskResultFactory,
)
from flymyai.core.response_factory.plain_inference_response_factory import (
    SSEInferenceResponseFactory,
)
from flymyai.core.authorizations import APIKeyClientInfo
from flymyai.core.exceptions import (
    ImproperlyConfiguredClientException,
    BaseFlyMyAIException,
    FlyMyAIAsyncTaskException,
)
from flymyai.core.models.successful_responses import (
    PredictionResponse,
    OpenAPISchemaResponse,
    PredictionPartial,
    AsyncPredictionTask,
    AsyncPredictionResponseList,
)
from flymyai.multipart import MultipartPayload

DEFAULT_RETRY_COUNT = os.getenv("FLYMYAI_MAX_RETRIES", 2)

_PossibleClients = TypeVar(
    "_PossibleClients", bound=Union[httpx.Client, httpx.AsyncClient]
)


_predict_timeout = httpx.Timeout(
    connect=int(os.getenv("FMA_CONNECT_TIMEOUT", 999999)),
    read=int(os.getenv("FMA_READ_TIMEOUT", 999999)),
    write=int(os.getenv("FMA_WRITE_TIMEOUT", 999999)),
    pool=int(os.getenv("FMA_POOL_TIMEOUT", 999999)),
)


class BaseClient(Generic[_PossibleClients]):
    """
    Base class for FlyMyAI clients
    """

    _client: _PossibleClients
    max_retries: int
    client_info: APIKeyClientInfo

    def __init__(
        self, apikey: str, model: Optional[str] = None, max_retries=DEFAULT_RETRY_COUNT
    ):
        self.client_info = APIKeyClientInfo(apikey)
        if model:
            self.client_info = self.client_info.copy_for_model(model)
        self._client = self._construct_client()
        self.max_retries = max_retries

    def amend_client_info(self, model: Optional[str] = None):
        if model:
            client_info = self.client_info.copy_for_model(model)
        else:
            client_info = self.client_info
        if not client_info.project_name or not client_info.username:
            raise ImproperlyConfiguredClientException(
                "model should be provided as <owner username>/<model>"
            )
        return client_info

    @overload
    async def predict(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> PredictionResponse: ...

    @overload
    def predict(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> PredictionResponse: ...

    def predict(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> PredictionResponse: ...

    @overload
    async def predict_async_task(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> AsyncPredictionTask: ...

    @overload
    def predict_async_task(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> AsyncPredictionTask: ...

    def predict_async_task(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> AsyncPredictionTask: ...

    @classmethod
    def _construct_task_result(cls, response):
        try:
            validated = AsyncTaskResultFactory(httpx_response=response).construct()
        except BaseFlyMyAIException as e:
            raise FlyMyAIAsyncTaskException.from_base_exception(e)
        try:
            return AsyncPredictionResponseList.from_response(validated, status=200)
        except pydantic.ValidationError as e:
            raise AsyncPredictionResponseList.convert_error(e) from e

    def _async_prediction_task_construct(
        self, response: httpx.Response, client_info: APIKeyClientInfo
    ) -> AsyncPredictionTask:
        async_prediction_task = AsyncPredictionTask[self.__class__].model_validate_json(
            json_data=response.content
        )
        async_prediction_task.client_info = client_info
        async_prediction_task.set_client(self)
        return async_prediction_task

    @overload
    async def prediction_task_result(
        self, prediction_task: AsyncPredictionTask, timeout: Optional[float] = None
    ): ...

    @overload
    def prediction_task_result(
        self, prediction_task: AsyncPredictionTask, timeout: Optional[float] = None
    ): ...

    def prediction_task_result(
        self, prediction_task: AsyncPredictionTask, timeout: Optional[float] = None
    ): ...

    @overload
    async def openapi_schema(
        self, model: Optional[str] = None, max_retries=None
    ) -> OpenAPISchemaResponse: ...

    @overload
    def openapi_schema(
        self, model: Optional[str] = None, max_retries=None
    ) -> OpenAPISchemaResponse: ...

    def openapi_schema(
        self, model: Optional[str] = None, max_retries=None
    ) -> OpenAPISchemaResponse: ...

    @overload
    async def stream(
        self,
        payload: dict,
        model: Optional[str] = None,
    ) -> AsyncIterator[PredictionPartial]: ...

    @overload
    def stream(
        self,
        payload: dict,
        model: Optional[str] = None,
    ) -> Iterator[PredictionPartial]: ...

    def stream(
        self,
        payload: dict,
        model: Optional[str] = None,
    ): ...

    def _stream_iterator(
        self, client_info, payload: MultipartPayload, is_long_stream: bool
    ) -> Union[Iterator[httpx.Response], AsyncIterator[httpx.Response]]:
        return self._client.stream(
            method="post",
            url=(
                client_info.prediction_path
                if not is_long_stream
                else client_info.prediction_stream_path
            ),
            **payload.serialize(),
            timeout=_predict_timeout,
            headers=client_info.authorization_headers,
            follow_redirects=True,
        )

    @staticmethod
    def _wrap_request(request_callback: Callable):
        response = request_callback()
        return SSEInferenceResponseFactory(httpx_response=response).construct()

    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        """
        Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        # If an error is thrown while constructing a client, self._client
        # may not be present
        if hasattr(self, "_client"):
            self._client.close()

    def _construct_client(self):
        raise NotImplemented

    @overload
    async def cancel_prediction(
        self,
        prediction_id: str,
        model: Optional[str] = None,
        client_info: APIKeyClientInfo = None,
    ): ...

    @overload
    def cancel_prediction(
        self,
        prediction_id: str,
        model: Optional[str] = None,
        client_info: APIKeyClientInfo = None,
    ): ...

    def cancel_prediction(
        self,
        prediction_id: str,
        model: Optional[str] = None,
        client_info: APIKeyClientInfo = None,
    ): ...
