import os
import uuid
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
from flymyai.core._response import FlyMyAIResponse
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
    PredictionEvent,
    AsyncPredictionTask,
    AsyncPredictionResponseList,
)
from flymyai.core.types.event_types import EventType
from flymyai.multipart import MultipartPayload

# Attempts for a request whose status allows another one (flymyai.core._retry).
# An environment value is a string: read it as a number, or every retry loop that
# compares the count with it would never stop.
DEFAULT_RETRY_COUNT = int(os.getenv("FLYMYAI_MAX_RETRIES", 2))

_PossibleClients = TypeVar(
    "_PossibleClients", bound=Union[httpx.Client, httpx.AsyncClient]
)

# Transport errors before a response arrived: the request is sent again (after a
# reconnect and a backoff), as the official OpenAI and Anthropic SDKs do. A stream
# that has delivered an event is never sent again (FlyMyAIStreamInterrupted).
_CONNECT_RECONNECT_EXC = (
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.ReadError,
    httpx.ConnectError,
    httpx.NetworkError,
    httpx.TimeoutException,
)

# How many times to reconnect and retry on connection/stream errors (1 initial + this many retries)
_RECONNECT_RETRIES = int(os.getenv("FMA_RECONNECT_RETRIES", "3"))

# The header that makes a repeated async submit one job (the gateway's
# apps/inference/idempotency.py): one key per logical call, the same on every attempt.
IDEMPOTENCY_HEADER = "Idempotency-Key"


class StreamEndedBeforeFirstEvent(httpx.RemoteProtocolError):
    """An HTTP 200 stream that closed before it delivered any event: nothing reached
    the caller, so the request is sent again like any cut before a response."""


def _new_idempotency_key() -> str:
    return f"flymyai-python-{uuid.uuid4()}"


def _rewind_files(serialized: dict) -> dict:
    """The same multipart form for another attempt: file objects back at their start.

    A form serialized once keeps its file names, so a repeated async submit carries
    the same fingerprint as the first and the gateway replays the first job.
    """
    files = serialized.get("files") or []
    for item in files:
        try:
            _name, field = item
            io_obj = field[1]
            io_obj.seek(0)
        except (TypeError, ValueError, AttributeError, IndexError):
            continue
    return serialized


class _StreamState:
    """How far a stream got: events delivered, and the prediction id it announced."""

    def __init__(self):
        self.events = 0
        self.prediction_id: Optional[str] = None

    def saw(self, response) -> None:
        self.events += 1
        if self.prediction_id is not None or not getattr(response, "is_event", False):
            return
        try:
            event = PredictionEvent.from_response(response)
        except Exception:  # noqa: BLE001 - an event of another shape
            return
        if event.event_type == EventType.STREAM_ID:
            self.prediction_id = event.prediction_id


def _status_error(stream: httpx.Response, content: bytes) -> BaseFlyMyAIException:
    """The FlyMyAI error for a response whose HTTP status is an error, headers kept."""
    return BaseFlyMyAIException.from_response(
        FlyMyAIResponse(
            status_code=stream.status_code,
            content=content,
            request=stream.request,
            headers=stream.headers,
        )
    )


def _is_reconnectable_error(exc: BaseException) -> bool:
    if isinstance(exc, _CONNECT_RECONNECT_EXC):
        return True
    if isinstance(exc, RuntimeError) and "client has been closed" in str(exc).lower():
        return True
    if type(exc).__name__ in ("ClosedResourceError", "BrokenResourceError"):
        return True
    # HTTP/2 connection closed then reused (e.g. "ConnectionState.CLOSED", "SEND_SETTINGS", "StreamReset")
    msg = str(exc)
    if (
        "ConnectionState.CLOSED" in msg
        or "SEND_SETTINGS" in msg
        or "StreamReset" in msg
    ):
        return True
    return False


_predict_timeout = httpx.Timeout(
    connect=int(os.getenv("FMA_CONNECT_TIMEOUT", 999999)),
    read=int(os.getenv("FMA_READ_TIMEOUT", 999999)),
    write=int(os.getenv("FMA_WRITE_TIMEOUT", 999999)),
    pool=int(os.getenv("FMA_POOL_TIMEOUT", 999999)),
)

_http2 = os.getenv("FLYMYAI_HTTP2", "true").lower() in ("1", "true", "yes")
_limits = httpx.Limits(
    max_connections=int(os.getenv("FMA_MAX_CONNECTIONS", "100")),
    max_keepalive_connections=int(os.getenv("FMA_MAX_KEEPALIVE_CONNECTIONS", "50")),
    keepalive_expiry=float(os.getenv("FMA_KEEPALIVE_EXPIRY", "60")),
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
        self,
        payload: dict,
        model: Optional[str] = None,
        max_retries=None,
        idempotency_key: Optional[str] = None,
    ) -> AsyncPredictionTask: ...

    @overload
    def predict_async_task(
        self,
        payload: dict,
        model: Optional[str] = None,
        max_retries=None,
        idempotency_key: Optional[str] = None,
    ) -> AsyncPredictionTask: ...

    def predict_async_task(
        self,
        payload: dict,
        model: Optional[str] = None,
        max_retries=None,
        idempotency_key: Optional[str] = None,
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

    def _reconnect_client(self):
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass
        self._client = self._construct_client()

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
