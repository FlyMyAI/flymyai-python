import os
from typing import (
    Callable,
    Awaitable,
    Generic,
    TypeVar,
    Union,
    overload,
    Iterator,
    AsyncGenerator,
    AsyncIterator,
    AsyncContextManager,
)

import httpx

from flymyai.core._response_factory import ResponseFactory
from flymyai.core._streaming import SSEDecoder
from flymyai.multipart.payload import MultipartPayload
from flymyai.core.authorizations import APIKeyClientInfo, ClientInfoFactory
from flymyai.core.exceptions import (
    FlyMyAIPredictException,
    FlyMyAIExceptionGroup,
    BaseFlyMyAIException,
    FlyMyAIOpenAPIException,
)
from flymyai.core.models import PredictionResponse, OpenAPISchemaResponse
from flymyai.utils.utils import retryable_callback, aretryable_callback

DEFAULT_RETRY_COUNT = os.getenv("FLYMYAI_MAX_RETRIES", 2)

_PossibleClients = TypeVar(
    "_PossibleClients", bound=Union[httpx.Client, httpx.AsyncClient]
)


_predict_timeout = httpx.Timeout(None, connect=10)


class BaseClient(Generic[_PossibleClients]):
    _client: _PossibleClients
    max_retries: int
    auth: APIKeyClientInfo

    def __init__(self, auth: APIKeyClientInfo | dict, max_retries=DEFAULT_RETRY_COUNT):
        if isinstance(auth, dict):
            self.auth = ClientInfoFactory(auth).build_auth()
        elif isinstance(auth, APIKeyClientInfo):
            self.auth = auth
        else:
            raise TypeError("Invalid credentials. dict required!")
        self._client = self._construct_client()
        self.max_retries = max_retries

    @overload
    async def predict(self, input_data: dict, max_retries=None) -> PredictionResponse:
        ...

    @overload
    def predict(self, input_data: dict, max_retries=None) -> PredictionResponse:
        ...

    def predict(self, input_data: dict, max_retries=None) -> PredictionResponse:
        ...

    @overload
    async def openapi_schema(self, max_retries=None) -> OpenAPISchemaResponse:
        ...

    @overload
    def openapi_schema(self, max_retries=None) -> OpenAPISchemaResponse:
        ...

    def openapi_schema(self, max_retries=None) -> OpenAPISchemaResponse:
        ...

    @staticmethod
    def _wrap_request(request_callback: Callable):
        response = request_callback()
        return ResponseFactory(httpx_response=response).construct()

    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        # If an error is thrown while constructing a client, self._client
        # may not be present
        if hasattr(self, "_client"):
            self._client.close()

    def _construct_client(self):
        raise NotImplemented


class BaseSyncClient(BaseClient[httpx.Client]):
    def _construct_client(self):
        return httpx.Client(
            headers=self.auth.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://flymy.ai/"),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()

    @classmethod
    def _sse_instant(cls, stream_iter_func: Callable[[], Iterator[httpx.Response]]):
        with stream_iter_func() as stream:
            stream: httpx.Response
            response = ResponseFactory(
                sse=next(SSEDecoder().iter(stream.iter_lines())),
                httpx_request=stream.request,
                httpx_response=stream,
            ).construct()
            return response

    def _predict(self, payload: MultipartPayload):
        try:
            return self._sse_instant(
                lambda: self._client.stream(
                    method="post",
                    url=self.auth.prediction_path,
                    **payload.serialize(),
                    timeout=_predict_timeout,
                    headers=self.auth.authorization_headers,
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIPredictException.from_response(e.response)

    def predict(self, payload: dict, max_retries=None):
        payload = MultipartPayload(payload)
        history, response = retryable_callback(
            lambda: self._predict(payload),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse(
            exc_history=history, output_data=response.json(), response=response
        )

    def _openapi_schema(self):
        try:
            return self._wrap_request(
                lambda: self._client.get(
                    self.auth.openapi_schema_path,
                    headers=self.auth.authorization_headers,
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIOpenAPIException.from_response(e.response)

    def openapi_schema(self, max_retries=None):
        history, response = retryable_callback(
            lambda: self._openapi_schema(),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse(
            exc_history=history, schema=response.json(), response=response
        )

    @classmethod
    def run_predict(cls, auth: dict, payload: dict):
        auth = ClientInfoFactory(raw_auth=auth).build_auth()
        with cls(auth) as client:
            return client.predict(payload)


class BaseAsyncClient(BaseClient[httpx.AsyncClient]):
    def _construct_client(self):
        return httpx.AsyncClient(
            headers=self.auth.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://flymy.ai/"),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_client"):
            await self._client.aclose()

    async def openapi_schema(self, max_retries=None):
        history, response = await aretryable_callback(
            lambda: self._openapi_schema(),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse(
            exc_history=history, schema=response.json(), response=response
        )

    def _openapi_schema(self):
        try:
            return self._wrap_request(
                lambda: self._client.get(
                    self.auth.openapi_schema_path,
                    headers=self.auth.authorization_headers,
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIOpenAPIException.from_response(e.response)

    @classmethod
    async def _sse_instant(
        cls, async_response_stream: Callable[[], AsyncContextManager[httpx.Response]]
    ):
        async with async_response_stream() as stream:
            sse = await anext(SSEDecoder().aiter(stream.aiter_lines()))
            response = ResponseFactory(
                sse=sse, httpx_request=stream.request, httpx_response=stream
            ).construct()
            return response

    def _predict(self, payload: MultipartPayload):
        try:
            return self._sse_instant(
                lambda: self._client.stream(
                    method="post",
                    url=self.auth.prediction_path,
                    timeout=_predict_timeout,
                    **payload.serialize(),
                    headers=self.auth.authorization_headers,
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIPredictException.from_response(e.response)

    async def predict(self, payload: dict, max_retries=None):
        payload = MultipartPayload(input_data=payload)
        history, response = await aretryable_callback(
            lambda: self._predict(payload),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse(
            exc_history=history, output_data=response.json(), response=response
        )

    @staticmethod
    async def _wrap_request(request_callback: Callable[..., Awaitable[httpx.Response]]):
        response = await request_callback()
        return ResponseFactory(httpx_response=response).construct()

    async def close(self):
        await self._client.aclose()

    @classmethod
    async def arun_predict(cls, auth: dict, payload: dict):
        auth = ClientInfoFactory(raw_auth=auth).build_auth()
        async with cls(auth) as client:
            return await client.predict(payload)
