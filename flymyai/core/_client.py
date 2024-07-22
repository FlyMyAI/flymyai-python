import os
from typing import (
    Callable,
    Awaitable,
    Generic,
    TypeVar,
    Union,
    overload,
    Iterator,
    AsyncContextManager,
    AsyncIterator,
    Optional,
)

import httpx

from flymyai.core._response import FlyMyAIResponse
from flymyai.core._response_factory import ResponseFactory
from flymyai.core._streaming import SSEDecoder
from flymyai.core.authorizations import APIKeyClientInfo
from flymyai.core.exceptions import (
    FlyMyAIPredictException,
    FlyMyAIExceptionGroup,
    BaseFlyMyAIException,
    FlyMyAIOpenAPIException,
    ImproperlyConfiguredClientException,
)
from flymyai.core.models import (
    PredictionResponse,
    OpenAPISchemaResponse,
    PredictionPartial,
    StreamDetails,
)
from flymyai.multipart.payload import MultipartPayload
from flymyai.utils.utils import retryable_callback, aretryable_callback

DEFAULT_RETRY_COUNT = os.getenv("FLYMYAI_MAX_RETRIES", 2)

_PossibleClients = TypeVar(
    "_PossibleClients", bound=Union[httpx.Client, httpx.AsyncClient]
)


_predict_timeout = httpx.Timeout(None, connect=10)


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
    ) -> PredictionResponse:
        ...

    @overload
    def predict(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> PredictionResponse:
        ...

    def predict(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> PredictionResponse:
        ...

    @overload
    async def openapi_schema(
        self, model: Optional[str] = None, max_retries=None
    ) -> OpenAPISchemaResponse:
        ...

    @overload
    def openapi_schema(
        self, model: Optional[str] = None, max_retries=None
    ) -> OpenAPISchemaResponse:
        ...

    def openapi_schema(
        self, model: Optional[str] = None, max_retries=None
    ) -> OpenAPISchemaResponse:
        ...

    @overload
    async def stream(
        self,
        payload: dict,
        model: Optional[str] = None,
    ) -> AsyncIterator[PredictionPartial]:
        ...

    @overload
    def stream(
        self,
        payload: dict,
        model: Optional[str] = None,
    ) -> Iterator[PredictionPartial]:
        ...

    def stream(
        self,
        payload: dict,
        model: Optional[str] = None,
    ):
        ...

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
        return ResponseFactory(httpx_response=response).construct()

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


class PredictionStream:
    stream_details: StreamDetails

    def __init__(self, response_iterator: Iterator):
        self.response_iterator = response_iterator

    def __iter__(self):
        return self

    def __next__(self):
        response_end = None
        try:
            next_resp: FlyMyAIResponse = self.response_iterator.__next__()
            response_end = next_resp
            return PredictionPartial.from_response(response_end)
        except BaseFlyMyAIException as e:
            response_end = e.response
            raise e
        finally:
            if not response_end:
                raise StopIteration()
            stream_details_marshalled = response_end.json().get("stream_details")
            if stream_details_marshalled:
                self.stream_details = StreamDetails.model_validate(
                    stream_details_marshalled
                )


class BaseSyncClient(BaseClient[httpx.Client]):
    def _construct_client(self):
        return httpx.Client(
            http2=True,
            headers=self.client_info.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://api.flymy.ai/"),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()

    @classmethod
    def _sse_instant(cls, stream_iter_func: Callable[[], Iterator[httpx.Response]]):
        """
        Fetch sse response on prediction
        :param stream_iter_func: context manager with underlying stream
        :return: FlyMyAIResponse
        """
        with stream_iter_func() as stream:
            stream: httpx.Response
            response = ResponseFactory(
                sse=next(SSEDecoder().iter(stream.iter_lines())),
                httpx_request=stream.request,
                httpx_response=stream,
            ).construct()
            return response

    def _predict(self, payload: MultipartPayload, client_info: APIKeyClientInfo):
        """
        Wrap predict method in sse
        """

        try:
            return self._sse_instant(
                lambda: self._stream_iterator(client_info, payload, False)
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIPredictException.from_response(e.response)

    def predict(self, payload: dict, model: Optional[str] = None, max_retries=None):
        """
        Wrap predict method in sse.
        Retries until max_retries or self.max_retries is reached
        :param model: flymyai/bert | None, If none - get self.client_info.<username/project_name>
        :param payload: anything for model
        :param max_retries: retries
        :return: PredictionResponse(exc_history, output_data, response):
                exc_history - list of exception history during prediction
                output_data - dict with prediction output
        """

        payload = MultipartPayload(payload)
        history, response = retryable_callback(
            lambda: self._predict(payload, self.amend_client_info(model)),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse.from_response(response, exc_history=history)

    def _stream(self, client_info: APIKeyClientInfo, payload: dict):
        payload = MultipartPayload(payload)
        response_iterator = self._stream_iterator(
            client_info, payload, is_long_stream=True
        )
        decoder = SSEDecoder()
        with response_iterator as sse_stream:
            for sse_partial in decoder.iter(sse_stream.iter_lines()):
                try:
                    response = ResponseFactory(
                        sse=sse_partial,
                        httpx_request=sse_stream.request,
                        httpx_response=sse_stream,
                    ).construct()
                except BaseFlyMyAIException as e:
                    raise FlyMyAIPredictException.from_response(e.response)
                yield response

    def stream(self, payload: dict, model: Optional[str] = None):
        stream_iter = self._stream(self.amend_client_info(model), payload)
        stream_wrapper = PredictionStream(stream_iter)
        return stream_wrapper

    def _openapi_schema(self, client_info: APIKeyClientInfo):
        """
        OpenAPI request for the current project, wrapped in executor-method (using HTTP/1)
        :return:
        """
        try:
            return self._wrap_request(
                lambda: self._client.get(
                    client_info.openapi_schema_path,
                    headers=client_info.authorization_headers,
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIOpenAPIException.from_response(e.response)

    def openapi_schema(self, model: Optional[str] = None, max_retries=None):
        """
        :param model: flymyai/bert
        :param max_retries: retries before give up
        :return:
        :return: OpenAPISchemaResponse(exc_history, openapi_schema, response):
                exc_history - dict with exceptions;
                openapi_schema - dict with openapi;
        """
        history, response = retryable_callback(
            lambda: self._openapi_schema(client_info=self.amend_client_info(model)),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse.from_response(
            exc_history=history, openapi_schema=response.json(), response=response
        )

    @classmethod
    def run_predict(cls, apikey: str, model: str, payload: dict):
        """
        :param apikey: fly-...
        :param model:  flymyai/bert
        :param payload: jsonable / multipart/form-data available data
        :return: PredictionResponse(exc_history, output_data, response):
                exc_history - list of exception history during prediction;
                output_data - dict with prediction output;
        """
        with cls(apikey, model) as client:
            return client.predict(payload)


class AsyncPredictionStream:
    stream_details: StreamDetails

    def __init__(self, response_iterator: AsyncIterator):
        self.response_iterator = response_iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        response_end = None
        try:
            next_resp: FlyMyAIResponse = await self.response_iterator.__anext__()
            response_end = next_resp
            return PredictionPartial.from_response(response_end)
        except BaseFlyMyAIException as e:
            response_end = e.response
            raise e
        finally:
            if not response_end:
                raise StopAsyncIteration()
            stream_details_marshalled = response_end.json().get("stream_details")
            if stream_details_marshalled:
                self.stream_details = StreamDetails.model_validate(
                    stream_details_marshalled
                )


class BaseAsyncClient(BaseClient[httpx.AsyncClient]):
    def _construct_client(self):
        return httpx.AsyncClient(
            http2=True,
            headers=self.client_info.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://api.flymy.ai/"),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_client"):
            await self._client.aclose()

    async def openapi_schema(self, model: Optional[str] = None, max_retries=None):
        """
        :param max_retries: retries before giving up
        :return:
        :return: OpenAPISchemaResponse(exc_history, openapi_schema, response):
                exc_history - dict with exceptions;
                openapi_schema - dict with openapi;
        """
        history, response = await aretryable_callback(
            lambda: self._openapi_schema(),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse.from_response(
            exc_history=history, openapi_schema=response.json(), response=response
        )

    def _openapi_schema(self, client_info: APIKeyClientInfo):
        """
        OpenAPI request for the current project, wrapped in executor-method (using HTTP/1)
        :return:
        """
        try:
            return self._wrap_request(
                lambda: self._client.get(
                    client_info.openapi_schema_path,
                    headers=client_info.authorization_headers,
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIOpenAPIException.from_response(e.response)

    @classmethod
    async def _sse_instant(
        cls, async_response_stream: Callable[[], AsyncContextManager[httpx.Response]]
    ):
        """
        A non-blocking approach to fetch a response stream
        :param async_response_stream: context manager with underlying stream
        :return: FlyMyAIResponse
        """
        async with async_response_stream() as stream:
            sse = await SSEDecoder().aiter(stream.aiter_lines()).__anext__()
            response = ResponseFactory(
                sse=sse, httpx_request=stream.request, httpx_response=stream
            ).construct()
            return response

    def _predict(self, client_info, payload: MultipartPayload):
        """
        Executes request and waits for sse data
        :param payload: model input data
        :return: FlyMyAIResponse or raise an exception
        """
        try:
            return self._sse_instant(
                lambda: self._client.stream(
                    method="post",
                    url=client_info.prediction_path,
                    timeout=_predict_timeout,
                    **payload.serialize(),
                    headers=client_info.authorization_headers,
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIPredictException.from_response(e.response)

    async def predict(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ) -> PredictionResponse:
        """
        Wrap predict method in sse.
        Retries until max_retries or self.max_retries is reached
        :param model: flymyai/bert
        :param payload: anything for model
        :param max_retries: retries
        :return: PredictionResponse(exc_history, output_data, response):
                exc_history - list of exception history during prediction
                output_data - dict with prediction output
        """
        payload = MultipartPayload(input_data=payload)
        history, response = await aretryable_callback(
            lambda: self._predict(self.amend_client_info(model), payload),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse.from_response(response, exc_history=history)

    async def _stream(self, client_info: APIKeyClientInfo, payload: dict):
        payload = MultipartPayload(payload)
        stream_iterator = self._stream_iterator(
            client_info, payload, is_long_stream=True
        )
        decoder = SSEDecoder()
        async with stream_iterator as sse_stream:
            async for sse_partial in decoder.aiter(sse_stream.aiter_lines()):
                try:
                    response = ResponseFactory(
                        sse=sse_partial,
                        httpx_request=sse_stream.request,
                        httpx_response=sse_stream,
                    ).construct()
                except BaseFlyMyAIException as e:
                    raise FlyMyAIPredictException.from_response(e.response)
                yield response

    def stream(self, payload: dict, model: Optional[str] = None, max_retries=None):
        stream_iter = self._stream(self.amend_client_info(model), payload)
        stream_wrapper = AsyncPredictionStream(stream_iter)
        return stream_wrapper

    @staticmethod
    async def _wrap_request(request_callback: Callable[..., Awaitable[httpx.Response]]):
        """
        Execute a request callback and return the response
        """
        response = await request_callback()
        return ResponseFactory(httpx_response=response).construct()

    async def close(self):
        """
        Close the client
        """
        await self._client.aclose()

    @classmethod
    async def arun_predict(cls, apikey: str, model: str, payload: dict):
        """
        Execute simple prediction out of a box
        :param model: flymyai/bert
        :param apikey: fly-...
        :param payload: {dict with prediction input}
        :return: PredictionResponse(exc_history, output_data, response)
                exc_history - list of exception history during prediction
                output_data - dict with prediction output
        """
        async with cls(apikey, model) as client:
            return await client.predict(payload)
