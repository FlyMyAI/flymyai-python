import asyncio
import os
from typing import Optional, Callable, AsyncContextManager, Awaitable

import httpx

from flymyai.core.response_factory.plain_inference_response_factory import (
    SSEInferenceResponseFactory,
)
from flymyai.core._retry import (
    exception_retry_delay,
    exception_should_retry,
    retry_delay,
)
from flymyai.core._streaming import SSEDecoder
from flymyai.core.authorizations import APIKeyClientInfo
from flymyai.core.clients.base_client import (
    BaseClient,
    IDEMPOTENCY_HEADER,
    StreamEndedBeforeFirstEvent,
    _StreamState,
    _new_idempotency_key,
    _predict_timeout,
    _http2,
    _limits,
    _is_reconnectable_error,
    _rewind_files,
    _status_error,
    _RECONNECT_RETRIES,
)
from flymyai.core.exceptions import (
    BaseFlyMyAIException,
    FlyMyAIOpenAPIException,
    FlyMyAIPredictException,
    FlyMyAIExceptionGroup,
    FlyMyAIAsyncTaskException,
    FlyMyAIStreamError,
    FlyMyAIStreamInterrupted,
)
from flymyai.core.models.successful_responses import (
    OpenAPISchemaResponse,
    PredictionResponse,
    AsyncPredictionTask,
)
from flymyai.core.stream_iterators.AsyncPredictionStream import AsyncPredictionStream
from flymyai.multipart import MultipartPayload
from flymyai.utils.utils import aretryable_callback


class BaseAsyncClient(BaseClient[httpx.AsyncClient]):
    def _construct_client(self):
        return httpx.AsyncClient(
            http2=_http2,
            limits=_limits,
            headers=self.client_info.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://api.flymy.ai/"),
            timeout=_predict_timeout,
        )

    async def _reconnect_client(self):
        if hasattr(self, "_client"):
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = self._construct_client()

    async def _awith_reconnect(self, fn):
        """Send again after a transport error before any response: reconnect, back off."""
        last_exc = None
        for attempt in range(1 + _RECONNECT_RETRIES):
            try:
                return await fn()
            except BaseException as e:
                last_exc = e
                if not _is_reconnectable_error(e):
                    raise
                if attempt < _RECONNECT_RETRIES:
                    await asyncio.sleep(retry_delay(attempt))
                    await self._reconnect_client()
                    continue
                raise
        assert last_exc is not None
        raise last_exc

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
            lambda: self._openapi_schema(client_info=self.amend_client_info(model)),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse.from_response(
            exc_history=history, openapi_schema=response.json(), response=response
        )

    async def _openapi_schema(self, client_info: APIKeyClientInfo):
        """
        OpenAPI request for the current project, wrapped in executor-method (using HTTP/1)
        :return:
        """
        try:
            return await self._awith_reconnect(
                lambda: self._wrap_request(
                    lambda: self._client.get(
                        client_info.openapi_schema_path,
                        headers=client_info.authorization_headers,
                    )
                )
            )
        except BaseFlyMyAIException as e:
            raise FlyMyAIOpenAPIException.from_response(e.response)

    async def cancel_prediction(
        self,
        prediction_id: str,
        model: Optional[str] = None,
        client_info: APIKeyClientInfo = None,
    ):
        if client_info:
            full_client_info = client_info
        else:
            full_client_info = self.amend_client_info(model)
        response = await self._awith_reconnect(
            lambda: self._client.patch(
                url=full_client_info.prediction_cancel_path,
                json={"infer_id": prediction_id},
            )
        )
        return SSEInferenceResponseFactory(
            httpx_response=response, httpx_request=response.request
        ).construct()

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
            if stream.status_code >= 400:
                # An error before the answer (a proxy page, a rate limit): its status
                # and headers (x-should-retry, retry-after) decide another attempt.
                await stream.aread()
                raise FlyMyAIPredictException.from_base_exception(
                    _status_error(stream, stream.content)
                )
            try:
                sse = await SSEDecoder().aiter(stream.aiter_lines()).__anext__()
            except StopAsyncIteration:
                raise StreamEndedBeforeFirstEvent(
                    "The response ended before its first event"
                ) from None
            try:
                response = SSEInferenceResponseFactory(
                    sse=sse, httpx_request=stream.request, httpx_response=stream
                ).construct()
                return response
            except BaseFlyMyAIException as e:
                raise FlyMyAIPredictException.from_base_exception(e)

    async def _predict(self, client_info, payload: MultipartPayload):
        """
        Executes request and waits for sse data
        :param payload: model input data
        :return: FlyMyAIResponse or raise an exception
        """
        return await self._awith_reconnect(
            lambda: self._sse_instant(
                lambda: self._client.stream(
                    method="post",
                    url=client_info.prediction_path,
                    timeout=_predict_timeout,
                    **payload.serialize(),
                    headers=client_info.authorization_headers,
                )
            )
        )

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

    async def predict_async_task(
        self,
        payload: dict,
        model: Optional[str] = None,
        max_retries=None,
        idempotency_key: Optional[str] = None,
    ) -> AsyncPredictionTask:
        """
        Submit an async prediction. Every attempt carries one ``Idempotency-Key``
        (yours, or one made for this call), so a submit sent again after a lost answer
        replays the first job instead of starting a second paid one.
        """
        payload = MultipartPayload(input_data=payload)
        client_info = self.amend_client_info(model)
        # Serialized once: the same file names on every attempt keep the gateway's
        # fingerprint of the request, which the key is bound to.
        serialized = payload.serialize()
        headers = {IDEMPOTENCY_HEADER: idempotency_key or _new_idempotency_key()}

        async def submit():
            response = await self._awith_reconnect(
                lambda: self._client.post(
                    client_info.prediction_async_path,
                    **_rewind_files(serialized),
                    headers=headers,
                )
            )
            try:
                return SSEInferenceResponseFactory(response).construct()
            except BaseFlyMyAIException as e:
                raise FlyMyAIAsyncTaskException.from_base_exception(e)

        try:
            _, response = await aretryable_callback(
                submit,
                max_retries or self.max_retries,
                FlyMyAIAsyncTaskException,
                FlyMyAIExceptionGroup,
            )
            return self._async_prediction_task_construct(response, client_info)
        except FlyMyAIExceptionGroup as group:
            # The submit's own error, as before; the attempts stay in __cause__.
            last = group.errors[-1] if group.errors else None
            if isinstance(last, FlyMyAIAsyncTaskException):
                raise last from group
            raise
        except BaseFlyMyAIException as e:
            raise FlyMyAIAsyncTaskException.from_base_exception(e)

    async def prediction_task_result(
        self, prediction_task: AsyncPredictionTask, timeout: Optional[float] = None
    ):
        prediction_id = prediction_task.prediction_id

        async def get_res():
            data_resp = await self._awith_reconnect(
                lambda: self._client.get(
                    url=(
                        prediction_task.client_info or self.client_info
                    ).prediction_result_path,
                    params={"request_id": prediction_id},
                )
            )
            return self._construct_task_result(data_resp)

        _, res = await aretryable_callback(
            lambda: get_res(),
            None,
            FlyMyAIAsyncTaskException,
            FlyMyAIExceptionGroup,
            timeout,
            0.5,
        )
        return res

    async def _stream(self, client_info: APIKeyClientInfo, payload: dict):
        """
        The stream's responses. Sent again only before its first event (a transport
        error, or a status that allows it); after that, a broken connection raises
        FlyMyAIStreamInterrupted and an error event FlyMyAIStreamError: the stream is
        never started a second time behind the caller's back (that repeated the
        beginning of the answer and its cost).
        """
        payload = MultipartPayload(payload)
        state = _StreamState()
        transport_retries = 0
        status_attempts = 0
        max_attempts = int(self.max_retries or 1)
        while True:
            try:
                async with self._stream_iterator(
                    client_info, payload, is_long_stream=True
                ) as sse_stream:
                    if sse_stream.status_code >= 400:
                        await sse_stream.aread()
                        raise _status_error(sse_stream, sse_stream.content)
                    decoder = SSEDecoder()
                    async for sse_partial in decoder.aiter(sse_stream.aiter_lines()):
                        try:
                            response = SSEInferenceResponseFactory(
                                sse=sse_partial,
                                httpx_request=sse_stream.request,
                                httpx_response=sse_stream,
                            ).construct()
                        except BaseFlyMyAIException as e:
                            raise FlyMyAIStreamError.from_base_exception(
                                e, state.prediction_id, state.events
                            ) from e
                        state.saw(response)
                        yield response
                    if not state.events:
                        raise StreamEndedBeforeFirstEvent(
                            "The stream ended before its first event"
                        )
                return
            except FlyMyAIStreamError:
                raise
            except BaseFlyMyAIException as e:
                # An error status before the first event.
                status_attempts += 1
                if status_attempts < max_attempts and exception_should_retry(e):
                    await asyncio.sleep(exception_retry_delay(status_attempts - 1, e))
                    continue
                raise FlyMyAIPredictException.from_base_exception(e) from e
            except Exception as e:
                if not _is_reconnectable_error(e):
                    raise
                if state.events:
                    raise FlyMyAIStreamInterrupted(
                        f"The stream broke after {state.events} event(s): {e!r}."
                        " It is not sent again automatically.",
                        prediction_id=state.prediction_id,
                        events_received=state.events,
                    ) from e
                if transport_retries >= _RECONNECT_RETRIES:
                    # Nothing arrived on any attempt: the connection error itself.
                    raise
                await asyncio.sleep(retry_delay(transport_retries))
                transport_retries += 1
                await self._reconnect_client()

    def stream(self, payload: dict, model: Optional[str] = None, max_retries=None):
        full_client_info = self.amend_client_info(model)
        stream_iter = self._stream(full_client_info, payload)
        stream_wrapper = AsyncPredictionStream(stream_iter, self, full_client_info)
        return stream_wrapper

    @staticmethod
    async def _wrap_request(request_callback: Callable[..., Awaitable[httpx.Response]]):
        """
        Execute a request callback and return the response
        """
        response = await request_callback()
        return SSEInferenceResponseFactory(httpx_response=response).construct()

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
