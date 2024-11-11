import os
from typing import Callable, Iterator, Optional

import httpx

from flymyai.core._streaming import SSEDecoder
from flymyai.core.authorizations import APIKeyClientInfo
from flymyai.core.clients.base_client import BaseClient, _predict_timeout
from flymyai.core.exceptions import (
    BaseFlyMyAIException,
    FlyMyAIOpenAPIException,
    FlyMyAIPredictException,
    FlyMyAIExceptionGroup,
    FlyMyAIAsyncTaskException,
)
from flymyai.core.models.successful_responses import (
    PredictionResponse,
    OpenAPISchemaResponse,
    AsyncPredictionTask,
)
from flymyai.core.response_factory.plain_inference_response_factory import (
    SSEInferenceResponseFactory,
)
from flymyai.core.stream_iterators.PredictionStream import PredictionStream
from flymyai.multipart import MultipartPayload
from flymyai.utils.utils import retryable_callback


class BaseSyncClient(BaseClient[httpx.Client]):
    def _construct_client(self):
        return httpx.Client(
            http2=True,
            headers=self.client_info.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://api.flymy.ai/"),
            timeout=_predict_timeout,
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
            response = SSEInferenceResponseFactory(
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
            raise FlyMyAIPredictException.from_base_exception(e)

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

    def predict_async_task(
        self, payload: dict, model: Optional[str] = None, max_retries=None
    ):
        payload = MultipartPayload(input_data=payload)
        client_info = self.amend_client_info(model)
        try:
            _, response = retryable_callback(
                lambda: self._client.post(
                    client_info.prediction_async_path, **payload.serialize()
                ),
                max_retries or self.max_retries,
                FlyMyAIAsyncTaskException,
                FlyMyAIExceptionGroup,
            )
            response = SSEInferenceResponseFactory(response).construct()
            return self._async_prediction_task_construct(response, client_info)
        except BaseFlyMyAIException as e:
            raise FlyMyAIAsyncTaskException.from_base_exception(e)

    def prediction_task_result(
        self, prediction_task: AsyncPredictionTask, timeout: Optional[float] = None
    ):
        prediction_id = prediction_task.prediction_id

        def get_res():
            resp = self._client.get(
                url=(
                    prediction_task.client_info or self.client_info
                ).prediction_result_path,
                params={"request_id": prediction_id},
            )
            return self._construct_task_result(resp)

        _, res = retryable_callback(
            lambda: get_res(),
            None,
            FlyMyAIAsyncTaskException,
            FlyMyAIExceptionGroup,
            timeout,
            0.5,
        )

        return res

    def _stream(self, client_info: APIKeyClientInfo, payload: dict):
        payload = MultipartPayload(payload)
        response_iterator = self._stream_iterator(
            client_info, payload, is_long_stream=True
        )
        decoder = SSEDecoder()
        with response_iterator as sse_stream:
            for sse_partial in decoder.iter(sse_stream.iter_lines()):
                try:
                    response = SSEInferenceResponseFactory(
                        sse=sse_partial,
                        httpx_request=sse_stream.request,
                        httpx_response=sse_stream,
                    ).construct()
                except BaseFlyMyAIException as e:
                    raise FlyMyAIPredictException.from_base_exception(e)
                yield response

    def stream(self, payload: dict, model: Optional[str] = None):
        full_client_info = self.amend_client_info(model)
        stream_iter = self._stream(full_client_info, payload)
        stream_wrapper = PredictionStream(stream_iter, self, full_client_info)
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

    def cancel_prediction(
        self,
        prediction_id: str,
        model: Optional[str] = None,
        client_info: APIKeyClientInfo = None,
    ):
        if client_info:
            full_client_info = client_info
        else:
            full_client_info = self.amend_client_info(model)
        response = self._client.patch(
            url=full_client_info.prediction_cancel_path,
            json={"infer_id": prediction_id},
        )
        return SSEInferenceResponseFactory(
            httpx_response=response, httpx_request=response.request
        ).construct()

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
