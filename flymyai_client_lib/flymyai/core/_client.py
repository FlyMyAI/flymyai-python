import os
from typing import Callable, Awaitable, Generic, TypeVar, Union, overload

import httpx

from multipart.payload import MultipartPayload
from core.authorizations import APIKeyClientInfo, ClientInfoFactory
from core.exceptions import FlyMyAIPredictException, FlyMyAIExceptionGroup
from core.models import PredictionResponse, OpenAPISchemaResponse
from utils.utils import retryable_callback, aretryable_callback

DEFAULT_RETRY_COUNT = os.getenv("FLYMYAI_MAX_RETRIES", 2)

_PossibleClients = TypeVar(
    "_PossibleClients", bound=Union[httpx.Client, httpx.AsyncClient]
)


_predict_timeout = httpx.Timeout(None, connect=10)


class BaseClient(Generic[_PossibleClients]):
    _client: _PossibleClients
    max_retries: int
    client_info: APIKeyClientInfo

    def __init__(
        self, client_info: APIKeyClientInfo | dict, max_retries=DEFAULT_RETRY_COUNT
    ):
        if isinstance(client_info, dict):
            self.client_info = ClientInfoFactory(client_info).build_client_info()
        elif isinstance(client_info, APIKeyClientInfo):
            self.client_info = client_info
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
        try:
            return response.raise_for_status()
        except httpx.HTTPError:
            raise FlyMyAIPredictException.from_response(response)

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
            headers=self.client_info.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://flymy.ai/"),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()

    def _predict(self, payload: MultipartPayload):
        return self._wrap_request(
            lambda: self._client.post(
                self.client_info.prediction_path,
                **payload.serialize(),
                timeout=_predict_timeout,
                headers=self.client_info.authorization_headers,
            )
        )

    def predict(self, payload: dict, max_retries=None):
        payload = MultipartPayload(payload)
        history, response = retryable_callback(
            lambda: self._predict(payload),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse(history, response.json())

    def _openapi_schema(self):
        return self._wrap_request(
            lambda: self._client.get(
                self.client_info.openapi_schema_path,
                headers=self.client_info.authorization_headers,
            )
        )

    def openapi_schema(self, max_retries=None):
        history, response = retryable_callback(
            lambda: self._openapi_schema(),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse(history, response.json())

    @classmethod
    def run_predict(cls, client_info: dict, payload: dict):
        client_info = ClientInfoFactory(raw_client_info=client_info).build_client_info()
        with cls(client_info) as client:
            return client.predict(payload)


class BaseAsyncClient(BaseClient[httpx.AsyncClient]):
    def _construct_client(self):
        return httpx.AsyncClient(
            headers=self.client_info.authorization_headers,
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
        return OpenAPISchemaResponse(history, response.json())

    def _openapi_schema(self):
        return self._wrap_request(
            lambda: self._client.get(
                self.client_info.openapi_schema_path,
                headers=self.client_info.authorization_headers,
            )
        )

    def _predict(self, payload: MultipartPayload):
        return self._wrap_request(
            lambda: self._client.post(
                self.client_info.prediction_path,
                timeout=_predict_timeout,
                **payload.serialize(),
                headers=self.client_info.authorization_headers,
            )
        )

    async def predict(self, payload: dict, max_retries=None):
        payload = MultipartPayload(input_data=payload)
        history, response = await aretryable_callback(
            lambda: self._predict(payload),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse(history, response.json())

    @staticmethod
    async def _wrap_request(request_callback: Callable[..., Awaitable[httpx.Response]]):
        response = await request_callback()
        try:
            return response.raise_for_status()
        except httpx.HTTPStatusError:
            raise FlyMyAIPredictException.from_response(response)

    async def close(self):
        await self._client.aclose()

    @classmethod
    async def arun_predict(cls, client_info: dict, payload: dict):
        client_info = ClientInfoFactory(raw_client_info=client_info).build_client_info()
        async with cls(client_info) as client:
            return await client.predict(payload)
